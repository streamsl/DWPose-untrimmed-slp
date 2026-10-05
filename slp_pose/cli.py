"""Command line: `slp-pose <command>` or `python -m slp_pose <command>` (spec §4.2 cli.py).

Argument parsing imports no torch: each command imports what it needs, and the GPU commands set
CUDA_VISIBLE_DEVICES before torch is imported. Dataset commands take --dataset NAME_OR_FILE (a
registered or installed dataset, or a dataset .py file: slp_pose.datasets) and --data-root
(paths.data_root); every command takes --models-dir (paths.models_dir). Output roots default to
the dataset's out_root(). Commands with an output directory log to stdout and to
<out root>/.state/logs/<command>_<time>.log.

Commands and options (argparse; defaults in brackets):
  fetch-models   --from PATH... --no-download
                 fetch.fetch_models into the models directory (both downloaded, or copied from --from).
  build-engines  --gpu N [0] --force
                 engines.build_engines(Settings(), cuda:0, force) on GPU N; prints engines.engine_summary.
  list-videos    --dataset --data-root --list
                 datasets.load_videos: checks the dataset's contract, prints counts (and every video).
  parity         --dataset [bobsl] --videos N [20] --windows N [3] --window-s S [30]
                 --full-videos N [2] --seed N [0] --gpu N [0] --out DIR [<out_root>_dev/parity]
                 --backend trt|torch [trt] --jitter batched_det tf32 (batched_det sets the jitter-relative limits, D17);
                 parity.run_parity writes the report JSON,
                 exit code 0 iff parity.gate passes.
  extract        --dataset [bobsl] --out-root PATH [dataset.out_root()] --gpus 0,1 [0,1]
                 --videos ID... --limit N --backend trt|torch [trt] --allow-frame-mismatch
                 -> run.run_extract (workers use the dataset's extraction rule, the parent derives
                 its primary rule); exit code 0 iff every video was extracted or already done.
  derive         --dataset --out-root --videos ID... [all committed] --primary-rule R [dataset rule;
                 per-frame or video-level, the framework's or the dataset's own: Dataset.rule]
                 -> record.derive per video (the current meta row, D19), then run.merge_meta (run.run_derive).
  check          --dataset --out-root --videos ID... [all committed] --no-crc
                 -> record.check per video (rules resolved by the dataset; the CRC-32 of every persons/
                 member unless --no-crc), plus record.derivation_problems
                 (outputs not derived with the dataset's primary rule, count settings and current meta definitions);
                 prints problems and render commands for the suggested windows; exit code 1 on problems.
  render         VIDEO_ID --dataset --out-root --start-s S [0] --duration-s S [30] | --full --persons
                 --out PATH [<out-root>/renders/<vid>_<start frame>.mp4, <vid>_full.mp4 with --full]
                 -> render.render_video.
  render-done    --dataset --out-root --vis-dir DIR [<out-root>_vis] --jobs N [3] --videos ID...
                 --limit N --follow --poll-s S [300] --no-persons
                 -> render_batch.render_committed (read-only on the output root; logs in <vis-dir>;
                 one run per vis dir); exit code 0 iff no video failed or was left unchecked mid-commit.
Exit codes: 0 success; 1 failure (failed videos, check problems, parity gate, missing models);
2 bad arguments (incl. an unknown or invalid dataset); 130 interrupted.
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from . import paths
from .datasets import Dataset, DatasetError, load_videos, resolve
from .env import configure_process
from .settings import BACKENDS, Settings

log = logging.getLogger('slp_pose')
_SHOWN_WINDOWS = 5   # render suggestions printed per video by `check`


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser of every command (no heavy imports)."""
    parser = argparse.ArgumentParser(prog='slp-pose',
                                     description='DWPose whole-body keypoints for untrimmed sign-language videos.')
    sub = parser.add_subparsers(dest='command', required=True, metavar='COMMAND')
    models = argparse.ArgumentParser(add_help=False)
    models.add_argument('--models-dir', type=Path,
                        help=f'checkpoints, ONNX and engines [${paths.MODELS_ENV}, <checkout>/models, '
                             f'~/.cache/slp_pose/models]')

    def command(name: str, help_text: str, dataset: bool = True) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text, parents=[models])
        if dataset:
            _dataset_options(p)
        return p

    p = command('fetch-models', 'download / copy the checkpoints into the models directory', dataset=False)
    p.add_argument('--from', dest='sources', action='append', type=Path, default=[], metavar='PATH',
                   help='file or directory to copy checkpoints from instead of downloading (repeatable)')
    p.add_argument('--no-download', dest='download', action='store_false', help='never use the network')

    p = command('build-engines', 'export ONNX and build the TensorRT engines', dataset=False)
    p.add_argument('--gpu', type=int, default=0, help='GPU index in PCI bus order [0]')
    p.add_argument('--force', action='store_true', help='rebuild existing engines (never mid-corpus)')

    p = command('list-videos', "check a dataset's videos and print a summary")
    p.add_argument('--list', action='store_true', help='also print every video (id, split, path)')

    p = command('parity', 'parity gate vs the official per-image reference')
    p.add_argument('--videos', type=int, default=20, help='random videos [20]')
    p.add_argument('--windows', type=int, default=3, help='random windows per video [3]')
    p.add_argument('--window-s', type=float, default=30.0, help='window length in seconds [30]')
    p.add_argument('--full-videos', type=int, default=2, help='whole videos compared [2]')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu', type=int, default=0, help='GPU index in PCI bus order [0]')
    p.add_argument('--out', type=Path, help='report directory [<dataset out_root>_dev/parity]')
    p.add_argument('--backend', choices=BACKENDS, default='trt')
    p.add_argument('--jitter', nargs='+', default=(), choices=('batched_det', 'tf32'), metavar='VARIANT',
                   help="also measure how much the official path deviates from itself; batched_det raises "
                        "the keypoint limits to 1.5x that noise, tf32 is only reported")

    p = command('extract', 'extract a dataset (resumable)')
    _out_root_option(p)
    p.add_argument('--gpus', type=_gpu_list, default=[0, 1], help='comma-separated GPU indices [0,1]')
    p.add_argument('--videos', nargs='+', metavar='ID', help='only these videos')
    p.add_argument('--limit', type=int, help='only the first N videos of the schedule')
    p.add_argument('--backend', choices=BACKENDS, default='trt')
    p.add_argument('--allow-frame-mismatch', action='store_true',
                   help='accept a decoded frame count that differs from ffprobe nb_frames')

    p = command('derive', 're-derive counts, primary, poses and the meta row on the CPU')
    _out_root_option(p)
    p.add_argument('--videos', nargs='+', metavar='ID', help='[every committed video]')
    p.add_argument('--primary-rule', metavar='RULE',
                   help="a per-frame or video-level rule, the framework's or the dataset's own [the dataset's "
                        "primary rule]")

    p = command('check', 'verify committed outputs and flag primary switches')
    _out_root_option(p)
    p.add_argument('--videos', nargs='+', metavar='ID', help='[every committed video]')
    p.add_argument('--no-crc', dest='verify_crc', action='store_false',
                   help='skip the CRC-32 check of persons/ (one sequential read of each file)')

    p = command('render', 'draw the saved keypoints over a video window')
    p.add_argument('video_id', metavar='VIDEO_ID')
    _out_root_option(p)
    p.add_argument('--start-s', type=float, default=0.0, help='[0]')
    length = p.add_mutually_exclusive_group()
    length.add_argument('--duration-s', type=float, default=30.0, help='[30]')
    length.add_argument('--full', action='store_true', help='the whole video, frame 0 to the end')
    p.add_argument('--persons', action='store_true', help='also draw the other people and candidate boxes')
    p.add_argument('--out', type=Path, help='[<out-root>/renders/<vid>_<start frame>.mp4, <vid>_full.mp4 with --full]')

    p = command('render-done', 'render every committed video, full length, while extraction runs')
    _out_root_option(p)
    p.add_argument('--vis-dir', type=Path, help='output directory [<out-root>_vis next to the output root]')
    p.add_argument('--jobs', type=_positive(int), default=3, help='render processes at a time, at nice +10 [3]')
    p.add_argument('--videos', nargs='+', metavar='ID', help='[every committed video]')
    p.add_argument('--limit', type=int, help='render at most N videos')
    p.add_argument('--follow', action='store_true',
                   help='keep rendering new commits until the extraction has stopped and nothing is left')
    p.add_argument('--poll-s', type=_positive(float), default=300.0,
                   help='with --follow: seconds between looks for new commits [300]')
    p.add_argument('--no-persons', dest='persons', action='store_false',
                   help='draw the primary signer only (default: every person and candidate box)')
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse `argv` (default sys.argv[1:]), run the command, return the process exit code."""
    configure_process()   # the console script does not go through __main__
    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except DatasetError as exc:
        print(f'slp-pose {args.command}: error: {exc}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        log.warning('interrupted')
        return 130


# --------------------------------------------------------------------------- commands
def _fetch_models(args: argparse.Namespace) -> int:
    _setup_logging(None, 'fetch-models')
    from .fetch import FetchError, fetch_models
    settings = _settings(args)
    try:
        done = fetch_models(settings, args.sources, download=args.download)
    except FetchError as exc:
        log.error('%s', exc)
        return 1
    for path, how in done:
        log.info('%-10s %s', how, path)
    log.info('models directory %s; next: build-engines', settings.models_dir)
    return 0


def _build_engines(args: argparse.Namespace) -> int:
    _pin_gpu(args.gpu)
    _setup_logging(None, 'build-engines')
    import torch
    from . import engines, env
    env.strict_fp32()
    env.check_environment(need_trt=True)
    built = engines.build_engines(_settings(args), torch.device('cuda', 0), force=args.force)
    print(json.dumps(engines.engine_summary(built), indent=1, default=str))
    return 0


def _list_videos(args: argparse.Namespace) -> int:
    dataset = _dataset(args)
    videos = load_videos(dataset)
    splits = collections.Counter(video.split for video in videos)
    print(f'{dataset!r}: {len(videos)} videos -> {dataset.out_root()}')
    print(f'  rules: extraction {dataset.extraction_rule}, primary {dataset.primary_rule}')
    counts = sorted(splits.items(), key=lambda item: (item[0] is None, item[0] or ''))
    print('  splits: ' + ', '.join(f'{split or "(none)"} {n}' for split, n in counts))
    if args.list:
        for video in videos:
            print(f'{video.video_id}\t{video.split or ""}\t{video.path}')
    return 0


def _parity(args: argparse.Namespace) -> int:
    _pin_gpu(args.gpu)
    dataset = _dataset(args)
    settings = _settings(args, backend=args.backend)
    default_root = dataset.out_root()
    out = (args.out or default_root.with_name(default_root.name + '_dev') / 'parity').resolve()
    _setup_logging(out, 'parity')
    from . import parity
    report = parity.run_parity(settings, dataset, num_videos=args.videos, windows=args.windows,
                               window_s=args.window_s, full_videos=args.full_videos, seed=args.seed, out_dir=out,
                               jitter=tuple(args.jitter))
    passed, reasons = parity.gate(report)
    for reason in reasons:
        log.info('gate: %s', reason)
    log.info('parity gate %s (%s backend); report in %s', 'PASSED' if passed else 'FAILED', args.backend, out)
    return 0 if passed else 1


def _extract(args: argparse.Namespace) -> int:
    dataset, settings = _dataset(args), _settings(args, backend=args.backend)
    root = _out_root(args, dataset)
    _setup_logging(root, 'extract')
    from .run import run_extract
    summary = run_extract(dataset, settings, root, args.gpus, video_ids=args.videos, limit=args.limit,
                          allow_frame_mismatch=args.allow_frame_mismatch)
    return 1 if summary.failed or summary.pending else 0


def _derive(args: argparse.Namespace) -> int:
    dataset, settings = _dataset(args), _settings(args)
    rule = _rule(dataset, args.primary_rule or dataset.primary_rule)   # before anything is written
    root = _out_root(args, dataset)
    _setup_logging(root, 'derive')
    from .run import run_derive
    derived, failed = run_derive(root, settings, rule, args.videos)
    log.info('derived %d videos, %d failed', len(derived), len(failed))
    return 1 if failed else 0


def _check(args: argparse.Namespace) -> int:
    dataset = _dataset(args)
    root = _out_root(args, dataset)
    _setup_logging(root, 'check')
    from . import record
    from .run import committed_videos
    from .video import probe
    sources = _video_paths(dataset)
    settings = _settings(args)
    bad = 0
    for vid in args.videos or committed_videos(root):
        try:
            report = record.check(root, vid, dataset=dataset, verify_crc=args.verify_crc)
            problems = report.problems + record.derivation_problems(root, vid, settings,
                                                                    dataset.rule(dataset.primary_rule))
        except Exception as exc:
            log.error('%s: cannot check: %s: %s', vid, type(exc).__name__, exc)
            bad += 1
            continue
        if problems:
            bad += 1
            log.error('%s: %d problems', vid, len(problems))
            for problem in problems:
                log.error('  %s', problem)
        log.info('%s: %s, %d primary switches, %d windows to review', vid, 'PROBLEMS' if problems else 'OK',
                 len(report.primary_switches), len(report.suggested_windows))
        if report.suggested_windows and vid in sources:
            fps = probe(sources[vid], vid).fps
            for start, end in report.suggested_windows[:_SHOWN_WINDOWS]:
                log.info('  python -m slp_pose render %s --dataset %s --data-root %s --out-root %s --start-s %.2f '
                         '--duration-s %.2f --persons', vid, args.dataset, dataset.data_root, root, start / fps,
                         (end - start) / fps)
    log.info('checked; %d videos with problems', bad)
    return 1 if bad else 0


def _render(args: argparse.Namespace) -> int:
    dataset = _dataset(args)
    root = _out_root(args, dataset)
    _setup_logging(root, 'render')
    from .render import render_video
    from .video import probe
    sources = _video_paths(dataset)
    if args.video_id not in sources:
        log.error('%s: no such video in %s', args.video_id, dataset.name)
        return 1
    if args.full and args.start_s:
        log.error('--full renders the whole video; it takes no --start-s')
        return 2
    video = probe(sources[args.video_id], args.video_id)
    if args.full:
        start, count, name = 0, None, f'{video.video_id}_full.mp4'
    else:
        start = round(args.start_s * video.fps_num / video.fps_den)
        count = round(args.duration_s * video.fps_num / video.fps_den)
        name = f'{video.video_id}_{start}.mp4'
    out = args.out or root / 'renders' / name
    from .render_batch import sigterm_as_interrupt
    with sigterm_as_interrupt():   # a killed render removes its .part file and stops its ffmpeg
        log.info('wrote %s', render_video(root, video, out, start, count, persons=args.persons))
    return 0


def _render_done(args: argparse.Namespace) -> int:
    dataset = _dataset(args)
    root = _out_root(args, dataset)
    from .render_batch import check_dirs, default_vis_dir, render_committed
    vis_dir = (args.vis_dir or default_vis_dir(root)).resolve()
    try:
        check_dirs(root, vis_dir)
    except ValueError as exc:
        log.error('%s', exc)
        return 2
    _setup_logging(vis_dir, 'render-done')   # never into the output root
    summary = render_committed(root, vis_dir, _video_paths(dataset), jobs=args.jobs,
                               video_ids=args.videos, limit=args.limit, follow=args.follow,
                               persons=args.persons, poll_s=args.poll_s)
    return 1 if summary.failed or summary.pending else 0


_COMMANDS: Dict[str, Callable[[argparse.Namespace], int]] = {
    'fetch-models': _fetch_models, 'build-engines': _build_engines, 'list-videos': _list_videos,
    'parity': _parity, 'extract': _extract, 'derive': _derive, 'check': _check, 'render': _render,
    'render-done': _render_done,
}


# --------------------------------------------------------------------------- helpers
def _dataset_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--dataset', default='bobsl', metavar='NAME_OR_FILE',
                        help='a registered or installed dataset name, or a dataset .py file [bobsl]')
    parser.add_argument('--data-root', type=Path, help=f'[${paths.DATA_ENV}, else ./data]')


def _out_root_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--out-root', type=Path, help="output root [the dataset's out_root]")


def _gpu_list(text: str) -> List[int]:
    try:
        gpus = [int(g) for g in text.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f'expected comma-separated GPU indices, got {text!r}') from None
    if len(set(gpus)) != len(gpus):
        raise argparse.ArgumentTypeError(f'GPU indices must be distinct, got {text!r}')
    return gpus


def _positive(kind: Callable[[str], float]) -> Callable[[str], float]:
    """argparse type: `kind` of the text, which must be > 0."""
    def parse(text: str) -> float:
        value = kind(text)
        if value <= 0:
            raise argparse.ArgumentTypeError(f'must be > 0, got {text!r}')
        return value
    return parse


def _dataset(args: argparse.Namespace) -> Dataset:
    return resolve(args.dataset, paths.data_root(args.data_root))


def _rule(dataset: Dataset, name: str):
    """dataset.rule(name); an unknown name is a DatasetError (exit code 2)."""
    try:
        return dataset.rule(name)
    except KeyError as exc:
        raise DatasetError(exc.args[0]) from None


def _settings(args: argparse.Namespace, **changes) -> Settings:
    return Settings(models_root=paths.models_dir(args.models_dir), **changes)


def _out_root(args: argparse.Namespace, dataset: Dataset) -> Path:
    return Path(args.out_root or dataset.out_root()).resolve()


def _video_paths(dataset: Dataset) -> Dict[str, Path]:
    return {video.video_id: Path(video.path) for video in load_videos(dataset)}


def _pin_gpu(gpu: int) -> None:
    """Make physical GPU `gpu` the only visible one; must happen before CUDA starts."""
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        raise RuntimeError('CUDA is already initialised in this process; cannot select the GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


def _setup_logging(root: Optional[Path], command: str) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if root is not None:
        logs = Path(root) / '.state' / 'logs'
        logs.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logs / f'{command}_{time.strftime("%Y%m%d-%H%M%S")}.log'))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', handlers=handlers, force=True)
