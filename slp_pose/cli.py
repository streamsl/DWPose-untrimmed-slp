"""Command line: `python -m slp_pose <command>` (spec §4.2 cli.py).

Argument parsing imports no torch: each command imports what it needs, and the GPU commands set
CUDA_VISIBLE_DEVICES before torch is imported. Paths default to the dataset's registry entry
(settings.DATASETS). Commands with an output directory log to stdout and to
<out root>/.state/logs/<command>_<time>.log.

Commands and options (argparse; defaults in brackets):
  build-engines  --gpu N [0] --force
                 engines.build_engines(Settings(), cuda:0, force) on GPU N; prints engines.engine_summary.
  parity         --dataset [bobsl] --videos N [20] --windows N [3] --window-s S [30]
                 --full-videos N [2] --seed N [0] --gpu N [0] --out DIR [<out_root>_dev/parity]
                 --backend trt|torch [trt]; parity.run_parity writes the report JSON,
                 exit code 0 iff parity.gate passes.
  extract        --dataset [bobsl] --out-root PATH [dataset.out_root] --gpus 0,1 [0,1]
                 --videos ID... --limit N --backend trt|torch [trt] --allow-frame-mismatch
                 -> run.run_extract; exit code 0 iff every video was extracted or already done.
  derive         --dataset --out-root --videos ID... [all committed] --primary-rule R [dataset rule]
                 -> record.derive per video, then run.merge_meta (run.run_derive).
  check          --dataset --out-root --videos ID... [all committed]
                 -> record.check per video; prints problems and render commands for the suggested
                 windows; exit code 1 on problems.
  render         VIDEO_ID --dataset --out-root --start-s S [0] --duration-s S [30] | --full --persons
                 --out PATH [<out-root>/renders/<vid>_<start frame>.mp4, <vid>_full.mp4 with --full]
                 -> render.render_video.
  render-done    --dataset --out-root --vis-dir DIR [<out-root>_vis] --jobs N [3] --videos ID...
                 --limit N --follow --poll-s S [300] --no-persons
                 -> render_batch.render_committed (read-only on the output root; logs in <vis-dir>;
                 one run per vis dir); exit code 0 iff no video failed.
Exit codes: 0 success; 1 failure (failed videos, check problems, parity gate); 2 bad arguments;
130 interrupted.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .select import PRIMARY_RULES
from .settings import BACKENDS, DATASETS, DatasetSpec, Settings

log = logging.getLogger('slp_pose')
_SHOWN_WINDOWS = 5   # render suggestions printed per video by `check`


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser of every command (no heavy imports)."""
    parser = argparse.ArgumentParser(prog='python -m slp_pose',
                                     description='DWPose whole-body keypoints for untrimmed sign-language videos.')
    sub = parser.add_subparsers(dest='command', required=True, metavar='COMMAND')

    p = sub.add_parser('build-engines', help='export ONNX and build the TensorRT engines')
    p.add_argument('--gpu', type=int, default=0, help='GPU index in PCI bus order [0]')
    p.add_argument('--force', action='store_true', help='rebuild existing engines (never mid-corpus)')

    p = sub.add_parser('parity', help='parity gate vs the official per-image reference')
    _dataset_option(p)
    p.add_argument('--videos', type=int, default=20, help='random videos [20]')
    p.add_argument('--windows', type=int, default=3, help='random windows per video [3]')
    p.add_argument('--window-s', type=float, default=30.0, help='window length in seconds [30]')
    p.add_argument('--full-videos', type=int, default=2, help='whole videos compared [2]')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gpu', type=int, default=0, help='GPU index in PCI bus order [0]')
    p.add_argument('--out', type=Path, help='report directory [<dataset out_root>_dev/parity]')
    p.add_argument('--backend', choices=BACKENDS, default='trt')

    p = sub.add_parser('extract', help='extract a dataset (resumable)')
    _dataset_option(p)
    _out_root_option(p)
    p.add_argument('--gpus', type=_gpu_list, default=[0, 1], help='comma-separated GPU indices [0,1]')
    p.add_argument('--videos', nargs='+', metavar='ID', help='only these videos')
    p.add_argument('--limit', type=int, help='only the first N videos of the schedule')
    p.add_argument('--backend', choices=BACKENDS, default='trt')
    p.add_argument('--allow-frame-mismatch', action='store_true',
                   help='accept a decoded frame count that differs from ffprobe nb_frames')

    p = sub.add_parser('derive', help='re-derive counts, primary and poses on the CPU')
    _dataset_option(p)
    _out_root_option(p)
    p.add_argument('--videos', nargs='+', metavar='ID', help='[every committed video]')
    p.add_argument('--primary-rule', choices=sorted(PRIMARY_RULES), help="[the dataset's rule]")

    p = sub.add_parser('check', help='verify committed outputs and flag primary switches')
    _dataset_option(p)
    _out_root_option(p)
    p.add_argument('--videos', nargs='+', metavar='ID', help='[every committed video]')

    p = sub.add_parser('render', help='draw the saved keypoints over a video window')
    p.add_argument('video_id', metavar='VIDEO_ID')
    _dataset_option(p)
    _out_root_option(p)
    p.add_argument('--start-s', type=float, default=0.0, help='[0]')
    length = p.add_mutually_exclusive_group()
    length.add_argument('--duration-s', type=float, default=30.0, help='[30]')
    length.add_argument('--full', action='store_true', help='the whole video, frame 0 to the end')
    p.add_argument('--persons', action='store_true', help='also draw the other people and candidate boxes')
    p.add_argument('--out', type=Path, help='[<out-root>/renders/<vid>_<start frame>.mp4, <vid>_full.mp4 with --full]')

    p = sub.add_parser('render-done', help='render every committed video, full length, while extraction runs')
    _dataset_option(p)
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
    args = build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except KeyboardInterrupt:
        log.warning('interrupted')
        return 130


# --------------------------------------------------------------------------- commands
def _build_engines(args: argparse.Namespace) -> int:
    _pin_gpu(args.gpu)
    _setup_logging(None, 'build-engines')
    import torch
    from . import engines, env
    env.strict_fp32()
    env.check_environment(need_trt=True)
    paths = engines.build_engines(Settings(), torch.device('cuda', 0), force=args.force)
    print(json.dumps(engines.engine_summary(paths), indent=1, default=str))
    return 0


def _parity(args: argparse.Namespace) -> int:
    _pin_gpu(args.gpu)
    dataset = DATASETS[args.dataset]
    settings = Settings(backend=args.backend)
    out = (args.out or settings.repo_root / f'{dataset.out_root}_dev' / 'parity').resolve()
    _setup_logging(out, 'parity')
    from . import parity
    report = parity.run_parity(settings, dataset, num_videos=args.videos, windows=args.windows,
                               window_s=args.window_s, full_videos=args.full_videos, seed=args.seed, out_dir=out)
    passed, reasons = parity.gate(report)
    for reason in reasons:
        log.info('gate: %s', reason)
    log.info('parity gate %s (%s backend); report in %s', 'PASSED' if passed else 'FAILED', args.backend, out)
    return 0 if passed else 1


def _extract(args: argparse.Namespace) -> int:
    dataset, settings = DATASETS[args.dataset], Settings(backend=args.backend)
    root = _out_root(args, dataset, settings)
    _setup_logging(root, 'extract')
    from .run import run_extract
    summary = run_extract(dataset, settings, root, args.gpus, video_ids=args.videos, limit=args.limit,
                          allow_frame_mismatch=args.allow_frame_mismatch)
    return 1 if summary.failed or summary.pending else 0


def _derive(args: argparse.Namespace) -> int:
    dataset, settings = DATASETS[args.dataset], Settings()
    root = _out_root(args, dataset, settings)
    _setup_logging(root, 'derive')
    from .run import run_derive
    derived, failed = run_derive(root, settings, args.primary_rule or dataset.primary_rule, args.videos)
    log.info('derived %d videos, %d failed', len(derived), len(failed))
    return 1 if failed else 0


def _check(args: argparse.Namespace) -> int:
    dataset, settings = DATASETS[args.dataset], Settings()
    root = _out_root(args, dataset, settings)
    _setup_logging(root, 'check')
    from . import record
    from .run import committed_videos
    from .video import probe
    paths = _video_paths(dataset, settings)
    bad = 0
    for vid in args.videos or committed_videos(root):
        try:
            report = record.check(root, vid)
        except Exception as exc:
            log.error('%s: cannot check: %s: %s', vid, type(exc).__name__, exc)
            bad += 1
            continue
        if report.problems:
            bad += 1
            log.error('%s: %d problems', vid, len(report.problems))
            for problem in report.problems:
                log.error('  %s', problem)
        log.info('%s: %s, %d primary switches, %d windows to review', vid, 'OK' if report.ok else 'PROBLEMS',
                 len(report.primary_switches), len(report.suggested_windows))
        if report.suggested_windows and vid in paths:
            fps = probe(paths[vid], vid).fps
            for start, end in report.suggested_windows[:_SHOWN_WINDOWS]:
                log.info('  python -m slp_pose render %s --dataset %s --out-root %s --start-s %.2f --duration-s %.2f '
                         '--persons', vid, dataset.name, root, start / fps, (end - start) / fps)
    log.info('checked; %d videos with problems', bad)
    return 1 if bad else 0


def _render(args: argparse.Namespace) -> int:
    dataset, settings = DATASETS[args.dataset], Settings()
    root = _out_root(args, dataset, settings)
    _setup_logging(root, 'render')
    from .render import render_video
    from .video import probe
    paths = _video_paths(dataset, settings)
    if args.video_id not in paths:
        log.error('%s: no such video in %s', args.video_id, dataset.name)
        return 1
    if args.full and args.start_s:
        log.error('--full renders the whole video; it takes no --start-s')
        return 2
    video = probe(paths[args.video_id], args.video_id)
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
    dataset, settings = DATASETS[args.dataset], Settings()
    root = _out_root(args, dataset, settings)
    from .render_batch import check_dirs, default_vis_dir, render_committed
    vis_dir = (args.vis_dir or default_vis_dir(root)).resolve()
    try:
        check_dirs(root, vis_dir)
    except ValueError as exc:
        log.error('%s', exc)
        return 2
    _setup_logging(vis_dir, 'render-done')   # never into the output root
    summary = render_committed(root, vis_dir, _video_paths(dataset, settings), jobs=args.jobs,
                               video_ids=args.videos, limit=args.limit, follow=args.follow,
                               persons=args.persons, poll_s=args.poll_s)
    return 1 if summary.failed else 0


_COMMANDS: Dict[str, Callable[[argparse.Namespace], int]] = {
    'build-engines': _build_engines, 'parity': _parity, 'extract': _extract, 'derive': _derive,
    'check': _check, 'render': _render, 'render-done': _render_done,
}


# --------------------------------------------------------------------------- helpers
def _dataset_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--dataset', choices=sorted(DATASETS), default='bobsl', help='[bobsl]')


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


def _out_root(args: argparse.Namespace, dataset: DatasetSpec, settings: Settings) -> Path:
    return (args.out_root or dataset.output_root(settings.repo_root)).resolve()


def _video_paths(dataset: DatasetSpec, settings: Settings) -> Dict[str, Path]:
    return {dataset.video_id(path): path for path in dataset.video_paths(settings.repo_root)}


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
