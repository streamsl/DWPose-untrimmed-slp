"""Subtitles for misaligned-slt (spec D30): <out root>/subs/<video id>.<track>.vtt and caption_source.

misaligned-slt reads a language root as poses/<vid>.npy + video_meta.csv + subs/<vid>.<track>.vtt
(data.loader.best_subtitle picks one track per video by its subtitles.preferred_suffixes; '.en.vtt'
is the canonical name). A dataset that has subtitles implements Dataset.captions(video, fps) ->
{track: [Cue, ...]} (datasets/base.py) in its own file; the track names the file ('en' ->
<vid>.en.vtt, 'en-manual' -> <vid>.en-manual.vtt). `write_subs` then, for every committed video:
- writes each track as WebVTT (one cue per block, 'HH:MM:SS.mmm --> HH:MM:SS.mmm', the text on one
  line), atomically (a temp file, then os.replace), only when its bytes change;
- drops cues whose text is empty or whose end is not after their start (misaligned-slt's parse_vtt
  would drop them too) and counts them;
- deletes .vtt files under subs/ that no committed video's tracks produce any more;
- records each video's caption source in <out root>/.state/captions.json: Dataset.caption_source
  (e.g. 'human') for a video with at least one track, 'none' without; run.merge_meta copies it into
  video_meta.csv's caption_source column (blank for a video not in the file).
Cue text is written as the dataset gives it; misaligned-slt cleans it at load time
(data.loader.clean_caption_text), and a dataset normalizes only where its sources disagree.

`write_splits` writes a dataset's own split files into its output root (spec D32): splits.csv, or one
splits_<name>.csv per split the dataset declares (Dataset.splits; BOBSL: splits_automatic.csv and
splits_manual.csv, one per subtitle alignment). Each is a CSV video_id,split that misaligned-slt reads as
its splits.signverse_csv (data.loader.build_splits; 'val' = dev, other split names and absent videos
are left out). The datasets are separate benchmarks, so their splits are never merged.

`read_vtt` reads a source WebVTT file leniently (BOM, CRLF, NOTE / STYLE / REGION blocks, cue
identifiers, multi-line text joined with a space) for datasets that ship VTT subtitles.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    from .datasets.base import Dataset

log = logging.getLogger(__name__)

TRACK = re.compile(r'[A-Za-z0-9][A-Za-z0-9-]*')    # no dots: misaligned-slt splits '<vid>.<track>.vtt' at the first dot
CAPTION_SOURCES = ('human', 'mt', 'shard', 'none')   # misaligned-slt poses/pose_io.py caption_source values
_BLOCK_SEPARATOR = re.compile(r'\n[ \t]*\n')
_NON_CUE_BLOCK = re.compile(r'(NOTE|STYLE|REGION)([ \t]|$)')
_TIMESTAMP = re.compile(r'(?:(\d+):)?(\d{1,2}):(\d{2})(?:[.,](\d+))?$')   # any number of decimals (Auslan V2)
Seconds = Union[float, Fraction]


@dataclass(frozen=True)
class Cue:
    """One subtitle: [start, end) in seconds of the video's timeline, and its text."""

    start: Seconds
    end: Seconds
    text: str


@dataclass
class SubsSummary:
    """What write_subs did."""

    videos: int = 0                                       # committed videos looked at
    captioned: int = 0                                    # videos with at least one track
    files_written: int = 0                                # .vtt files whose bytes changed
    files_unchanged: int = 0
    files_removed: int = 0                                # stale .vtt files deleted
    cues: Dict[str, int] = field(default_factory=dict)    # track -> cues written
    dropped: Dict[str, int] = field(default_factory=dict)  # track -> cues dropped (empty text or end <= start)


def parse_timestamp(text: str) -> Fraction:
    """'HH:MM:SS.mmm' or 'MM:SS.mmm' -> seconds, exactly. Lenient: ',' for '.', and any number of
    decimals ('00:26:59.09' is 1619.09 s, as in some Auslan News V2 files)."""
    match = _TIMESTAMP.match(text.strip())
    if match is None:
        raise ValueError(f'not a WebVTT timestamp: {text!r}')
    hours, minutes, seconds, fraction = match.groups()
    whole = int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)
    return Fraction(whole) + (Fraction(int(fraction), 10 ** len(fraction)) if fraction else 0)


def read_vtt(path: Union[str, os.PathLike]) -> List[Cue]:
    """The cues of a WebVTT file, in file order (module docstring). ValueError for a file that is not
    WebVTT or has a block without a timing line."""
    raw = Path(path).read_bytes().decode('utf-8-sig').replace('\r\n', '\n').replace('\r', '\n')
    blocks = [block for block in _BLOCK_SEPARATOR.split(raw.strip('\n')) if block.strip()]
    if not blocks or not blocks[0].startswith('WEBVTT'):
        raise ValueError(f'{path}: not a WebVTT file')
    cues = []
    for i, block in enumerate(blocks):
        lines = block.split('\n')
        if i > 0 and _NON_CUE_BLOCK.match(lines[0]):
            continue
        timing = next((j for j, line in enumerate(lines) if '-->' in line), None)
        if timing is None and i == 0:
            continue                       # the header alone
        if timing is None or (i > 0 and timing > 1):
            raise ValueError(f'{path}: block without a timing line: {lines[:3]}')
        start, rest = lines[timing].split('-->', 1)
        end = rest.split()[0] if rest.split() else ''
        cues.append(Cue(parse_timestamp(start), parse_timestamp(end), ' '.join(lines[timing + 1:])))
    return cues


def to_millis(seconds: Seconds) -> int:
    """Seconds -> whole milliseconds, rounded half up (exact for Fractions and floats)."""
    millis = Fraction(seconds) * 1000 + Fraction(1, 2)
    return millis.numerator // millis.denominator


def format_timestamp(seconds: Seconds) -> str:
    """Seconds -> 'HH:MM:SS.mmm', rounded to whole milliseconds (half up)."""
    millis = to_millis(seconds)
    if millis < 0:
        raise ValueError(f'negative time {seconds}')
    hours, rest = divmod(millis, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    return f'{hours:02d}:{minutes:02d}:{rest // 1000:02d}.{rest % 1000:03d}'


def clean_cues(cues: Iterable[Cue]) -> Tuple[List[Cue], int]:
    """(cues to write, number dropped): whitespace collapsed to single spaces ('-->' in a text would
    read as a timing line, so it becomes '->'); a cue is dropped when its text is then empty or its
    end, in whole milliseconds, is not after its start."""
    out, dropped = [], 0
    for cue in cues:
        text = ' '.join(str(cue.text).split()).replace('-->', '->')
        if not text or to_millis(cue.end) <= to_millis(cue.start):
            dropped += 1
            continue
        out.append(Cue(cue.start, cue.end, text))
    return out, dropped


def vtt_text(cues: Sequence[Cue]) -> str:
    """The WebVTT file of cleaned cues (clean_cues), in the given order."""
    blocks = [f'{format_timestamp(c.start)} --> {format_timestamp(c.end)}\n{c.text}\n' for c in cues]
    return 'WEBVTT\n\n' + '\n'.join(blocks)


def subs_path(out_root: Path, video_id: str, track: str) -> Path:
    return Path(out_root) / 'subs' / f'{video_id}.{track}.vtt'


def captions_json(out_root: Path) -> Path:
    """<out root>/.state/captions.json: {video id: caption source} (run.merge_meta reads it)."""
    return Path(out_root) / '.state' / 'captions.json'


def load_caption_sources(out_root: Path) -> Dict[str, str]:
    """{video id: caption source} from .state/captions.json; {} when there is none."""
    path = captions_json(out_root)
    if not path.is_file():
        return {}
    return {str(k): str(v) for k, v in json.loads(path.read_text(encoding='utf-8')).items()}


def _write_if_changed(path: Path, text: str) -> bool:
    """Write `text` (utf-8, '\\n') to `path` atomically unless it already holds those bytes."""
    data = text.encode('utf-8')
    try:
        if path.read_bytes() == data:
            return False
    except FileNotFoundError:
        pass
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return True


def write_subs(dataset: 'Dataset', out_root: Path, videos: Optional[Sequence] = None) -> SubsSummary:
    """Write subs/ and .state/captions.json for the committed videos of `dataset` (module docstring).
    `videos` defaults to load_videos(dataset); only videos with a done marker under `out_root` are
    written, and the subs/ files of other videos are removed. ValueError for a bad track name or a
    caption source misaligned-slt does not know."""
    from .datasets.base import load_videos
    from .record import load_persons_fps
    from .run import committed_videos
    out_root = Path(out_root)
    source = dataset.caption_source
    if source not in CAPTION_SOURCES:
        raise ValueError(f'{dataset.name}: caption_source {source!r} is not one of {CAPTION_SOURCES}')
    committed = set(committed_videos(out_root))
    todo = [v for v in (load_videos(dataset) if videos is None else videos) if v.video_id in committed]
    (out_root / 'subs').mkdir(parents=True, exist_ok=True)
    summary, sources, keep = SubsSummary(videos=len(todo)), {}, set()
    for video in todo:
        fps = load_persons_fps(out_root / 'persons' / f'{video.video_id}.npz')
        tracks = dataset.captions(video, fps) or {}
        written = 0
        for track, cues in sorted(tracks.items()):
            if not TRACK.fullmatch(track):
                raise ValueError(f'{dataset.name}: {video.video_id}: track name {track!r} must match {TRACK.pattern}')
            kept, dropped = clean_cues(cues)
            summary.dropped[track] = summary.dropped.get(track, 0) + dropped
            if not kept:
                continue
            path = subs_path(out_root, video.video_id, track)
            keep.add(path.name)
            if _write_if_changed(path, vtt_text(kept)):
                summary.files_written += 1
            else:
                summary.files_unchanged += 1
            summary.cues[track] = summary.cues.get(track, 0) + len(kept)
            written += 1
        sources[video.video_id] = source if written else 'none'
        summary.captioned += bool(written)
    for path in (out_root / 'subs').glob('*.vtt'):
        if path.name not in keep:
            path.unlink()
            summary.files_removed += 1
    state = captions_json(out_root)
    state.parent.mkdir(parents=True, exist_ok=True)
    _write_if_changed(state, json.dumps(dict(sorted(sources.items())), indent=0) + '\n')
    return summary


def caption_source_rows(rows: Mapping[str, Mapping[str, object]], sources: Mapping[str, str]) -> Dict[str, dict]:
    """`rows` (video_meta rows by video id) with caption_source set from `sources` where present."""
    return {vid: dict(row, caption_source=sources.get(vid, row.get('caption_source'))) for vid, row in rows.items()}


def write_splits(dataset: 'Dataset', out_root: Path, videos: Optional[Sequence] = None) -> Dict[str, Dict[str, int]]:
    """Write the split files of `dataset` (Dataset.splits of `videos`, default load_videos(dataset)) into
    `out_root`: header video_id,split and one row per video of the split, in the dataset's video order,
    atomically and only when the bytes change; delete the other splits*.csv files there. Returns
    {file name: {split: videos}}. ValueError for a split name that is not [A-Za-z0-9-]+."""
    import csv
    import io
    from .datasets.base import load_videos
    videos = load_videos(dataset) if videos is None else videos
    order = {video.video_id: i for i, video in enumerate(videos)}
    out_root = Path(out_root)
    counts: Dict[str, Dict[str, int]] = {}
    for name, splits in sorted(dataset.splits(videos).items()):
        if name and not TRACK.fullmatch(name):
            raise ValueError(f'{dataset.name}: split name {name!r} must match {TRACK.pattern}')
        file_name = f'splits_{name}.csv' if name else 'splits.csv'
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator='\n')
        writer.writerow(('video_id', 'split'))
        writer.writerows((vid, splits[vid]) for vid in sorted(splits, key=lambda vid: order[vid]))
        out_root.mkdir(parents=True, exist_ok=True)
        _write_if_changed(out_root / file_name, buffer.getvalue())
        per_split: Dict[str, int] = {}
        for split in splits.values():
            per_split[split] = per_split.get(split, 0) + 1
        counts[file_name] = per_split
    for path in out_root.glob('splits*.csv'):
        if path.name not in counts:
            path.unlink()
    return counts
