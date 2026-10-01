"""The dataset plugin contract: a Dataset lists its Videos; the framework checks and indexes them.

A dataset is one Python file holding one Dataset subclass (README 'Adding a dataset',
examples/my_dataset.py). The subclass sets a `name`, optionally the signer rules and a split
order, and implements `videos()`. Everything else (checking ids, files, splits and rules,
scheduling, the video index, extraction, derive, check, render) is the framework's job.

Imports only the rule registries (select, signer: numpy), never torch.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..select import PRIMARY_RULES
from ..signer import VIDEO_RULES

SAFE_ID = re.compile(r'[A-Za-z0-9._-]+')
SEGMENT_SUFFIX = re.compile(r'_segment_\d+$')   # misaligned-slt names clips <video_id>_segment_<n>
INDEX_COLUMNS = ('video_id', 'split', 'source')  # fixed leading columns of <out root>/video_ids.csv
_SHOWN_PROBLEMS = 10


class DatasetError(ValueError):
    """A dataset definition or its video list breaks the plugin contract."""


@dataclass(frozen=True)
class Video:
    """One source video of a dataset.

    Attributes:
        video_id: output file stem (poses/<video_id>.npy): [A-Za-z0-9._-]+, unique within the
            dataset, not ending in '_segment_<n>' (misaligned-slt's clip suffix). safe_id() helps.
        path: the video file.
        split: e.g. 'train', or None. Scheduling follows Dataset.split_order.
        info: extra str columns for <out root>/video_ids.csv (e.g. the name in the annotations).
    """

    video_id: str
    path: Path
    split: Optional[str] = None
    info: Mapping[str, str] = field(default_factory=dict)


class Dataset:
    """Base class of every dataset. Subclass it, set the class attributes, implement videos().

    Class attributes:
        name: registry and log name, [A-Za-z0-9._-]+; the default output root uses it.
        primary_rule: the final signer rule behind poses/: per-frame (select.PRIMARY_RULES) or
            video-level (signer.VIDEO_RULES, applied on the CPU right after each commit).
        extraction_rule: the per-frame rule the GPU workers select and pose with; None means
            primary_rule, which must then be per-frame. A video-level primary rule chooses among
            the posed people, so pick an extraction rule that poses the signer.
        split_order: when set, splits are extracted in this order (videos without a split last)
            and every Video.split must be one of them or None.

    The framework constructs it as `cls(data_root)`; an overriding __init__ must call
    super().__init__(data_root). After construction `extraction_rule` is never None.
    """

    name: str = ''
    primary_rule: str = 'largest_bbox'
    extraction_rule: Optional[str] = None
    split_order: Tuple[str, ...] = ()

    def __init__(self, data_root: os.PathLike) -> None:
        check_dataset_class(type(self))
        self.data_root = Path(data_root)
        self.extraction_rule = self.extraction_rule or self.primary_rule

    def videos(self) -> Iterable[Video]:
        """Every video of the dataset (any order); the framework validates the list."""
        raise NotImplementedError(f'{type(self).__name__}.videos()')

    def out_root(self) -> Path:
        """Default output root: <data root>/<name>/dwpose (override to put it elsewhere)."""
        return self.data_root / self.name / 'dwpose'

    def __repr__(self) -> str:
        return f'{type(self).__name__}(name={self.name!r}, data_root={str(self.data_root)!r})'


def safe_id(text: str) -> str:
    """`text` with every run of characters outside [A-Za-z0-9._-] replaced by '_' (for ids made
    from file names with spaces); the result can still collide, which load_videos reports."""
    return re.sub(r'[^A-Za-z0-9._-]+', '_', text).strip('_')


def check_dataset_class(cls: type) -> None:
    """Raise DatasetError unless `cls` is a concrete Dataset subclass with a valid name, rules and
    split order."""
    if not (isinstance(cls, type) and issubclass(cls, Dataset)) or cls is Dataset:
        raise DatasetError(f'{cls!r} is not a subclass of slp_pose.datasets.Dataset')
    where = f'{cls.__module__}.{cls.__qualname__}'
    if not isinstance(cls.name, str) or not SAFE_ID.fullmatch(cls.name):
        raise DatasetError(f'{where}: name {cls.name!r} must match {SAFE_ID.pattern}')
    if cls.primary_rule not in PRIMARY_RULES and cls.primary_rule not in VIDEO_RULES:
        raise DatasetError(f'{cls.name}: unknown primary_rule {cls.primary_rule!r}; per-frame rules '
                           f'{sorted(PRIMARY_RULES)}, video-level rules {sorted(VIDEO_RULES)}')
    posing = cls.extraction_rule or cls.primary_rule
    if posing not in PRIMARY_RULES:
        hint = ' (a video-level primary_rule needs a per-frame extraction_rule)' if cls.extraction_rule is None else ''
        raise DatasetError(f'{cls.name}: extraction rule {posing!r} is not a per-frame rule '
                           f'{sorted(PRIMARY_RULES)}{hint}')
    order = cls.split_order
    distinct = isinstance(order, tuple) and len(set(order)) == len(order)
    if not distinct or not all(isinstance(s, str) and s for s in order):
        raise DatasetError(f'{cls.name}: split_order must be a tuple of distinct non-empty str, got {order!r}')


def load_videos(dataset: Dataset) -> List[Video]:
    """dataset.videos() sorted by video_id, after checking the contract: at least one video, Video
    items, safe and unique ids, existing files, splits within split_order (when set), str info
    values that do not shadow the fixed index columns. Raises DatasetError listing the problems,
    or naming the path when videos() finds a source (e.g. its metadata) missing or unreadable;
    other OSErrors (I/O faults) propagate unchanged."""
    try:
        videos = list(dataset.videos())
    except (FileNotFoundError, NotADirectoryError, PermissionError) as exc:
        raise DatasetError(f'{dataset.name}: videos() cannot read its sources under the data root '
                           f'{dataset.data_root}: {exc}') from exc
    if not videos:
        raise DatasetError(f'{dataset.name}: videos() found no videos under the data root {dataset.data_root}')
    problems: List[str] = []
    seen: Dict[str, Path] = {}
    for video in videos:
        if not isinstance(video, Video):
            problems.append(f'{video!r} is not a Video')
            continue
        problems += _video_problems(dataset, video, seen)
    if problems:
        shown = '; '.join(problems[:_SHOWN_PROBLEMS])
        more = f' (and {len(problems) - _SHOWN_PROBLEMS} more)' if len(problems) > _SHOWN_PROBLEMS else ''
        raise DatasetError(f'{dataset.name}: {len(problems)} problem(s) in videos(): {shown}{more}')
    return sorted(videos, key=lambda v: v.video_id)


def _video_problems(dataset: Dataset, video: Video, seen: Dict[str, Path]) -> List[str]:
    vid, out = video.video_id, []
    if not isinstance(vid, str) or not SAFE_ID.fullmatch(vid) or SEGMENT_SUFFIX.search(vid):
        out.append(f'video id {vid!r} must match {SAFE_ID.pattern} and not end in _segment_<n>')
    elif vid in seen:
        out.append(f'video id {vid!r} is used by {seen[vid]} and {video.path}')
    else:
        seen[vid] = Path(video.path)
    if not Path(video.path).is_file():
        out.append(f'{vid}: no such file {video.path}')
    if video.split is not None and (not isinstance(video.split, str) or
                                    (dataset.split_order and video.split not in dataset.split_order)):
        out.append(f'{vid}: split {video.split!r} is not in split_order {dataset.split_order}')
    for key, value in video.info.items():
        if key in INDEX_COLUMNS or not isinstance(key, str) or not isinstance(value, str):
            out.append(f'{vid}: info {key!r}: {value!r} must map a str other than {INDEX_COLUMNS} to a str')
    return out


def schedule(videos: Iterable[Video], split_order: Sequence[str], size: Mapping[str, int]) -> List[Video]:
    """Extraction order: split_order (videos of other or no split last), then the largest file
    first (`size` = bytes per video id), then by video_id."""
    rank = {split: i for i, split in enumerate(split_order)}
    return sorted(videos, key=lambda v: (rank.get(v.split, len(rank)), -size[v.video_id], v.video_id))


def index_table(videos: Sequence[Video], data_root: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """(header, rows) of <out root>/video_ids.csv: INDEX_COLUMNS (split '' when None, source
    relative to `data_root` when inside it), then every info key in first-seen order ('' where a
    video lacks it)."""
    extra: Dict[str, None] = {}
    for video in videos:
        extra.update(dict.fromkeys(video.info))
    header = list(INDEX_COLUMNS) + list(extra)
    rows = [dict({key: '' for key in extra}, video_id=v.video_id, split=v.split or '',
                 source=_relative(Path(v.path), Path(data_root)), **v.info) for v in videos]
    return header, rows


def _relative(path: Path, root: Path) -> str:
    """`path` relative to `root` when it lies under it by name, else absolute. Symlinks are not
    followed, so a symlinked dataset folder or video keeps its path under the data root."""
    path, root = os.path.abspath(path), os.path.abspath(root)
    rel = os.path.relpath(path, root)
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        return Path(path).as_posix()
    return Path(rel).as_posix()
