"""BOBSL (BBC-Oxford British Sign Language): 2,212 BBC episodes with an in-vision interpreter.

Layout under the data root: BOBSL/original_data/videos/mp4/<episode id>.mp4 and the split lists
BOBSL/original_data/metadata/subset2episode.json ({split: [episode id, ...]}). The video id is the
file stem; outputs go to BOBSL/dwpose/.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from . import register
from .base import Dataset, Video


@register
class Bobsl(Dataset):
    """BOBSL episodes, scheduled val, test, train, challenge_test (spec assumption 2)."""

    name = 'bobsl'
    # D13: the interpreter's spot, empty frames when the interpreter is off screen; the parent derives
    # it on the CPU after each commit. Workers still pose with largest_bbox, so the extraction hash of
    # the videos committed before the switch is unchanged (they are re-derived, not re-extracted).
    primary_rule = 'signer_track'
    extraction_rule = 'largest_bbox'
    split_order = ('val', 'test', 'train', 'challenge_test')

    @property
    def source(self) -> Path:
        return self.data_root / 'BOBSL' / 'original_data'

    def out_root(self) -> Path:
        return self.data_root / 'BOBSL' / 'dwpose'

    def videos(self) -> List[Video]:
        splits = self._splits()
        paths = sorted((self.source / 'videos' / 'mp4').glob('*.mp4'))
        return [Video(path.stem, path, splits.get(path.stem)) for path in paths]

    def _splits(self) -> Dict[str, str]:
        """{episode id: split}; an episode listed twice keeps its first split in split_order."""
        with open(self.source / 'metadata' / 'subset2episode.json', encoding='utf-8') as f:
            raw = json.load(f)
        out: Dict[str, str] = {}
        rank = {split: i for i, split in enumerate(self.split_order)}
        for split in sorted(raw, key=lambda s: rank.get(s, len(rank))):
            for episode in raw[split]:
                out.setdefault(str(episode), split)
        return out
