"""BOBSL (BBC-Oxford British Sign Language): 2,212 BBC episodes with an in-vision interpreter.

Layout under the data root: BOBSL/original_data/videos/mp4/<episode id>.mp4 and the split lists
BOBSL/original_data/metadata/subset2episode.json ({split: [episode id, ...]}). The video id is the
file stem; outputs go to BOBSL/dwpose/.

Repaired sources (D26): an episode in REPAIRED is damaged in the official BOBSL v1.4 copy itself
(no download fixes it), so it is read from BOBSL/repaired/<episode id>.mp4 when that file exists,
else from the original. The repaired copy has the same frames, frame size and frame rate, with the
damaged frames painted black (lossless H.264; BOBSL/repaired/README.txt says how it was made and
checked). The extractor finds nobody on a black frame, so those frames are stored as empty (no
signer). Publish a repaired copy under a temporary name and rename it into place.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from . import register
from .base import Dataset, Video

# Episode id -> its damaged frames as half-open [start, stop) runs: every frame whose packets overlap
# the zero bytes in the source, through the next IDR keyframe.
REPAIRED: Dict[str, Tuple[Tuple[int, int], ...]] = {
    # sha256 30745d5a... (the same bytes in bobsl_v1_4_videos_mp4.tar): two 1,044,480-byte zero runs at
    # offsets 169,713,664 and 172,859,392; OpenCV stops at frame 71,837 of 90,577. 1,282 frames damaged.
    '5976836975695491321': ((71840, 72447), (73259, 73934)),
}


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
        return [Video(path.stem, self._repaired(path), splits.get(path.stem)) for path in paths]

    def _repaired(self, path: Path) -> Path:
        """The repaired copy of a REPAIRED episode when that file exists, else `path`."""
        repaired = self.data_root / 'BOBSL' / 'repaired' / path.name
        return repaired if path.stem in REPAIRED and repaired.is_file() else path

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
