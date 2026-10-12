"""BOBSL (BBC-Oxford British Sign Language): 2,212 BBC episodes with an in-vision interpreter.

Layout under the data root: BOBSL/original_data/videos/mp4/<episode id>.mp4 and the split lists
BOBSL/original_data/metadata/subset2episode.json ({split: [episode id, ...]}). The video id is the
file stem; outputs go to BOBSL/dwpose/.

Repaired sources (D26): an episode in REPAIRED is damaged in the official BOBSL v1.4 copy itself
(no download fixes it), so it is read from BOBSL/repaired/<episode id>.mp4 when that file exists,
else from the original. The repaired copy has the same frames, frame size and frame rate, with the
damaged frames painted black (lossless H.264; BOBSL/repaired/README.txt says how it was made and
checked). The extractor finds nobody on a black frame, so those frames are stored as empty (no
signer). Publish a repaired copy under a temporary name and rename it into place. An episode read
from its damaged original (no repaired copy) is declared damaged (Video.known_damage, D28): `check`
then expects decoder errors from it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from . import register
from .base import Dataset, Video

# Episode id -> the frames its repaired copy paints black, as half-open [start, stop) runs: every frame
# that depends on damaged data (its packets overlap the zero bytes, or it references a frame that does),
# through the next IDR keyframe.
REPAIRED: Dict[str, Tuple[Tuple[int, int], ...]] = {
    # sha256 30745d5a... (the same bytes in bobsl_v1_4_videos_mp4.tar): two 1,048,576-byte zero runs at
    # offsets 169,711,104 and 172,856,832; OpenCV stops at frame 71,837 of 90,577. 1,287 frames black in
    # the copy rebuilt for D28 (the first copy, D26, froze 71837-71839 and 73257-73258 on the frame before
    # instead and listed 1,282; BOBSL/repaired/README.txt says which copy is in place).
    '5976836975695491321': ((71837, 72447), (73257, 73934)),
}
_DAMAGED_ORIGINAL = ('damaged in the official BOBSL v1.4 archive (zero runs; bobsl.REPAIRED) and read from the '
                     'original, as BOBSL/repaired/ has no copy of it: libav reports decoding errors')


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
        out = []
        for path in paths:
            source = self._repaired(path)
            damaged = path.stem in REPAIRED and source == path
            out.append(Video(path.stem, source, splits.get(path.stem),
                             known_damage=_DAMAGED_ORIGINAL if damaged else None))
        return out

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
