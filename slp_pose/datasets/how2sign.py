"""How2Sign: American Sign Language instructional videos, one signer in a studio (frontal camera).

Layout under the data root: How2Sign/{train,val,test}/<video name>.mp4, the How2Sign "raw videos" of
the frontal RGB camera (e.g. --7E2sU6zP4-5-rgb_front.mp4), and the sentence timings
How2Sign/how2sign_realigned_{train,val,test}.csv (tab-separated; VIDEO_NAME = the file stem). The
split is the folder (train, val, test); the video id is How2Sign's own VIDEO_NAME, so the sentence
timings join on it; outputs go to How2Sign/dwpose/.

Frame rates differ between videos (24 fps for most; 23.976, 30, 50 and 60 for some) and are kept:
each video is stored at its own rate, like Auslan News. 3 of the 2,529 videos are 1920x1080, the
rest 1280x720. About 170 video ids start with '-': on the command line pass such an id as
`--videos=<id>` (one id per option), or select videos with --limit.

Subtitles: one track 'en' = the rows of how2sign_realigned_{train,val,test}.csv (TAB separated)
whose VIDEO_NAME is the video: SENTENCE from START_REALIGNED to END_REALIGNED (seconds), in sentence
number order (SENTENCE_ID '<VIDEO_ID>_<number>'). In a TRUNCATED video a sentence that starts after
the last decoded frame has no poses and is left out (one that straddles the end is kept; misaligned-
slt treats its end as unreliable).

Truncated sources: a video in TRUNCATED is cut short in the official How2Sign archive itself (the
archive's zip holds the same bytes, so no download fixes it): its container header states more frames
than its data holds. Its Video carries the frames that do decode (Video.expected_frames), and the GPU
worker requires exactly that count instead of ffprobe's nb_frames, so the normal run extracts it.
misaligned-slt reads the video's rate as T / duration_s = its native rate.

Damaged sources: decoding a video in DAMAGED prints libav errors on every read, because the
file itself is damaged. Its Video carries the note (Video.known_damage), so `check` logs those
decoder errors as expected instead of reporting them as a problem (a read fault would print others).
"""
from __future__ import annotations

import csv
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Tuple

from ..subs import Cue
from . import register
from .base import Dataset, Video

SPLITS = ('val', 'test', 'train')
REALIGNED = 'how2sign_realigned_{split}.csv'
REALIGNED_COLUMNS = ('VIDEO_ID', 'VIDEO_NAME', 'SENTENCE_ID', 'SENTENCE_NAME', 'START_REALIGNED', 'END_REALIGNED',
                     'SENTENCE')
# Video id -> the frames its truncated file decodes to (Video.expected_frames).
TRUNCATED: Dict[str, int] = {
    # train, 24 fps: the container lists 4,340 frames, the data ends after 4,088 (ffmpeg: "partial file"
    # at byte 212,911,030; the file has 212,910,080 bytes): the last 10.5 s are missing. Every sentence
    # of its realigned timings ends by 136.7 s, so no annotated sentence loses a frame.
    'cH27PiYX6gI-9-rgb_front': 4088,
    # train, 24 fps: the container lists 3,792 frames (158 s), the file holds 111,194,112 bytes and its data
    # ends inside frame 2,134 (ffmpeg: "partial file" at byte 111,204,767): only the first 88.9 s exist.
    # Its zip member has the same CRC (c2a53c27), so the archive itself is truncated. 13 of its 26
    # realigned sentences end by 88.35 s and are intact; 88.86-92.16 s is cut and the other 12 are lost.
    'CH7AviIr0-0-8-rgb_front': 2133,
}
# Video id -> why decoding it prints errors (Video.known_damage): damaged as downloaded, or truncated as
# distributed (TRUNCATED).
DAMAGED: Dict[str, str] = {
    # The same error at the same place on every read (there is no archive copy of val/test to compare
    # with); a 16-pixel block in the background, the poses there are unaffected.
    'G3RvsnzQrXQ-10-rgb_front': 'one damaged block near frame 1988 (82.8 s) in the source as downloaded: libav '
                                'reports a decoding error there on every read',
    'cH27PiYX6gI-9-rgb_front': 'truncated in the official How2Sign archive (TRUNCATED): libav reports "partial '
                               'file"',
    'CH7AviIr0-0-8-rgb_front': 'truncated in the official How2Sign archive (TRUNCATED, only 2,133 of 3,792 frames): '
                               'libav reports "partial file" and invalid NAL units at the end',
}


@register
class How2Sign(Dataset):
    """How2Sign frontal raw videos, scheduled val, test, train (the small splits finish first)."""

    name = 'how2sign'
    # One signer fills the frame in front of a green screen, so the largest box is the signer (one
    # person in every frame of a 13-video sample).
    primary_rule = 'largest_bbox'
    split_order = SPLITS

    @property
    def source(self) -> Path:
        return self.data_root / 'How2Sign'

    def out_root(self) -> Path:
        return self.source / 'dwpose'

    def captions(self, video: Video, fps: Fraction) -> Dict[str, List[Cue]]:
        cues = [cue for _, cue in sorted(self._sentences().get(video.video_id, []), key=lambda item: item[0])]
        if video.video_id in TRUNCATED:
            last = Fraction(TRUNCATED[video.video_id]) / fps   # the end of the last decoded frame
            cues = [cue for cue in cues if cue.start < last]
        return {'en': cues}

    def _sentences(self) -> Dict[str, List[Tuple[int, Cue]]]:
        """{VIDEO_NAME: [(sentence number, Cue)]} of the three realigned CSVs (read once)."""
        if getattr(self, '_realigned', None) is None:
            out: Dict[str, List[Tuple[int, Cue]]] = {}
            for split in SPLITS:
                path = self.source / REALIGNED.format(split=split)
                with open(path, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    if tuple(reader.fieldnames or ()) != REALIGNED_COLUMNS:
                        raise ValueError(f'{path}: columns {reader.fieldnames}, expected {REALIGNED_COLUMNS}')
                    for row in reader:
                        number = int(row['SENTENCE_ID'].rsplit('_', 1)[1])
                        cue = Cue(Fraction(row['START_REALIGNED']), Fraction(row['END_REALIGNED']), row['SENTENCE'])
                        out.setdefault(row['VIDEO_NAME'], []).append((number, cue))
            self._realigned = out
        return self._realigned

    def videos(self) -> List[Video]:
        return [Video(path.stem, path, split, expected_frames=TRUNCATED.get(path.stem),
                      known_damage=DAMAGED.get(path.stem))
                for split in SPLITS for path in sorted((self.source / split).glob('*.mp4'))]
