"""BOBSL (BBC-Oxford British Sign Language): 2,212 BBC episodes with an in-vision interpreter.

Layout under the data root: BOBSL/original_data/videos/mp4/<episode id>.mp4 and the official split
lists BOBSL/original_data/metadata/subset2episode.json ({split: [episode id, ...]}). The video id is the
file stem; outputs go to BOBSL/dwpose/.

Splits (one split file per subtitle alignment): splits_automatic.csv = the official split of every
episode (train, val, test, challenge_test; Video.split), for the automatic subtitles;
splits_manual.csv = the same official split of the 55 manually aligned episodes (train 16, val 4 =
Sent-Val, test 35 = Sent-Test; video_ids.csv's manual_split column), for the manual subtitles. The
official split is signer-independent: no interpreter is in two splits.

Subtitles (both signing-aligned versions, as separate tracks): 'en' =
automatic_annotations/signing_aligned_subtitles/auto_sat_aligned/<episode id>.vtt (every episode,
challenge_test included) -> subs/<id>.en.vtt, misaligned-slt's default; 'en-manual' =
manual_annotations/signing_aligned_subtitles/<episode id>.vtt (the 55 episodes of splits_manual.csv)
-> subs/<id>.en-manual.vtt, chosen with misaligned-slt's subtitles.preferred_suffixes [.en-manual.vtt].
The manual files tag cues whose signing does not match the subtitle ([SUB-DOES-NOT-MATCH-SIGNS] 1,199,
[NOT-SIGNED] 940 incl. 'NOT SIGNED' / 'NOT_SIGNED', [DUPLICATED] 71, [UNANNOTATED] 8, [NOT-ENGLISH] 5,
[INAPPROPRIATE-SIGN] 5); such a cue is dropped (misaligned-slt strips bracket tags and would keep the
text as a translation target). The automatic files carry no tags. The original broadcast subtitles
(original_data/subtitles/audio-aligned/, timed to the speech) are not used.

Metadata (Dataset.split_tracks, subs.write_meta_files): video_meta.csv goes with splits_automatic.csv
and 'en', video_meta_manual.csv with splits_manual.csv and 'en-manual'. Their unreliable_spans
(Bobsl.unreliable_spans): for 'en-manual', the span of every dropped tagged cue, read from the
original manual files (subs/ no longer holds them; without those files the column is unknown); for
'en', none ([]): the automatic files carry no tags, and misaligned-slt itself marks the gap around a
collapsed alignment (a cue shorter than its subtitles.min_duration_s), as in every corpus.

Episode 5976836975695491321 is damaged in the official v1.4 archive (two 1 MiB runs of zero bytes;
decoders stop at frame 71,837 of 90,577). The outputs were extracted from a repaired copy put in place
of the original file: the same 90,577 frames, with the 1,287 frames that depend on the damaged data
([71837, 72447) and [73257, 73934)) black, so they are stored as empty frames. With the official file,
`extract` fails that episode on its frame count; `--videos 5976836975695491321 --allow-frame-mismatch`
extracts its 71,837 decodable frames instead (`check` then reports its decoder errors).
"""
from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..signer import SignerTrackParams, track_rule
from ..subs import Cue, read_vtt
from . import register
from .base import Dataset, Video

AUTO_SUBTITLES = Path('automatic_annotations') / 'signing_aligned_subtitles' / 'auto_sat_aligned'
MANUAL_SUBTITLES = Path('manual_annotations') / 'signing_aligned_subtitles'
QUALITY_TAG = re.compile(r'\[[A-Z][A-Z _-]*\]')   # e.g. [NOT-SIGNED]: the manual files' tags (module docstring)

# signer_track_bobsl, BOBSL's primary rule: the framework's signer_track (its parameters spelled out)
# with step 5's gate: away from the main spot's segments, a steady person is the signer only where they
# sit and look like one of the main spot's anchors (median box IoU >= 0.5, box size and shoulder width
# within 1.3 times that anchor's), so the interpreter on a spot of her own stays and programme people (a
# pianist, a singing presenter, close-ups, people under end credits) do not. A video without a main spot
# has nobody to compare with: step 5 is as in signer_track there.
PARAMS = SignerTrackParams(link_iou=0.5, max_gap_s=1.0, min_track_s=20.0, window_s=60.0, prior=None,
                           merge_gap_s=60.0, min_visit_s=1.0, main_share=0.5, min_density=0.8, max_visit_share=0.5,
                           anchor_s=20.0, fit_iou=0.8, fit_scale=1.25, max_scale=1.3, scale_window_s=1.0,
                           min_scale=0.25, point_conf=0.3, other_iou=0.5, other_scale=1.3)
SIGNER_TRACK_BOBSL = track_rule('signer_track_bobsl', PARAMS)


@register
class Bobsl(Dataset):
    """BOBSL episodes; split = the official split (val, test, train, challenge_test); info: manual_split
    (the official split of the 55 manually aligned episodes, else '')."""

    name = 'bobsl'
    # The signer is the interpreter's spot (signer_track_bobsl), derived on the CPU after each commit
    # from the people the GPU workers posed with largest_bbox (up to 3 per frame, select.py); frames
    # where the interpreter is off screen are empty.
    primary_rule = 'signer_track_bobsl'
    extraction_rule = 'largest_bbox'
    rules = (SIGNER_TRACK_BOBSL,)
    split_order = ('val', 'test', 'train', 'challenge_test')
    split_tracks = (('automatic', 'en'), ('manual', 'en-manual'))

    @property
    def source(self) -> Path:
        return self.data_root / 'BOBSL' / 'original_data'

    def out_root(self) -> Path:
        return self.data_root / 'BOBSL' / 'dwpose'

    def videos(self) -> List[Video]:
        official = self._official_splits()
        manual = {path.stem for path in (self.data_root / 'BOBSL' / MANUAL_SUBTITLES).glob('*.vtt')}
        paths = sorted((self.source / 'videos' / 'mp4').glob('*.mp4'))
        out = []
        for path in paths:
            vid = path.stem
            manual_split = (official.get(vid) or '') if vid in manual else ''
            out.append(Video(vid, path, official.get(vid), dict(manual_split=manual_split)))
        return out

    def splits(self, videos: Sequence[Video]) -> Dict[str, Dict[str, str]]:
        """splits_automatic.csv (the official split) and splits_manual.csv (the same split of the
        manually aligned episodes)."""
        return {'automatic': {v.video_id: v.split for v in videos if v.split},
                'manual': {v.video_id: v.info['manual_split'] for v in videos if v.info.get('manual_split')}}

    def captions(self, video: Video, fps: Fraction) -> Dict[str, List[Cue]]:
        root, tracks = self.data_root / 'BOBSL', {}
        auto = root / AUTO_SUBTITLES / f'{video.video_id}.vtt'
        if auto.is_file():
            tracks['en'] = read_vtt(auto)
        manual = root / MANUAL_SUBTITLES / f'{video.video_id}.vtt'
        if manual.is_file():
            tracks['en-manual'] = [cue for cue in read_vtt(manual) if not QUALITY_TAG.search(cue.text)]
        return tracks

    def unreliable_spans(self, video_id: str, track: str, cues: Sequence[Cue]) -> List[Tuple[Fraction, Fraction]]:
        """'en-manual': the span of every cue that `captions` drops for its tag, from the original
        manual file (FileNotFoundError without it: unknown); 'en': none (module docstring)."""
        if track == 'en-manual':
            manual = self.data_root / 'BOBSL' / MANUAL_SUBTITLES / f'{video_id}.vtt'
            return [(cue.start, cue.end) for cue in read_vtt(manual) if QUALITY_TAG.search(cue.text)]
        return []

    def _official_splits(self) -> Dict[str, str]:
        """{episode id: official split}; an episode listed twice keeps its first split in split_order."""
        with open(self.source / 'metadata' / 'subset2episode.json', encoding='utf-8') as f:
            raw = json.load(f)
        out: Dict[str, str] = {}
        rank = {split: i for i, split in enumerate(self.split_order)}
        for split in sorted(raw, key=lambda s: rank.get(s, len(rank))):
            for episode in raw[split]:
                out.setdefault(str(episode), split)
        return out
