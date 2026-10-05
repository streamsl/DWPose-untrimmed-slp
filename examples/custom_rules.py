"""A dataset with signer rules of its own (spec D22): every .mp4 under <data root>/left_signer/videos/,
whose interpreter stands at the LEFT of the frame, beside a presenter who may be larger.

    slp-pose list-videos --dataset examples/custom_rules.py --data-root /path/to/data --list
    slp-pose extract     --dataset examples/custom_rules.py --data-root /path/to/data --gpus 0

A rule that serves one dataset only is defined in its file and listed in `rules`; primary_rule and
extraction_rule then name it, beside the framework's generic rules (largest_bbox, highest_score,
signer_track):
- `left_largest` (per frame, a FrameRule; the GPU workers pose with it): the largest box whose
  centre lies in the left 40 % of the frame, else the largest box, so the interpreter is posed;
- `signer_track_left` (video level, a VideoRule; the parent derives it on the CPU after each
  commit): the framework's signer_track (spots of IoU-linked tracks, D14, with D24's spot identity:
  box size, shoulder width and anchors; it reads the shoulder keypoints) with a left position
  prior, built with signer.track_rule. SignerTrackParams(prior='left', fit_iou=0, fit_scale=0,
  max_scale=0, min_scale=0) would give D14's box-only spots, which read no keypoints.
Their names and params are hashed: changing `left_largest` re-extracts the committed videos,
changing `signer_track_left` re-derives them on the CPU. Change the params, or the name, whenever
the code of a rule changes. slp_pose/datasets/auslan_news.py is a larger example (a video-level
rule built from signer.py's building blocks plus signing motion).
"""
import functools

import numpy as np

from slp_pose import signer
from slp_pose.datasets import Dataset, FrameRule, Video, safe_id


def left_largest(boxes, scores, frame_size, max_x):
    """Index of the largest box whose centre x is < max_x * W, else of the largest box."""
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    left = (boxes[:, 0] + boxes[:, 2]) / 2 < max_x * frame_size[0]
    return int(np.argmax(np.where(left, area, -np.inf))) if left.any() else int(np.argmax(area))


def left_prior(boxes):
    """A spot's prior key (larger wins): minus the median box centre x of its boxes."""
    return -float(np.median((boxes[:, 0] + boxes[:, 2]) / 2))


LEFT = dict(max_x=0.4)


class LeftSigner(Dataset):
    name = 'left_signer'
    extraction_rule = 'left_largest'
    primary_rule = 'signer_track_left'
    rules = (FrameRule('left_largest', functools.partial(left_largest, **LEFT), LEFT),
             signer.track_rule('signer_track_left', signer.SignerTrackParams(prior='left'), left_prior))

    def videos(self):
        folder = self.data_root / self.name / 'videos'
        return [Video(safe_id(path.stem), path) for path in sorted(folder.glob('*.mp4'))]
