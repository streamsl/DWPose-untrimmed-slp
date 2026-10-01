"""A complete slp_pose dataset in one file: every .mp4 under <data root>/my_dataset/videos/.

    slp-pose list-videos --dataset examples/my_dataset.py --data-root /path/to/data --list
    slp-pose extract     --dataset examples/my_dataset.py --data-root /path/to/data --gpus 0

Outputs go to <data root>/my_dataset/dwpose/ (poses/, persons/, video_meta.csv, video_ids.csv).
Optional: <data root>/my_dataset/splits.csv with lines `<file name>,<split>` (dev, test or train).
"""
import csv

from slp_pose.datasets import Dataset, Video, safe_id


class MyDataset(Dataset):
    name = 'my_dataset'
    primary_rule = 'largest_bbox'   # the signer: a per-frame rule, or video-level such as 'signer_track'
    extraction_rule = None          # the per-frame rule the GPUs pose with; None = primary_rule
    split_order = ('dev', 'test', 'train')

    def videos(self):
        folder = self.data_root / self.name
        splits = {}
        if (folder / 'splits.csv').is_file():
            with open(folder / 'splits.csv', newline='') as f:
                splits = dict(csv.reader(f))
        return [Video(video_id=safe_id(path.stem), path=path, split=splits.get(path.name),
                      info={'file_name': path.name})
                for path in sorted((folder / 'videos').glob('*.mp4'))]
