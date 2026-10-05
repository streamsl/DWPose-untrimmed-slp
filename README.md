# slp-pose: DWPose whole-body keypoints for untrimmed sign-language videos

`slp-pose` (Python package `slp_pose`, repository `DWPose-untrimmed-slp`) runs DWPose on every frame
of full-length sign-language videos and writes one keypoint array per video for the signer, in the
`youtube-sl-25` pose schema that [`misaligned-slt`](#6-using-the-outputs-in-misaligned-slt) reads.
A dataset is one Python file; BOBSL and Auslan-Daily News are built in.

1. [What it does](#1-what-it-does)
2. [Install on a fresh Linux machine](#2-install-on-a-fresh-linux-machine-with-an-nvidia-gpu)
3. [Usage](#3-usage)
4. [Adding a dataset](#4-adding-a-dataset)
5. [Built-in datasets](#5-built-in-datasets)
6. [Using the outputs in misaligned-slt](#6-using-the-outputs-in-misaligned-slt)
7. [Development](#7-development)

## 1. What it does

For each frame, at the video's native frame rate and resolution:

1. **Detect** people with YOLOX-L 640 (mmdetection checkpoint).
2. **Pose** up to 3 people with DWPose-L 384x288 (COCO-WholeBody, 133 keypoints, flip test on). Both
   models run as TensorRT fp32 engines with TF32 off.
3. **Pick the signer** with the dataset's signer rule ([section 4](#signer-rules)). Every posed
   person is stored, so you can re-pick the signer later on the CPU with `derive`.

`slp-pose parity` compares this pipeline with the official mmdet/mmpose per-image code on sampled
frames.

Outputs, under the dataset's output root:

```
<out root>/
  poses/<video_id>.npy      the signer, per frame
  persons/<video_id>.npz    the raw record of every frame
  video_meta.csv            one row per finished video
  video_ids.csv             every video of the dataset
  .state/                   logs, done markers, meta rows, failed.jsonl, gpu.csv, lock (run bookkeeping)
```

| File | Content |
|---|---|
| `poses/<vid>.npy` | `(T, 133, 3)` float32. `T` = decoded frames = ffprobe `nb_frames`, native fps, no resampling. COCO-WholeBody order: 0–16 body, 17–22 feet, 23–90 face, 91–111 left hand, 112–132 right hand (the person's own left and right). Channel 0 = `x / W`, channel 1 = `y / H` of the original frame, channel 2 = the raw DWPose SimCC score (never clipped; it can exceed 1). A frame with no signer is all zeros. |
| `persons/<vid>.npz` | The raw record: every person candidate (detector score > 0.1) with its box, score and count/pose-set flags; keypoints of the posed people (up to 3 per frame); the signer's index per frame (`kpt_primary`, -1 = none); the meta row; and provenance (git sha, hashes, engine and checkpoint sha256, library versions, GPU). `poses/` and the meta row are pure functions of it. |
| `video_meta.csv` | `video_id,duration_s,width,height,caption_source,undetected_ratio,masked_runs,empty_runs,handless_runs`, byte-identical to what misaligned-slt's `save_video_meta` writes. `duration_s = T * den / num` from the exact frame rate, so `T / duration_s` gives back the fps. `caption_source` is blank. `undetected_ratio` is the share of frames with nobody in the detector's count set. The three run columns are JSON lists of half-open `[a, b]` frame runs of `poses/` (`[]` = none), computed with the algorithm and constants of misaligned-slt's `prepare_yt25.py` at commit 3eb85af (`meta.MSLT_COMMIT`). The constants are in seconds, converted to frames at the video's own rate: 0.5 s to merge or keep a masked run, 1 s for an empty run, 2 s motion windows, i.e. 13 / 25 / 50 frames at 25 fps and 15 / 30 / 60 at 29.97 fps. `masked_runs` = frames where a second real body (8 or more visible body joints) is present and moves its arms: the standard deviation of its elbows and wrists over a 2 s window is above 0.7 of the signer's shoulder width, or cannot be measured. Each other body is followed as one person by where its shoulders are, not by its detector rank; `empty_runs` = frames whose `poses/` row holds no real body (no signer, or too few visible joints); `handless_runs` = frames whose `poses/` row shows no hand (fewer than 10 of 21 points above 0.3). misaligned-slt cuts masked and empty runs out of the stream and quarantines captions that lie mostly in handless runs. With a per-frame signer rule every posed person counts for `masked_runs`; with a video-level rule (`signer_track`, `signer_track_right`: BOBSL, Auslan News) only the posed people whose box centre lies in the interpreter's region (the box of the signer spot the rule used in that frame), so programme footage elsewhere on screen is not cut. The extraction commits each video with an older meta row (no run columns), and the parent derives it right after, so a row with blank run cells only appears for a video that is committed but not derived yet. |
| `video_ids.csv` | `video_id,split,source`, then the dataset's own columns. `source` is relative to the data root (absolute for a file outside it). It is rewritten at every `extract` start. |

## 2. Install on a fresh Linux machine with an NVIDIA GPU

Tested on Linux x86_64 with Python 3.11 (CPU side) and on 2x RTX 3090 Ti with Python 3.8.

### Prerequisites

- **OS:** Linux x86_64 with glibc >= 2.28.
- **GPU driver:** NVIDIA driver >= 535 (needed by TensorRT 10).
- **Tools:** `ffmpeg` built with libx264, plus `ffprobe` (Debian/Ubuntu: `sudo apt-get install -y ffmpeg`), git and curl.
- **Python 3.11:** either `python3.11` on `PATH`, or [uv](https://docs.astral.sh/uv/)
  (`curl -LsSf https://astral.sh/uv/install.sh | sh`), in which case the setup script installs
  Python 3.11 itself. Another interpreter can be given with `PYTHON=/path/to/python3.11`.
- **No CUDA toolkit**, on any GPU (H100 included): the pip wheels bring their own CUDA libraries.
  The prebuilt mmcv 2.1.0 has CUDA code only up to sm_86, so its GPU ops cannot run on an H100. `slp-pose`
  uses only one of them, the detector's NMS, and where it cannot run uses the same NMS recomputed bit for
  bit in torch (D27).

### Step 1: clone and build the environment

```bash
git clone https://github.com/streamsl/DWPose-untrimmed-slp.git
cd DWPose-untrimmed-slp
scripts/setup_env.sh
```

A clone holds no data. Create `data/` and place the videos as in
[section 5](#5-built-in-datasets), or keep the data anywhere and pass `--data-root` (or set
`$SLP_POSE_DATA`).

`scripts/setup_env.sh` builds `.venv` in six steps. Each step skips what is already done, so after
a failure you can fix the cause and re-run it.

1. Checks the system.
2. Creates the venv. It refuses to touch a venv that it did not create.
3. Installs `requirements.txt`: exact versions, every file hash-checked.
4. Deletes TensorRT's Windows-only builder files (2.9 GB; 1.8 GB with `--python 3.8`).
5. Installs this checkout editable, which provides the `.venv/bin/slp-pose` command.
6. Runs `pip check` and `scripts/check_env.py`.

A fresh run took 18 min, mostly downloading (6.9 GB); the venv is 8.5 GB. Options:

- `--venv DIR`: create the venv somewhere other than `.venv`.
- `--python 3.8`: use `requirements-py38.txt` (TensorRT 10.13.3.9), the environment that the
  built-in datasets were extracted with (2x RTX 3090 Ti).

Run `scripts/setup_env.sh --help` for the full list.

### Step 2: checkpoints

```bash
.venv/bin/slp-pose fetch-models
```

This downloads both checkpoints into the models directory: YOLOX-L 640 from the mmdetection model zoo
(217 MB) and DWPose-L `dw-ll_ucoco_384.pth` (407 MB) from its Hugging Face mirror
(`https://huggingface.co/yzd-v/DWPose`). It checks both sha256 values
(`d3bd2b23…54da1`, `0d9408b1…a7c07`) and never keeps a file whose sha256 is wrong. Without internet
access, copy the files from another machine (or get DWPose-L from the Google Drive / Baidu links in the
[DWPose README](https://github.com/IDEA-Research/DWPose)) and run
`slp-pose fetch-models --no-download --from <file or directory>` (`--from` can be repeated).

The models directory is the first match of: `--models-dir`, `$SLP_POSE_MODELS`, `<checkout>/models`
(when `slp_pose` runs from a source checkout, as it does after `setup_env.sh`; `fetch-models`
creates the directory), `${XDG_CACHE_HOME:-~/.cache}/slp_pose/models`.

### Step 3: TensorRT engines

```bash
.venv/bin/slp-pose build-engines --gpu 0
```

This exports ONNX into `<models dir>/onnx/` and builds the TensorRT engines into
`<models dir>/engines/`: fp32, TF32 off, detector batch 1–64, pose batch 1–256. It takes about
5 min and never overwrites existing engines without `--force`.

- An engine is tied to the GPU model and the TensorRT version, so build the engines on the target
  machine, and again for every other GPU model. One `extract` run loads one engine set on all its
  GPUs.
- Never rebuild in the middle of a corpus. The engine sha256 is part of every video's extraction
  hash, so after a rebuild, resuming re-extracts every video.

### Step 4: check the environment

`setup_env.sh` already ran this once; run it again after installing a GPU or a driver.
`.venv/bin/python scripts/check_env.py` checks:

- Python version and library versions against the lock file;
- the `slp_pose` start-up checks;
- ffmpeg and the driver;
- that torch and torchvision have CUDA code for every GPU (mmcv's is only reported: where it has none,
  `slp-pose` uses its own exact NMS);
- with a GPU visible, slp-pose's and torchvision's NMS on it (and mmcv's, compared with ours, where mmcv has
  code for that GPU) and a TensorRT builder.

Add `--arch 9.0` to require code for a GPU that is not in the machine.

### Step 5: quick parity check

Run this on the target GPU, with your dataset, before the first extraction. It compares `slp-pose`
with the official mmdet/mmpose code (see [parity](#parity)):

```bash
.venv/bin/slp-pose parity --dataset auslan_news --videos 6 --windows 2 --window-s 20 --full-videos 0 --gpu 0 --jitter batched_det
```

These 12 windows of 20 s took 20 min at 1280x720 on one RTX 3090 Ti. Exit code 0 means the gate
passed. The report goes to `<out root>_dev/parity/parity_trt.json` and `.md`. Then
[extract](#extract).

### Installing only the pip package

`slp-pose` is not on PyPI. Install it from a clone into a Python 3.11 venv:

```bash
python3.11 -m venv slp-venv && . slp-venv/bin/activate
pip install -U pip 'setuptools<82'
pip install 'numpy<2' torch==2.1.2 torchvision==0.16.2 --extra-index-url https://download.pytorch.org/whl/cu121
pip install --no-build-isolation chumpy==0.70
git clone https://github.com/streamsl/DWPose-untrimmed-slp.git
pip install './DWPose-untrimmed-slp[trt]' tensorrt-cu12==10.16.1.11 \
    --extra-index-url https://pypi.nvidia.com -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
rm slp-venv/lib/python3.11/site-packages/tensorrt_libs/libnvinfer_builder_resource_win*   # optional, 2.9 GB
```

What each step is for:

- torch comes from the PyTorch index, mmcv 2.1.0 from the OpenMMLab wheels (PyPI has only a slow
  sdist), and TensorRT from `pypi.nvidia.com`.
- `chumpy` is an mmpose requirement that is never imported; its sdist cannot be built with build
  isolation, hence the separate `--no-build-isolation` line.
- `setuptools<82`: mmengine still imports `pkg_resources`.
- `slp_pose` checks the exact TensorRT version at start-up: 10.16.1.11 on Python 3.11, 10.13.3.9 on
  3.8–3.10.

This route was tested in a fresh Python 3.11 venv and resolved to the same runtime versions as
`requirements.txt`. It took about an hour, because of a slow PyTorch index and pip backtracking
through opencv versions; `scripts/setup_env.sh` is faster and installs exactly the locked files. You
also need ffmpeg and the driver from the prerequisites.

Then fetch the checkpoints, build the engines and run the commands below as `slp-pose ...`.
Checkpoints and engines go to `~/.cache/slp_pose/models` unless `--models-dir` or `$SLP_POSE_MODELS`
says otherwise. A wheel can be built with `pip wheel . --no-deps`.

## 3. Usage

`slp-pose <command>` and `python -m slp_pose <command>` are the same; `slp-pose <command> --help`
lists every option.

Options common to the commands:

- `--dataset NAME_OR_FILE` (dataset commands): a built-in or installed dataset name, or a dataset
  `.py` file. The default is `bobsl`.
- `--data-root` (dataset commands): default `$SLP_POSE_DATA`, else `./data`.
- `--models-dir` (every command).
- `--out-root` (commands that use an output root): default the dataset's `out_root()`.

GPU indices are in PCI bus order, as `nvidia-smi` shows them. Exit codes: 0 success; 1 failure
(failed videos, `check` problems, a failed parity gate, missing models); 2 bad arguments or an
invalid dataset; 130 interrupted.

### list-videos

```bash
slp-pose list-videos --dataset auslan_news --list
```

Checks the dataset's contract (ids, files, splits, rules). It prints the output root, the rules and
the video count per split; `--list` also prints every video.

### extract

```bash
slp-pose extract --dataset auslan_news --gpus 0,1
```

- **Workers:** one worker process per GPU (`--gpus`, default `0,1`). Videos run in the dataset's
  split order, largest file first.
- **Commits:** a video is committed atomically when it is finished. The extraction rule poses the
  people, then the parent derives the signer with the primary rule on the CPU.
- **Progress:** one progress line a minute (videos, fps per GPU, ETA). `video_meta.csv` is merged
  when the run starts, every 50 commits, and when it ends (also after Ctrl-C, SIGTERM or an error),
  so it lists every committed video. A run that was killed outright (SIGKILL, a crash) is caught up
  by the next run's first merge.
- **Resume:** re-run the same command. Committed videos are skipped. A video whose signer rule,
  count-set thresholds or `video_meta.csv` definitions changed is re-derived on the CPU
  (`render-done` renders it again only where the signer rows or the people counts change); one
  whose engines, checkpoints, configs,
  candidate / pose-set thresholds, backend or extraction rule changed is re-extracted. Unfinished
  work in `.work/` is discarded.
- **Stop:** Ctrl-C, or SIGTERM to the parent process. Each worker stops after its current chunk,
  nothing half-written is kept, the CSV is merged, and the next `extract` continues. To find the
  parent, `pgrep -af 'slp.pose extract'` lists it (the workers do not match), then
  `kill -TERM <pid>`. A `render-done --follow` of the root keeps following for 15 minutes after
  the stop; if the restart comes later, start `render-done` again once `extract` has started.
- **Failures:** a video gets 2 attempts, and each failed attempt is a line in `.state/failed.jsonl`.
  A worker that crashes, or that is silent for 10 min, is respawned (up to 3 times an hour per GPU).
  The exit code is 1 when any video failed or is still pending.
- **Lock:** one `extract` or `derive` at a time per output root (`.state/lock`).
- **Options:** `--videos ID ...` and `--limit N` restrict the run; `--backend torch` uses PyTorch
  instead of TensorRT (slower); `--allow-frame-mismatch` accepts a decoded frame count that differs
  from ffprobe's.

Long runs are best started detached, so that they survive the terminal:

```bash
setsid nohup slp-pose extract --dataset auslan_news --gpus 0,1 > extract_stdout.log 2>&1 < /dev/null &
```

Run state in `<out root>/.state/`:

| Path | Content |
|---|---|
| `logs/<command>_<time>.log` | the log of each command run on this root, also printed on stdout (`render-done` logs to its vis dir and `parity` to its report dir instead) |
| `failed.jsonl` | failed attempts |
| `gpu.csv` | `nvidia-smi` samples every 10 s |
| `done/<vid>.json` | commit markers |
| `meta/<vid>.json` | meta rows |
| `lock` | the run lock |

### check

```bash
slp-pose check --dataset auslan_news
```

Checks every committed video:

- that every member of `persons/` matches its CRC-32 (one sequential read of each file; skip it with
  `--no-crc` when the volume is slow);
- the invariants between `poses/` and `persons/`. A video derived by older code (another
  `meta_version`, e.g. every committed video right after an update of the signer rule or the meta
  definitions) gets one line instead, `derived by other code (meta_version 4, ...)`, because its
  signer picks and meta row follow the older rules. That is not damage: run `derive`;
- that the video was derived with the dataset's primary rule;
- primary-signer switches (frames where the signer's box jumps). For these it prints `render`
  commands for the windows worth reviewing.

The exit code is 1 when any video has a problem.

### derive

```bash
slp-pose derive --dataset bobsl --primary-rule signer_track
```

CPU only. It re-picks the signer and recounts people from `persons/`, then rewrites the signer
index and meta row in `persons/`, `poses/` (only when its rows change), and `video_meta.csv` (with
the current run columns). The detections and keypoints stay as they are. `--videos ID ...` limits
it; the default is every committed video. It takes a few seconds per video (CPU), plus rewriting
`persons/` (about 0.19 GB per hour of BOBSL video) and, when the signer rows change, `poses/`
(about 0.14 GB per hour). A derive that changes only the meta row reads `poses/` and leaves it
untouched.

- A per-frame rule that picks a person who was never posed fails for that video, because that rule
  needs a re-extract.
- It takes the output root's lock, so it cannot run beside an `extract` of the same root.
- It first reads each `persons/` file once to check its CRCs. A damaged file (or one that is no
  readable zip) is left as it is, and the video fails with a message (rewriting it would give the
  damaged rows valid CRCs). Re-extract such a video: delete `.state/done/<vid>.json` and run
  `extract`. If the rows that derive then reads again differ from what the CRC check read (a
  faulty read of the disk), it writes nothing and the video fails: run `derive` again.

A rule other than the dataset's `primary_rule` is flagged by `check` and undone by the next
`extract`. To change the rule for good, edit `primary_rule` in the dataset file and keep its
extraction rule: the next `extract` then re-derives every committed video, and re-extracts none.

### render and render-done

```bash
slp-pose render abc_news_2022-10-16 --dataset auslan_news --start-s 750 --duration-s 30 --persons
slp-pose render abc_news_2022-10-16 --dataset auslan_news --full
```

`render` draws the saved keypoints over the source video on the CPU:

- `--persons` also draws the other people and the candidate boxes.
- `--full` renders the whole video.
- The output is H.264 (libx264 veryfast, CRF 26) at the exact source frame rate, written to
  `<out root>/renders/<vid>_<start frame>.mp4` (`<vid>_full.mp4` with `--full`) unless `--out` is
  given.

`render-done` renders every committed video in full into `<out root>_vis/<vid>.mp4` (`--vis-dir`):

- It draws every person and candidate box; `--no-persons` draws the signer only.
- It runs `--jobs` render processes at a time (default 3), at nice +10 with the GPUs hidden.
- It never writes into the output root, so it can run beside `extract`.
- It keeps a render record per video in `<vis dir>/.state/rendered/`, so a video is rendered again
  when what it draws changes (a re-extract, or a `derive` that changes the signer or the people
  counts), with a different `--no-persons`, or when the mp4 was replaced. A `derive` that changes
  only `video_meta.csv` does not re-render. Records written before this rule are converted while
  the video is unchanged: run `render-done --limit 0` (renders nothing) before such a `derive`.
- After an update that changes the signer rule, most rendered videos are rendered again (the
  2026-10-05 spot identity changes the signer in about 80-90 % of BOBSL videos: roughly 9-11 h
  for 786 BOBSL renders at 3 jobs). Start `render-done` once the re-derive of the committed videos
  has finished, or give it `--limit`, so that it does not compete with the re-derive for the disk.
- A rendered video it looks at while a commit of that video is in progress is looked at again at
  the next pass, not rendered. The exit code is 1 when any video failed, or was still mid-commit
  when `render-done` ended (run it again).
- Only one `render-done` can run per vis dir. Logs go to `<vis dir>/.state/logs/`.

With `--follow` it looks for new commits every `--poll-s` seconds (default 300) and exits once
nothing is left and the extraction has been stopped for 15 minutes, so stopping `extract` with
SIGTERM and starting the same command again does not end it. Start it after `extract` has taken
its lock, i.e. after the extract log shows its start line (`<dataset> -> <out root>: N videos, ...`).
If you start it earlier, it warns and returns once the videos committed so far are rendered.

```bash
setsid nohup slp-pose render-done --dataset auslan_news --follow > render_done_stdout.log 2>&1 < /dev/null &
```

### parity

```bash
slp-pose parity --dataset bobsl --gpu 0 --jitter batched_det
```

Compares `slp-pose` with the official per-image `inference_detector` + `inference_topdown`, in strict
fp32, on the same decoded frames. On a GPU that the prebuilt mmcv has no CUDA code for (an H100), the
official detector's mmcv NMS runs `detpost.head_nms` instead, the same computation bit for bit; the
report's `provenance.official_nms` says which one ran.

- **Sample:** by default 20 random videos x 3 windows of 30 s, plus 2 whole videos (`--videos`,
  `--windows`, `--window-s`, `--full-videos`, `--seed`). That takes hours, mostly in the official
  code at about 16–20 fps.
- **Report:** `<out root>_dev/parity/parity_<backend>.json` and `.md` (`--out`).
- **Exit code:** 0 iff the gate passes.

What the gate requires:

- 0 count and identity mismatches (a frame that differs only by borderline detector scores is
  excused);
- box max diff < 0.01 px and detector score max diff < 1e-3 (a person whose box differs by a
  detector-head NMS near-tie is excused from these and the keypoint metrics);
- of the confident keypoints (official score > 0.3), at most 5e-5 moving more than 1.5 SimCC
  steps and at most 2e-3 moving more than 0.5 steps, each overall and for each hand. A step is one
  SimCC bin of the person's crop, about 0.7 px for BOBSL-sized people;
- keypoint score max diff < 0.01, over the keypoints that moved at most 1.5 steps.

With `--jitter batched_det`, the run also measures how much the official code deviates from itself
when it batches its detections. Each keypoint limit then becomes the larger of the fixed limit and
1.5x the baseline's value for the same metric and part. This matters at 1280x720, where the official
code's own deviation already exceeds some of the fixed limits. Without the baseline, the fixed limits
apply. `tf32` is an informational baseline that never changes a limit.

A finished report can be re-gated offline, without a GPU:

```bash
python -c "from slp_pose import parity; r = parity.load_report('OLD/parity_trt.json'); print(parity.gate(r)); parity.write_report(r, 'NEW')"
```

## 4. Adding a dataset

A dataset is one Python file holding one `Dataset` subclass. This complete example is
[`examples/my_dataset.py`](examples/my_dataset.py), shown here without its docstring. It takes
every `.mp4` under `<data root>/my_dataset/videos/`, plus optional splits from
`<data root>/my_dataset/splits.csv` (lines `<file name>,<split>`):

```python
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
```

```bash
slp-pose list-videos --dataset examples/my_dataset.py --data-root /path/to/data --list
slp-pose extract --dataset examples/my_dataset.py --data-root /path/to/data --gpus 0
```

The outputs go to `/path/to/data/my_dataset/dwpose/`. Run `list-videos` first: it checks the
contract below without touching a GPU.

**How `--dataset` finds a dataset:**

- a path ending in `.py`: that file, which must define (or `@register`) exactly one `Dataset`
  subclass;
- a built-in name: `bobsl`, `auslan_news` (`slp_pose/datasets/`);
- `package.module:ClassName`: that class of an importable module;
- an installed plugin: an entry point in the group `slp_pose.datasets`, named after the dataset. In
  the plugin package's `pyproject.toml`:

  ```toml
  [project.entry-points."slp_pose.datasets"]
  my_dataset = "my_package.my_dataset:MyDataset"   # or a module that defines exactly one Dataset
  ```

**The contract** (`slp_pose/datasets/base.py`). The framework constructs `cls(data_root)`, so
`self.data_root` is the `--data-root`. It then checks the class and the video list, schedules the
videos and runs every command on them.

| Member | Meaning |
|---|---|
| `name` | Dataset name, `[A-Za-z0-9._-]+`. Used for the registry, the logs and the default output root. |
| `videos()` | Returns every video as `Video(video_id, path, split=None, info={})`. `video_id` is the output file stem: `[A-Za-z0-9._-]+`, unique, and not ending in `_segment_<n>`. `safe_id()` replaces other characters with `_`. `path` must be an existing file. `split` is a str or None. `info` holds extra str columns for `video_ids.csv`. |
| `out_root()` | Output root. Default `<data root>/<name>/dwpose`; override it to put the outputs elsewhere. |
| `primary_rule` | The final signer rule behind `poses/`, per-frame or video-level. Default `largest_bbox`. |
| `extraction_rule` | The per-frame rule the GPU workers pose with. `None` = `primary_rule`, which must then be per-frame. |
| `rules` | The dataset's own signer rules ([below](#rules-of-your-own)), a tuple of `FrameRule` and `VideoRule`. Default none: only the framework's rules. |
| `split_order` | When set, splits are extracted in this order (videos without a split last), and every `split` must be one of them or None. |

If you will use the outputs in misaligned-slt, avoid dots in video ids: it finds captions by
cutting `<vid>.<lang>.vtt` at the first dot.

### Signer rules

The workers pose up to 3 people per frame and store them all in `persons/`:

1. the extraction rule's pick;
2. the largest box;
3. the highest-score box;
4. then the rest of the pose set by detector score.

The primary rule then picks the signer among the posed people. Pick an extraction rule that gets
the signer posed. The framework's rules serve every dataset; Auslan News adds two of its own
(`slp_pose/datasets/auslan_news.py`), which only it can use:

| Rule | Kind | Picks |
|---|---|---|
| `largest_bbox` | per-frame | the largest box |
| `highest_score` | per-frame | the highest detector score |
| `signer_track` | video-level | the signer's spot. Boxes are linked into tracks and pooled into spots. The spot present in at least 50 % of the frames is the signer, with zeros while it is empty (signer off screen). A person joins the spot only at the interpreter's size (box and shoulder width within 25 %) and, for a visit shorter than 20 s, exactly where the interpreter sits; a box whose shoulders are more than 1.3 times the interpreter's (a close-up after a cut), less than a quarter of hers (barely seen shoulders of a close-up) or not visible does not count. Elsewhere, a steady person present for at least 20 s counts. It reads the shoulders of the keypoints. |
| `right_largest` (Auslan News) | per-frame | the largest box whose centre lies in the right 40 % of the frame, else the largest box |
| `signer_track_right` (Auslan News) | video-level | `signer_track` with a preference for the right-hand spot, and a steady person elsewhere counts only if their wrists keep moving like a signer's. While the spot's interpreter is off screen for at least 10 s (and at most 3 minutes), a person present for at least half of that time whose wrists keep moving like a signer's (an on-site or a picture-in-picture interpreter) is the signer for the whole absence; a brief movement such as an itch scratched, or a keypoint glitch, does not count |

- With a per-frame primary rule, only frames with nobody in the pose set (detector score > 0.3)
  are zeros. A video-level rule also leaves zeros wherever the signer is absent.
- Changing `primary_rule` later costs only a CPU derive, as long as the extraction rule stays the
  same. Set `extraction_rule` explicitly before you change a `primary_rule` that it defaults to.
- Changing the extraction rule re-extracts, except a switch between `largest_bbox` and
  `highest_score`.

### Rules of your own

A rule that only your dataset needs stays in your dataset file. Define it there, list it in
`rules`, and name it in `primary_rule` or `extraction_rule` like a framework rule (so does
`derive --primary-rule`). From [`examples/custom_rules.py`](examples/custom_rules.py), whose
interpreter stands on the left:

```python
class LeftSigner(Dataset):
    name = 'left_signer'
    extraction_rule = 'left_largest'
    primary_rule = 'signer_track_left'
    rules = (FrameRule('left_largest', functools.partial(left_largest, **LEFT), LEFT),
             signer.track_rule('signer_track_left', signer.SignerTrackParams(prior='left'), left_prior))
```

- **Per-frame:** `FrameRule(name, pick, params=None)`. `pick(boxes, scores, frame_size)` gets one
  frame's pose set (`boxes` (n,4) xyxy pixels, `scores` (n,), `frame_size` (W, H)) and returns the
  index of the signer. The GPU workers pose with it when it is the extraction rule. `pick` (and
  `choose` below) can be any callable: a function, a `functools.partial`, or an object with
  `__call__`, hashable or not.
- **Video-level:** `VideoRule(name, choose, params)`. `choose(people)` gets the posed people of a
  whole video (`PosedPeople`: boxes, scores, per-frame offsets, keypoints, fps, frame size) and
  returns a `SignerChoice`: per frame the signer's position among that frame's posed people (-1 =
  nobody) and the interpreter's region box (NaN = none; the meta columns count the people inside
  it). `derive` applies it on the CPU. Build it from scratch, or from `slp_pose/signer.py`:
  `track_rule` is `signer_track` with your parameters and position prior, and `spot_layout`,
  `signer_segments` and `signing_boxes` are its steps and the signing-motion measure that Auslan
  News' `signer_track_right` uses. `SignerTrackParams()` includes the spot identity (`anchor_s`,
  `fit_iou`, `fit_scale`, `max_scale`, `scale_window_s`, `min_scale`, `point_conf`); `fit_iou=0,
  fit_scale=0, max_scale=0, min_scale=0` gives the box-only spots of before, which read no
  keypoints. The signing-motion defaults (`MotionParams()`) were calibrated on one news dataset
  (D21): spell out your own values, as `auslan_news.py` does.
- Names match `[A-Za-z0-9._-]+`, are unique and are not a framework rule's name.
- The name and `params` (a JSON-able dict) are hashed, the code is not. Changing them for the
  extraction rule re-extracts the committed videos; for the primary rule, it re-derives them on
  the CPU. So when you change a rule's code, change its `params` or its name too. A `FrameRule`
  without `params` is known by its name alone.
- The GPU workers import your dataset to find its per-frame rule, so define the class at the top
  level of a `.py` file, a module or a plugin, not inside a function.
- Code that takes a rule takes the rule object (`dataset.rule(name)`), or a name alone. A name
  alone finds a framework rule or a built-in dataset's own rule, such as Auslan News'
  `signer_track_right`, never your dataset's. So check your outputs with `slp-pose check --dataset
  <your file>`, or in Python with `record.check(root, video_id, dataset=dataset)`.

## 5. Built-in datasets

Both datasets keep the interpreter only. A frame where no interpreter is on screen is stored as
zeros, even when other people are posed (they stay in `persons/`).

| | BOBSL (`bobsl`) | Auslan-Daily News V1 + V2 (`auslan_news`) |
|---|---|---|
| Videos | 2,212 BBC episodes, 149.4 M frames, 444x444, 25 fps | 125 ABC news broadcasts (V1: 45, 2022–2023; V2: 80, 2023–2025), 5.65 M frames, 1280x720 (six 2025 V2 videos: 1024x576), 29.97 fps (V1) or 25 fps (V2) |
| Video id | the episode id (file stem), e.g. `5085344787448740525` | `abc_news_YYYY-MM-DD` |
| Splits | `val` 32, `test` 250, `train` 1658, `challenge_test` 272 (extracted in this order) | V1 from the annotation workbook: `dev` 4, `test` 6, `train` 35; V2: none |
| `video_ids.csv` extras | none | `subset` (`News_V1` / `News_V2`) and `annotation_name`: the workbook's `D_M_YYYY` for V1, the file stem (the VTT name) for V2 |
| Rules | extraction `largest_bbox`, primary `signer_track` | extraction `right_largest`, primary `signer_track_right` (its own rules, in `auslan_news.py`) |
| Signer policy | the in-vision interpreter at their usual position. While the interpreter is off screen the frame is zeros, not a programme person. | the interpreter in the right-hand panel, or centred full screen at the end of a broadcast. A change of interpreter mid-video counts as one signer. An on-site interpreter beside a press-conference speaker is kept as the signer, also while the panel interpreter is off screen, and so is a picture-in-picture interpreter while the panel is gone; the frames between the inset's showings stay zeros, even when a signing interviewee fills the screen. A still speaker, officer or newsreader is not picked. |
| Output root | `<data root>/BOBSL/dwpose/` | `<data root>/Auslan-Daily/dwpose/` |
| Runtime, 2x RTX 3090 Ti | about 100 fps per GPU, about 8.5 days | about 75 fps per GPU (median per video, range 58–88; measured with `render-done` running beside it), about 10 h |

Place the data under the data root like this:

```
<data root>/BOBSL/original_data/videos/mp4/<episode id>.mp4
<data root>/BOBSL/original_data/metadata/subset2episode.json      {split: [episode id, ...]}

<data root>/Auslan-Daily/News_V1/ABC News With Auslan - D M YYYY.mp4
<data root>/Auslan-Daily/News_V1_Annotation.xlsx                  splits (its 'EA<n>' Web News rows are skipped)
<data root>/Auslan-Daily/News_V2/ABC News With Auslan - D M YYYY.mp4
```

One BOBSL episode, `5976836975695491321`, is damaged in the official v1.4 copy itself (two runs of
zero bytes; decoders stop at frame 71,837 of 90,577). `bobsl.py` reads it from
`<data root>/BOBSL/repaired/5976836975695491321.mp4` when that file exists, else from the original.
The repaired copy has the same 90,577 frames at 25 fps, with the 1,282 damaged frames
([71840, 72447) and [73259, 73934), listed in `bobsl.REPAIRED`) painted black. Those frames are
stored as empty (no signer). `video_ids.csv` shows the path that was used. Publish such a copy under
a temporary name and rename it into place.

Only the TV news part of Auslan-Daily is used. The Communication and Web News parts are left out
because they have too many multi-person videos. The BOBSL licence keeps its outputs local.

## 6. Using the outputs in misaligned-slt

An output root has the layout of a misaligned-slt language root. To use it, add a language entry
for it to misaligned-slt's `configs/data.yaml`, with the output root as `root`, and list that entry
in `active_languages` (and in the `pretrain_languages` pool if it should be pooled):

```yaml
languages:
    bobsl:
        name: BSL (BOBSL, slp-pose DWPose)
        target_lang: en_XX
        pretrained_slt: checkpoints/openasl_pose_only_slt.pth
        root: /path/to/data/BOBSL/dwpose     # the slp-pose output root
        pose:
            fps: 25.0      # fallback only: the real fps per video comes from video_meta.csv
            width: 444
            height: 444
```

misaligned-slt reads `poses/` and `video_meta.csv` unchanged. Its loader needs the `masked_runs`,
`empty_runs` and `handless_runs` columns, and `slp-pose` fills all three. The loader takes each
video's fps from `duration_s`: `T / duration_s` is 25 or 29.97 up to float rounding, which its
conversion absorbs. It converts its time constants (Λ_min, δ and the run rules) from seconds to
frames at that rate, so the shortest cut is 0.5 s: 13 frames at 25 fps, 15 at 29.97 fps.

The run columns follow misaligned-slt's own `prepare_yt25.py` at the commit named by
`slp_pose.meta.MSLT_COMMIT` (currently 3eb85af). A derived record stores that commit as
`provenance['mslt_commit']`. With a misaligned-slt checkout next to this repository (or at
`MSLT_ROOT`, with `MSLT_PYTHON` naming a Python that can import it), `tests/test_meta_runs.py`
compares our columns with its functions. When misaligned-slt changes these rules, port the change,
bump `meta.META_VERSION`, and run `slp-pose derive`, which re-derives every video on the CPU. Any
change to the run columns changes misaligned-slt's annotation fingerprints, and it refuses
segmenters trained on the older labels.

Two definitions are ours, for corpora with an in-vision interpreter: BOBSL and Auslan News, whose
signer rule is video-level.

- `masked_runs` counts only the people inside the interpreter's region. On a news programme the
  whole frame would cut the footage the interpreter signs over: 19.6 % of one Auslan video against
  0.48 %.
- `empty_runs` marks the frames whose `poses/` row has no real body.

misaligned-slt's own data has no such region.

Things to know before training:

- **One frame rate per pool:** misaligned-slt gives a batch and a stage-2 pool one frame floor, and
  it raises when their videos' rates give different frame counts. Auslan News V1 is 29.97 fps and
  V2 is 25 fps, so they cannot share a pool. Keep V2 out of the splits (as it is now), or register
  V1 and V2 as two languages with separate roots. A per-subset output root would belong in
  `slp_pose/datasets/auslan_news.py`. BOBSL is 25 fps throughout.
- **Captions:** put them at `<root>/subs/<video_id>.<lang>.vtt` (`.en.vtt` for English). Videos
  without one are dropped. BOBSL's subtitles and Auslan News V1's workbook need converting first.
- **Splits:** misaligned-slt takes them from the one CSV that `splits.signverse_csv` names, and
  refuses to run if none of the ids match. Merge the `video_id,split` rows of `video_ids.csv` into
  that CSV, with a blank (or new) `sign_language` value so its SignVerse download stage never plans
  them. It reads `val` as `dev`, and drops videos with any other split or a blank one, e.g. BOBSL
  `challenge_test` and all of Auslan News V2.
- **Case lexicon:** misaligned-slt pools the training captions of every `en_XX` language into one
  case lexicon, stored next to the root (`<root>/../case_lexicon.en_XX.json`). Adding BOBSL or
  Auslan News changes that pool for every English language.
- **Body confidence:** our channel 2 is DWPose's raw score. SignVerse's body scores are all 1.0,
  an artefact of its extraction, so after misaligned-slt's normaliser our mean body confidence is
  about 0.96 against 1.0. Every threshold (0.3) treats them alike.

## 7. Development

### Refreshing the lock

`scripts/lock_requirements.sh` (needs uv) regenerates `requirements.txt` with the newest versions
within the constraints listed in it:

- mmdet 3.3.0, so mmcv < 2.2 on torch 2.1;
- numpy < 2;
- TensorRT < 11;
- setuptools < 82.

Afterwards, review the diff and build a venv from the new file (`scripts/setup_env.sh --venv DIR`,
which ends with `scripts/check_env.py`). A new TensorRT also has to go into `slp_pose/env.py`
(`TENSORRT_VERSIONS`), and needs rebuilt, parity-gated engines.

### Layout

| Path | What |
|---|---|
| `slp_pose/cli.py`, `__main__.py` | the `slp-pose` command line |
| `slp_pose/settings.py`, `paths.py` | `Settings` (thresholds, batches, precision, model paths); data root and models directory resolution |
| `slp_pose/datasets/` | the dataset contract (`base.py`), registry and resolution (`__init__.py`), `bobsl.py`, `auslan_news.py` |
| `slp_pose/env.py` | start-up checks: pinned versions, which mmpose copy is imported, `CUDA_DEVICE_ORDER`, TF32 off |
| `slp_pose/fetch.py` | `fetch-models` |
| `slp_pose/engines.py` | model building, ONNX export, TensorRT build, TensorRT and torch engines |
| `slp_pose/video.py`, `prep.py` | ffprobe and frame reading; exact CPU letterbox and pose crops (mmcv / mmpose maths) |
| `slp_pose/detpost.py`, `posepost.py` | GPU detector and pose post-processing |
| `slp_pose/select.py`, `signer.py`, `rules.py` | the framework's per-frame and video-level signer rules and their building blocks; rules by name |
| `slp_pose/record.py`, `meta.py`, `types.py` | the `persons/` record, `poses/` derivation, commit and resume, `video_meta.csv`; shared types |
| `slp_pose/worker.py`, `run.py` | one worker per GPU; scheduling, supervision and the CSV merge |
| `slp_pose/render.py`, `render_batch.py` | `render` and `render-done` |
| `slp_pose/parity.py` | the parity gate |
| `slp_pose/model_configs/` | the YOLOX and DWPose configs, shipped as package data and never edited (their sha256 is in the extraction hash) |
| `models/` | a checkout's models directory, made by `fetch-models` and `build-engines` (its contents are gitignored): checkpoints, `onnx/`, `engines/`, each generated file with a `.sha256` sidecar |
| `examples/my_dataset.py`, `custom_rules.py` | the example dataset; one with signer rules of its own |
| `scripts/` | `setup_env.sh`, `check_env.py`, `lock_requirements.sh` |
| `requirements.txt`, `requirements-py38.txt` | hash-pinned locks for Python 3.11 and 3.8 |
