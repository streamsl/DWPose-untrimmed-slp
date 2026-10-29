# slp-pose: DWPose whole-body keypoints for untrimmed sign-language videos

`slp-pose` (Python package `slp_pose`) runs DWPose on every frame of full-length sign-language
videos and writes one keypoint array per video for the signer, in the `youtube-sl-25` pose schema
that [misaligned-slt](#7-using-the-outputs-in-misaligned-slt) reads. BOBSL, Auslan-Daily News and
How2Sign are built in; any other dataset is one Python file.

1. [Quick start](#1-quick-start)
2. [How it works](#2-how-it-works)
3. [Outputs](#3-outputs)
4. [Installation](#4-installation)
5. [Commands](#5-commands)
6. [Built-in datasets](#6-built-in-datasets)
7. [Using the outputs in misaligned-slt](#7-using-the-outputs-in-misaligned-slt)
8. [Adding a dataset](#8-adding-a-dataset)
9. [Development](#9-development)

## 1. Quick start

On Linux x86_64 with an NVIDIA GPU (driver >= 535) and `ffmpeg`:

```bash
git clone https://github.com/streamsl/DWPose-untrimmed-slp.git
cd DWPose-untrimmed-slp
scripts/setup_env.sh                                    # builds .venv (Python 3.11)
.venv/bin/slp-pose fetch-models                         # the two checkpoints
.venv/bin/slp-pose build-engines --gpu 0                # TensorRT engines for this GPU
.venv/bin/slp-pose list-videos --dataset auslan_news    # checks the dataset under ./data
.venv/bin/slp-pose extract --dataset auslan_news        # on GPUs 0,1; --gpus 0 for one GPU
.venv/bin/slp-pose check --dataset auslan_news          # verifies the outputs
.venv/bin/slp-pose subs --dataset auslan_news           # subtitles, split file and video_meta.csv for misaligned-slt
```

A clone holds no data: put the videos under `./data` as in [section 6](#data-layout), or pass
`--data-root`.

## 2. How it works

For each frame, at the video's native frame rate and resolution:

1. **Detect** people with YOLOX-L 640 (the mmdetection checkpoint).
2. **Pose** up to 3 people with DWPose-L 384x288 (COCO-WholeBody, 133 keypoints, flip test on).
   Both models run as TensorRT engines in strict fp32 (TF32 off).
3. **Pick the signer** with the dataset's [signer rule](#signer-rules). Every posed person is
   stored, so the signer can be picked again later on the CPU with `derive`.
4. **Commit** each finished video atomically; an interrupted run resumes where it stopped.
   [`slp-pose parity`](#parity) checks the pipeline against the official mmdet/mmpose code.

## 3. Outputs

```
<out root>/
  poses/<video_id>.npy         the signer, per frame
  persons/<video_id>.npz       the raw record: every candidate and posed person
  video_meta.csv               one row per committed video
  video_meta_<name>.csv        the same for the videos of another split file (BOBSL: video_meta_manual.csv)
  video_ids.csv                every video of the dataset
  subs/<video_id>.<track>.vtt  subtitles (slp-pose subs)
  splits*.csv                  the dataset's split files (slp-pose splits)
  .state/                      logs/, done/, meta/, failed.jsonl, gpu.csv, captions.json, lock
```

- **`poses/<vid>.npy`**: `(T, 133, 3)` float32. `T` is the number of decoded frames (ffprobe's
  `nb_frames`, or the count the dataset states for a source truncated as distributed), at the
  native frame rate. Keypoints are in COCO-WholeBody order: 0–16 body, 17–22 feet, 23–90 face,
  91–111 left hand, 112–132 right hand (the person's own left and right).
  Channel 0 is `x / W`, channel 1 is `y / H` of the original frame, channel 2 is the raw DWPose
  SimCC score (not clipped; it can exceed 1). A frame without a signer is all zeros.
- **`persons/<vid>.npz`**: every person candidate (detector score > 0.1) with its box, score and
  set flags; the keypoints of the posed people (up to 3 per frame); the signer's index per frame
  (`kpt_primary`, -1 = none); the meta row; and the provenance (git sha, hashes, engine and
  checkpoint sha256, library versions, GPU). `poses/` and the meta row are computed from it.
- **`video_meta.csv`**: the columns
  `video_id,duration_s,width,height,caption_source,undetected_ratio,empty_runs,handless_runs,switch_runs,default_signing_runs,copy_of,unreliable_spans`,
  byte-identical to what misaligned-slt's `save_video_meta` writes:
  - `width`, `height`: the original frame size; `duration_s = T * den / num` from the exact frame
    rate, so `T / duration_s` gives the fps back.
  - `caption_source`: `human` for a video with subtitles, `none` without, once `subs` has run.
  - `undetected_ratio`: the share of frames with nobody in the detector's count set.
  - `empty_runs`: frames whose `poses/` row is not a real body (no signer, or too few joints).
  - `handless_runs`: frames whose `poses/` row shows no hand.
  - `switch_runs`, `default_signing_runs`: always `[]`. They mark misaligned-slt's own signer switch,
    which never runs on these outputs (the dataset's signer rule picks the signer).
  - `copy_of`: written by misaligned-slt's copy check. slp-pose never computes it and keeps it when
    it rewrites the file (blank for a new video). When `subs` changes the subtitles of the file's
    track, it blanks it, as misaligned-slt's own subs stage does: then run the copy check again.
  - `unreliable_spans`: `[start_s, end_s]` spans in seconds where the file's subtitle track does not
    hold the signing (`[]` = checked, none; blank = unknown). misaligned-slt quarantines every
    caption unit that overlaps one and reads its frames as unlabelled. Only BOBSL has such spans
    ([section 6](#6-built-in-datasets)).
  - Any other column, e.g. one that a newer misaligned-slt writes, is kept after these.

  The run columns are JSON lists of half-open `[a, b]` frame runs (`[]` = none), computed like
  misaligned-slt's `prepare_yt25.py` at `meta.MSLT_COMMIT`, at the video's own rate. misaligned-slt
  cuts empty runs out of the stream and quarantines captions that lie mostly in handless runs.
  `slp_pose/meta.py` has the exact definitions and constants. `masked_runs` (frames where a second
  real body moves its arms) is not a column, because misaligned-slt does not read it; it stays in
  the meta row inside `persons/` and `.state/meta/`.
- **`video_meta_<name>.csv`**: one file for each split file after the first of the dataset's
  `split_tracks` ([section 8](#subtitles-and-splits)), each a benchmark of its own. BOBSL has
  `video_meta_manual.csv` for `splits_manual.csv` and the `en-manual` track: the same columns as
  `video_meta.csv` for the 55 videos of that split file, with their own `copy_of` and
  `unreliable_spans`.
- **`subs/<vid>.<track>.vtt`**: the video's subtitles from the dataset's own annotations, as
  WebVTT. `<track>` is `en` (misaligned-slt's default) or another track such as BOBSL's
  `en-manual`. A video without subtitles has no file.
- **`video_ids.csv`**: `video_id,split,source`, then the dataset's own columns. `source` is relative
  to the data root (absolute for a file outside it). It is rewritten at every `extract` and `subs`.

## 4. Installation

Tested with Python 3.11 on the CPU side and with Python 3.8 on 2x RTX 3090 Ti, the environment the
built-in datasets were extracted with.

### Prerequisites

- Linux x86_64 with glibc >= 2.28 and an NVIDIA driver >= 535 (for TensorRT 10).
- No CUDA toolkit: the pip wheels bring their own CUDA libraries, on any GPU (H100 included).
- `ffmpeg` built with libx264, `ffprobe` (Debian/Ubuntu: `sudo apt-get install -y ffmpeg`) and git.
- Python 3.11 on `PATH` or as `PYTHON=/path/to/python3.11`, or [uv](https://docs.astral.sh/uv/),
  from which the setup script installs it.

### Environment

`scripts/setup_env.sh` builds `.venv` from the hash-pinned lock file `requirements.txt` (about 7 GB
of downloads, 8.5 GB on disk), installs this checkout in editable mode (`.venv/bin/slp-pose`) and
runs `scripts/check_env.py`. Each step skips what is done: after a failure, fix it and run it again.
- `--venv DIR`: create the venv somewhere else.
- `--python 3.8`: install `requirements-py38.txt` (TensorRT 10.13.3.9) instead.

### Checkpoints

`.venv/bin/slp-pose fetch-models` downloads YOLOX-L 640 from the mmdetection model zoo (217 MB) and
DWPose-L `dw-ll_ucoco_384.pth` (407 MB) from its Hugging Face mirror
(`https://huggingface.co/yzd-v/DWPose`), and checks their sha256. Without internet access, copy the
files from elsewhere (DWPose-L is also linked from the
[DWPose README](https://github.com/IDEA-Research/DWPose)) and run
`slp-pose fetch-models --no-download --from <file or directory>` (`--from` can be repeated).
The models directory is the first of: `--models-dir`, `$SLP_POSE_MODELS`, `<checkout>/models` when
running from a source checkout, `${XDG_CACHE_HOME:-~/.cache}/slp_pose/models`.

### TensorRT engines

`.venv/bin/slp-pose build-engines --gpu 0` exports ONNX into `<models dir>/onnx/` and builds fp32
engines into `<models dir>/engines/`. Existing engines are kept unless you pass `--force`.
- An engine is tied to the GPU model and the TensorRT version: build the engines on the machine
  that runs `extract`. One `extract` run uses one engine set on all its GPUs.
- Do not rebuild in the middle of a dataset: the engine sha256 is part of every video's
  extraction hash, so after a rebuild a resumed `extract` re-extracts every video.

### Checking the setup

`.venv/bin/python scripts/check_env.py` checks the library versions, ffmpeg, the driver and that
torch has CUDA code for every GPU (`--arch 9.0`: also for a GPU not in the machine); run it again
after changing a GPU or the driver. Before the first extraction on a new GPU, run a short
[parity](#parity) check:

```bash
.venv/bin/slp-pose parity --dataset auslan_news --videos 6 --windows 2 --window-s 20 --full-videos 0 --gpu 0 --jitter batched_det
```

### Installing with pip only

`slp-pose` is not on PyPI. To install it from a clone into a Python 3.11 venv without the lock
files:

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

mmcv comes from the OpenMMLab wheels; `chumpy` (listed by mmpose, never imported) builds only
without build isolation; mmengine needs `setuptools<82`. `slp_pose` checks the exact TensorRT
version at start-up: 10.16.1.11 on Python 3.11, 10.13.3.9 on 3.8–3.10.

## 5. Commands

`slp-pose <command>` and `python -m slp_pose <command>` are the same; `--help` lists every option.

| Command | What it does | Key options |
|---|---|---|
| `fetch-models` | download and check the two checkpoints | `--from`, `--no-download` |
| `build-engines` | export ONNX and build the TensorRT engines | `--gpu`, `--force` |
| `list-videos` | check a dataset without a GPU; print its rules and videos per split | `--list` |
| [`extract`](#extract) | pose every video and pick the signer (resumable) | `--gpus`, `--videos`, `--limit` |
| [`check`](#check) | verify the committed videos | `--videos`, `--no-crc` |
| [`derive`](#derive) | pick the signer again on the CPU from `persons/` | `--primary-rule`, `--videos` |
| [`render`](#render-and-render-done) | draw the keypoints over a window of a video | `--start-s`, `--full`, `--persons` |
| [`render-done`](#render-and-render-done) | render every committed video in full | `--follow`, `--jobs` |
| [`parity`](#parity) | compare with the official per-image code | `--videos`, `--gpu`, `--jitter` |
| [`subs`](#subs-splits-and-meta) | write `subs/`, `caption_source`, the split files and the metadata files | |
| [`splits`](#subs-splits-and-meta) | write the split files | |
| [`meta`](#subs-splits-and-meta) | rewrite the metadata files without reading `persons/` or `poses/` | |

Common options: `--dataset NAME_OR_FILE` (a built-in or installed name, or a dataset `.py` file;
default `bobsl`), `--data-root` (default `$SLP_POSE_DATA`, else `./data`), `--out-root` (default
the dataset's output root) and `--models-dir` ([Checkpoints](#checkpoints)). GPUs are numbered in
PCI bus order, as in `nvidia-smi`. Exit codes: 0 success; 1 failure (failed videos, `check`
problems, a failed parity gate, missing models); 2 bad arguments or dataset; 130 interrupted.

### extract

- **Workers:** one process per GPU (`--gpus`, default `0,1`); videos run in split order, largest
  file first. `--videos ID ...` and `--limit N` restrict the run (pass an id that starts with `-`
  as `--videos=<id>`); `--backend torch` uses PyTorch instead of TensorRT (slower).
- **Commits:** a finished video is committed atomically, then given the final signer rule.
  The metadata files are merged at the start, every 50 commits or 30 minutes, and at the end.
- **Resume:** run the same command again. Committed videos are skipped; a changed signer rule,
  count-set threshold or meta definition re-derives a video on the CPU; changed engines,
  checkpoints, configs, candidate or pose-set thresholds, backend or extraction rule re-extract it.
- **Stop:** Ctrl-C, or SIGTERM to the parent (`pgrep -af 'slp.pose extract'`). Workers stop after
  their current chunk, nothing half-written is kept, and the next run continues.
- **Failures:** a video gets 2 attempts, each failure a line in `.state/failed.jsonl`. A worker that
  crashes or is silent for 10 minutes is restarted (up to 3 times an hour per GPU). The exit code
  is 1 when a video failed or is still pending.
- **Frame count:** the decoded count must equal ffprobe's `nb_frames` (or the dataset's
  `Video.expected_frames`); `--allow-frame-mismatch` accepts any count for the whole run.
- **Read faults:** a file that fails a verification read is read again, up to 3 reads in all;
  each retry is a WARNING line.
- **Decoder errors:** libav's and OpenCV's messages go to `.state/logs/decode_gpu<N>_<time>.log`.
- **Lock:** one `extract` or `derive` at a time per output root (`.state/lock`).

Start long runs detached:
```bash
setsid nohup slp-pose extract --dataset auslan_news --gpus 0,1 > extract_stdout.log 2>&1 < /dev/null &
```

### check

Checks every committed video (or `--videos ID ...`) and exits with 1 when any has a problem:
- every member of `persons/` against its CRC-32 (`--no-crc` skips this read);
- the invariants between `poses/`, `persons/`, the meta row and the done marker;
- that the video was derived with the dataset's final rule and meta definitions (else: `derive`);
- that decoding it printed no decoder errors, unless the dataset declares the source damaged;
- frames where the signer's box jumps, with `render` commands for the windows worth reviewing.

### derive

`slp-pose derive --dataset bobsl --primary-rule signer_track_bobsl` picks the signer again and
recounts people from `persons/` on the CPU, then rewrites `persons/`, the meta row, the metadata
files and `poses/` (only when its rows change). Detections and keypoints stay as they
are. `--videos ID ...` limits it; the default is every committed video.
- A video fails when a per-frame rule picks a person who was never posed (that rule needs a
  re-extraction) or when its `persons/` fails the CRC check (delete `.state/done/<vid>.json` and
  run `extract` to re-extract it).
- A rule other than the dataset's `primary_rule` is flagged by `check` and undone by the next
  `extract`; to change the rule for good, see [Signer rules](#signer-rules).

### render and render-done

`slp-pose render VIDEO_ID --start-s 750 --duration-s 30` draws the saved keypoints over the source
video on the CPU into `<out root>/renders/<vid>_<start frame>.mp4` (the whole video into
`<vid>_full.mp4` with `--full`; `--out`), as H.264 at the exact source frame rate. `--persons` also
draws the other posed people and the candidate boxes.

`render-done` renders every committed video in full into `<out root>_vis/<vid>.mp4` (`--vis-dir`),
`--jobs` (default 3) at a time at nice +10, without GPUs and without writing into the output root,
so it can run beside `extract`. It draws every person (`--no-persons`: the signer only) and renders
a video again only when what it draws changes. `--follow` looks for new commits every `--poll-s`
seconds (default 300) until the extraction has stopped for 15 minutes:
```bash
setsid nohup slp-pose render-done --dataset auslan_news --follow > render_done_stdout.log 2>&1 < /dev/null &
```

### parity

`slp-pose parity --dataset bobsl --gpu 0 --jitter batched_det` compares `slp-pose` with the official
per-image `inference_detector` + `inference_topdown` in strict fp32 on the same decoded frames: by
default 20 random videos x 3 windows of 30 s plus 2 whole videos (`--videos`, `--windows`,
`--window-s`, `--full-videos`, `--seed`). The report goes to
`<out root>_dev/parity/parity_<backend>.json` and `.md` (`--out`); the exit code is 0 when the gate
passes (thresholds: `parity.ParityThresholds`). Use `--jitter batched_det` at 720p: it measures how
much the official code differs from itself and raises the keypoint limits to 1.5x that. To gate a
finished report again without a GPU:
```bash
python -c "from slp_pose import parity; r = parity.load_report('REPORT_DIR/parity_trt.json'); print(parity.gate(r)); parity.write_report(r, 'OUT_DIR')"
```

### subs, splits and meta

`subs` writes each committed video's subtitles to `subs/<vid>.<track>.vtt`, deletes `.vtt` files no
video produces any more, fills `caption_source` and rewrites `video_ids.csv`, then writes the split
files and the metadata files as `splits` and `meta` do. Cues with an empty text, or whose end is not
after their start, are dropped and counted in the log. It takes the output root's lock. Run it
again after the annotations or videos change.

`splits` writes the dataset's split files into its output root, each a CSV `video_id,split`:
`splits.csv`, or one `splits_<name>.csv` per split the dataset declares (BOBSL:
`splits_automatic.csv` and `splits_manual.csv`), and removes other `splits*.csv` files there. When
the root has a `video_meta.csv`, it then rewrites the metadata files, which list the videos of the
split files.

`meta` rewrites the metadata files (`video_meta.csv` and every `video_meta_<name>.csv`, and removes
other `video_meta_*.csv` files) from the output root alone: the meta row columns from
`video_meta.csv`, which `extract` and `derive` write from `.state/`; the videos of each file from
the split files; `unreliable_spans` from `subs/` and the annotation files the dataset reads for it.
It reads neither `persons/` nor `poses/` and needs no videos, so it takes seconds on any copy of an
output root. The metadata columns are not part of the record (`meta.META_VERSION` does not cover
them), so a change of these columns alone needs no `derive`, and `check` is not affected:
```bash
slp-pose meta --dataset bobsl --data-root /path/to/data --out-root /path/to/data/BOBSL/dwpose
```
It keeps the columns it does not compute (`copy_of`, and any column it does not know). Where an
`unreliable_spans` value cannot be computed because its annotation files are missing, the file
keeps its earlier value (blank at first) and the log names the missing files. While a file holds a
blank `unreliable_spans` cell, `meta`, `splits` and `subs` exit with 1: misaligned-slt reads a blank
cell as no span, so it would read the dropped cues of those videos as gaps.

## 6. Built-in datasets

| | BOBSL (`bobsl`) | Auslan-Daily News V1 + V2 (`auslan_news`) | How2Sign (`how2sign`) |
|---|---|---|---|
| Videos | 2,212 episodes, 444x444, 25 fps | 125 broadcasts, 1280x720 (six 1024x576), 29.97 or 25 fps | 2,529 `rgb_front`, 1280x720 (three 1920x1080), own rate (mostly 24 fps) |
| Video id | episode id, e.g. `5085344787448740525` | `abc_news_YYYY-MM-DD` | `VIDEO_NAME`, e.g. `--7E2sU6zP4-5-rgb_front` |
| Splits | automatic: val 32, test 250, train 1658, challenge_test 272; manual: dev 4, test 15, train 36 | dev 8, test 16, train 101 | val 132, test 184, train 2213 |
| Subtitles | `en` (automatic), `en-manual` (55 episodes) | `en` | `en` |
| Metadata files | `video_meta.csv` (automatic split, `en`), `video_meta_manual.csv` (manual split, `en-manual`) | `video_meta.csv` | `video_meta.csv` |
| `video_ids.csv` extras | `manual_split` | `subset`, `annotation_name` | none |
| Rules (extraction / final) | `largest_bbox` / `signer_track_bobsl` | `right_largest` / `signer_track_right` | `largest_bbox` / `largest_bbox` |
| Output root | `<data root>/BOBSL/dwpose/` | `<data root>/Auslan-Daily/dwpose/` | `<data root>/How2Sign/dwpose/` |

The signer kept is BOBSL's interpreter, never a programme person; Auslan News's interpreter in the
right-hand panel or full screen, or an on-site or picture-in-picture one while the panel is gone,
never a newsreader; and How2Sign's one person in front of a green screen. BOBSL's manual split
re-splits its 55 manually aligned episodes by interpreter, so like the official split it is
signer-independent: score a model trained on it on its own `test` only. On 2x RTX 3090 Ti,
extraction took about 8.5 days for BOBSL and about 10 hours each for Auslan News and How2Sign.

BOBSL's manual metadata file marks where its subtitles do not hold the signing (`unreliable_spans` of
`video_meta_manual.csv`): every cue of the manual files that slp-pose drops for a quality tag such as
`[NOT-SIGNED]`. Only the original manual files hold these cues, so `subs` and `meta` read
`<data root>/BOBSL/manual_annotations/signing_aligned_subtitles/`; without them the column stays
blank (unknown).

The automatic track (`video_meta.csv`), Auslan News and How2Sign have no such spans (`[]`). SAT's
collapsed alignments (automatic cues of 4 frames, about a quarter of them) need none: misaligned-slt
itself marks the gap around a cue shorter than its `subtitles.min_duration_s` as unlabelled, in every
corpus.

### Data layout

```
<data root>/BOBSL/original_data/videos/mp4/<episode id>.mp4
<data root>/BOBSL/original_data/metadata/subset2episode.json      {split: [episode id, ...]}
<data root>/BOBSL/automatic_annotations/signing_aligned_subtitles/auto_sat_aligned/<episode id>.vtt
<data root>/BOBSL/manual_annotations/signing_aligned_subtitles/<episode id>.vtt

<data root>/Auslan-Daily/News_V1/ABC News With Auslan - D M YYYY.mp4
<data root>/Auslan-Daily/News_V1_Annotation.xlsx                  V1 subtitles
<data root>/Auslan-Daily/News_V2/ABC News With Auslan - D M YYYY.mp4
<data root>/Auslan-Daily/News_V2_Aligned_VTTS/ABC News With Auslan - D M YYYY.vtt   V2 subtitles

<data root>/How2Sign/{val,test,train}/<VIDEO_NAME>.mp4            the frontal raw videos
<data root>/How2Sign/how2sign_realigned_{val,test,train}.csv      sentence timings
```

Only the TV news part of Auslan-Daily is used. The BOBSL licence keeps its outputs local.

### Known damaged sources

- **BOBSL `5976836975695491321`** is damaged in the official v1.4 archive (decoders stop at frame
  71,837 of 90,577). Its outputs come from a repaired copy whose frames [71837, 72447) and
  [73257, 73934) are black (stored as empty). With the official file, `extract` fails it on its
  frame count: build such a copy, or run it alone with `--videos 5976836975695491321
  --allow-frame-mismatch` for its 71,837 decodable frames (`check` then reports decoder errors).
- **How2Sign `cH27PiYX6gI-9-rgb_front` and `CH7AviIr0-0-8-rgb_front`** (train) are truncated as
  distributed: they decode to 4,088 of 4,340 and 2,133 of 3,792 frames (`how2sign.TRUNCATED`). The
  first loses no annotated sentence; the second keeps 13 of its 26.
- **How2Sign `G3RvsnzQrXQ-10-rgb_front`** and **Auslan `abc_news_2022-09-11`** are damaged as
  downloaded (one block each, away from the signer; poses unaffected). These and the truncated
  videos are in `how2sign.DAMAGED` and `auslan_news.DAMAGED`, so `check` expects their errors.

The dataset files (`slp_pose/datasets/bobsl.py`, `auslan_news.py`, `how2sign.py`) document the
details: subtitle sources and dropped cues, BOBSL's manual split, and each signer rule.

## 7. Using the outputs in misaligned-slt

An output root has the layout of a misaligned-slt language root: `poses/`, `video_meta.csv` and,
after `slp-pose subs`, `subs/`, the split files and the other metadata files. Each benchmark is
separate: train and evaluate on one at a time, with its own split file, the subtitle track that goes
with it and its own metadata file. The two BOBSL benchmarks share one root, so the manual one names
its metadata file (misaligned-slt's `video_meta`; the default is `<root>/video_meta.csv`).

| Benchmark | `root` | `video_meta` | `splits.signverse_csv` | `subtitles.preferred_suffixes` |
|---|---|---|---|---|
| BOBSL, automatic alignment (official split) | `data/BOBSL/dwpose` | (default) | `data/BOBSL/dwpose/splits_automatic.csv` | `[.en.vtt]` |
| BOBSL, manual alignment (55 episodes) | `data/BOBSL/dwpose` | `data/BOBSL/dwpose/video_meta_manual.csv` | `data/BOBSL/dwpose/splits_manual.csv` | `[.en-manual.vtt]` |
| Auslan-Daily News | `data/Auslan-Daily/dwpose` | (default) | `data/Auslan-Daily/dwpose/splits.csv` | `[.en.vtt]` |
| How2Sign | `data/How2Sign/dwpose` | (default) | `data/How2Sign/dwpose/splits.csv` | `[.en.vtt]` |

For example, BOBSL with its manual alignment in misaligned-slt's `configs/data.yaml`:

```yaml
active_languages: [bobsl_manual]
languages:
    bobsl_manual:
        name: BSL (BOBSL, manual alignment, re-split by interpreter)
        target_lang: en_XX
        pretrained_slt: {ar: checkpoints/openasl_pose_only_slt.pth, dlm: checkpoints/openasl_pose_only_slt.pth}
        root: /path/to/data/BOBSL/dwpose          # the slp-pose output root
        video_meta: /path/to/data/BOBSL/dwpose/video_meta_manual.csv
        splits:
            signverse_csv: /path/to/data/BOBSL/dwpose/splits_manual.csv
        subtitles:
            preferred_suffixes: [.en-manual.vtt]
        pose: {width: 444, height: 444}    # fallbacks only: the metadata file has the real ones
```

Things to know:
- Each video carries its own rate (`T / duration_s`); Auslan News and How2Sign mix rates, so
  misaligned-slt must accept mixed rates.
- `val` is read as `dev`; unlisted videos and other splits (BOBSL's `challenge_test`) are left out.
- Videos whose `caption_source` is `none` are dropped.
- misaligned-slt's copy check writes `copy_of` into each metadata file; slp-pose keeps it, until
  `subs` changes the subtitles of that file's track ([section 3](#3-outputs)).
- Run `subs` or `meta` for BOBSL where the original manual annotation files are, else the
  `unreliable_spans` of `video_meta_manual.csv` stay blank (unknown) and the command exits with 1.
- The run columns follow `meta.MSLT_COMMIT`: when misaligned-slt changes them, port the change to
  `slp_pose/meta.py`, bump `meta.META_VERSION` and run `slp-pose derive`. When it changes only the
  columns of the file, port the change to the CSV part of `slp_pose/meta.py` and run
  `slp-pose meta`.

## 8. Adding a dataset

A dataset is one Python file with one `Dataset` subclass:

```python
from slp_pose.datasets import Dataset, Video, safe_id


class MyDataset(Dataset):
    name = 'my_dataset'
    primary_rule = 'largest_bbox'

    def videos(self):
        folder = self.data_root / self.name / 'videos'
        return [Video(video_id=safe_id(p.stem), path=p) for p in sorted(folder.glob('*.mp4'))]
```

[`examples/my_dataset.py`](examples/my_dataset.py) is the complete version, with splits from a CSV:

```bash
slp-pose list-videos --dataset examples/my_dataset.py --data-root /path/to/data --list
slp-pose extract --dataset examples/my_dataset.py --data-root /path/to/data --gpus 0
```

The outputs go to `/path/to/data/my_dataset/dwpose/`. `--dataset` takes a `.py` file that defines
(or `@register`s) exactly one `Dataset` subclass, a built-in name (`bobsl`, `auslan_news`,
`how2sign`), `package.module:ClassName`, or the name of an installed plugin: an entry point in the
group `slp_pose.datasets`, declared in the plugin package's `pyproject.toml`:

```toml
[project.entry-points."slp_pose.datasets"]
my_dataset = "my_package.my_dataset:MyDataset"   # or a module that defines exactly one Dataset
```

### The contract

The framework constructs `cls(data_root)`, checks it and its videos (`slp_pose/datasets/base.py`):

| Member | Meaning |
|---|---|
| `name` | `[A-Za-z0-9._-]+`; names the registry entry, the logs and the default output root. |
| `videos()` | Every video as `Video(video_id, path, split=None, info={})`; `info` = extra `video_ids.csv` columns. |
| `out_root()` | The output root; default `<data root>/<name>/dwpose`. |
| `primary_rule` | The final signer rule behind `poses/`, per-frame or video-level; default `largest_bbox`. |
| `extraction_rule` | The per-frame rule the GPU workers pose with; `None` = `primary_rule` (then per-frame). |
| `rules` | The dataset's own signer rules ([below](#rules-of-your-own)); default none. |
| `split_order` | When set, splits are extracted in this order, and every split must be one of them. |
| `captions(video, fps)` | The video's subtitles ([below](#subtitles-and-splits)); default none. |
| `caption_source` | What the subtitles are: `human` (default), `mt` or `shard`. |
| `splits(videos)` | The split files; default one file from each video's `split`. |
| `split_tracks` | The subtitle track of each split file, as `(split name, track)` pairs, each with its own metadata file ([below](#subtitles-and-splits)); default `(('', 'en'),)`. |
| `unreliable_spans(video_id, track, cues)` | Where a track does not hold the signing ([below](#subtitles-and-splits)); default `[]`. |

A `video_id` is the output file stem: `[A-Za-z0-9._-]+`, unique and not ending in `_segment_<n>`
(`safe_id()` replaces other characters with `_`); its `path` must exist. For misaligned-slt, avoid
dots in ids: it finds subtitles by cutting `<vid>.<lang>.vtt` at the first dot. Two optional
`Video` fields, never hashed, describe damaged sources (list them in the dataset's file):
- `expected_frames`: the number of frames a truncated source decodes to; the worker then requires
  exactly that count instead of ffprobe's.
- `known_damage`: why a source prints decoder errors on every read; `check` then logs them as
  expected instead of reporting them.

### Subtitles and splits

A dataset with subtitles implements `captions(video, fps)` and returns
`{track: [Cue(start, end, text), ...]}`, times in seconds of the video; `fps` is the exact frame
rate of the committed poses (a `Fraction`), for annotations timed in frames. Name the default track
`en` and another version e.g. `en-manual` (letters, digits and `-`). Do here any cleaning only your
dataset needs (e.g. dropping cues the annotators marked as unusable); misaligned-slt cleans the
text itself. A dataset with several split files, e.g. one per subtitle track, overrides
`splits(videos)` and returns `{name: {video_id: split}}`; each name becomes `splits_<name>.csv`.

Each split file goes with one subtitle track, which `split_tracks` names, e.g. BOBSL's
`(('automatic', 'en'), ('manual', 'en-manual'))`. The first pair's metadata file is
`video_meta.csv` (every committed video); every other pair gets `video_meta_<name>.csv` with the
videos of `splits_<name>.csv`. A dataset whose annotations mark cues whose signing does not match
drops them in `captions` and returns their spans from `unreliable_spans(video_id, track, cues)`:
`(start, end)` in seconds, any order. `cues` is the track as `subs` wrote it. Return `[]` when
there are none (the default). When the spans cannot be known, return `None` or let the
`FileNotFoundError` of a missing source through: the column then keeps its earlier value, and the
log names the missing file. `meta` calls it from the output root alone, where the videos may be
absent, so it gets the video id, not the `Video`.

`read_vtt` reads a WebVTT file:

```python
from slp_pose.datasets import read_vtt

    def captions(self, video, fps):
        path = self.data_root / self.name / 'subtitles' / f'{video.video_id}.vtt'
        return {'en': read_vtt(path)} if path.is_file() else None
```

### Signer rules

The workers pose up to 3 people per frame and store them all: the extraction rule's pick, the
largest box, the highest-scoring box, then the rest of the pose set (detector score > 0.3) by score.
The final rule picks the signer among them on the CPU, so choose an extraction rule that poses the
signer. `slp_pose/signer.py` and the dataset files describe each rule in full.

| Rule | Kind | Picks |
|---|---|---|
| `largest_bbox` | per-frame | the largest box |
| `highest_score` | per-frame | the highest detector score |
| `signer_track` | video-level | the person at the spot present in at least half the frames, or a steady person elsewhere |
| `signer_track_bobsl` (BOBSL) | video-level | `signer_track`; elsewhere only someone who sits and looks like the interpreter |
| `right_largest` (Auslan News) | per-frame | the largest box centred in the right 40 % of the frame, else the largest |
| `signer_track_right` (Auslan News) | video-level | `signer_track` preferring the right-hand spot; elsewhere only someone who signs |

- With a per-frame final rule, only frames with nobody in the pose set are zeros; a video-level
  rule also leaves zeros wherever the signer is absent.
- Changing `primary_rule` costs only a CPU derive as long as the extraction rule stays the same,
  so set `extraction_rule` explicitly before changing a `primary_rule` it defaults to. Changing the
  extraction rule re-extracts, except a switch between `largest_bbox` and `highest_score`.

### Rules of your own

A rule that only your dataset needs stays in your dataset file: define it there, list it in
`rules`, and name it in `primary_rule`, `extraction_rule` or `derive --primary-rule`. From
[`examples/custom_rules.py`](examples/custom_rules.py), whose interpreter stands on the left:

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
  index of the signer.
- **Video-level:** `VideoRule(name, choose, params)`. `choose(people)` gets the posed people of a
  whole video (`PosedPeople`: boxes, scores, per-frame offsets, keypoints, fps, frame size) and
  returns a `SignerChoice`: per frame the signer's position among the posed people (-1 = nobody)
  and the interpreter's region (NaN = none; the meta row's `masked_runs` counts the people inside
  it).
- `pick` and `choose` can be any callable. Rule names match `[A-Za-z0-9._-]+`, are unique, and are
  not a framework rule's name. `slp_pose/signer.py` has building blocks for video-level rules
  (`track_rule`, `SignerTrackParams`, the spot and motion steps), as `examples/custom_rules.py`
  shows.
- The name and `params` (a JSON-able dict) are hashed, the code is not: changing them for the
  extraction rule re-extracts the committed videos, for the final rule it re-derives them. So when
  you change a rule's code, change its `params` or its name too.
- The GPU workers import your dataset to find its per-frame rule: define the class at the top level
  of a `.py` file, a module or a plugin.
- Check your outputs with `slp-pose check --dataset <your file>`, which knows your rules through
  your dataset (in Python: `record.check(root, video_id, dataset=dataset)`).

## 9. Development

`scripts/lock_requirements.sh` (needs uv) regenerates `requirements.txt` with the newest versions
within its constraints (mmdet 3.3.0, so mmcv < 2.2 on torch 2.1; numpy < 2; TensorRT < 11;
setuptools < 82). Review the diff and build a venv from it (`scripts/setup_env.sh --venv DIR`). A
new TensorRT version also goes into `slp_pose/env.py` (`TENSORRT_VERSIONS`) and needs rebuilt,
parity-checked engines.

| Path | What |
|---|---|
| `slp_pose/cli.py`, `__main__.py`, `settings.py`, `paths.py` | the command line; thresholds, batching, model and data paths |
| `slp_pose/datasets/` | the dataset contract (`base.py`), the registry and the built-in datasets |
| `slp_pose/env.py`, `fetch.py`, `engines.py` | start-up checks; `fetch-models`; ONNX export and TensorRT engines |
| `slp_pose/video.py`, `prep.py`, `detpost.py`, `posepost.py` | decoding, the exact letterbox and crops, post-processing |
| `slp_pose/select.py`, `signer.py`, `rules.py` | the signer rules and their building blocks |
| `slp_pose/record.py`, `meta.py`, `types.py` | the `persons/` record, commit, resume, derive, check; the metadata files |
| `slp_pose/worker.py`, `run.py`, `decodelog.py` | the GPU workers, scheduling and supervision, decode logs |
| `slp_pose/render.py`, `render_batch.py`, `subs.py`, `parity.py` | `render`, `render-done`, `subs`, `splits`, `meta`, `parity` |
| `slp_pose/model_configs/` | the YOLOX and DWPose configs (never edited: their sha256 is in the extraction hash) |
| `examples/`, `scripts/` | the example datasets; `setup_env.sh`, `check_env.py`, `lock_requirements.sh` |
| `requirements.txt`, `requirements-py38.txt` | hash-pinned locks for Python 3.11 and 3.8 |
