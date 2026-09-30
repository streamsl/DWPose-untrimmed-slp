# Reset model path:

./DWPose/wholebody.py
Line: 29-32

# Infer a folder : run with a folder of images

# Infer one sample: run with only one video sample

## Change the input data / data folder

## Set Line: 110 to show the pose visualization and change the save folder (main.py Line 112)

## Set batch size of the detection model (Line 68) and pose estimation model (Line 69).

## Current batch size is fully used in one GPU 3090 (24 GB).

RUN:
python main.py
