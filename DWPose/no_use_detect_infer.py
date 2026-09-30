import cv2
import torch
import numpy as np

import mmdet
from mmdet.apis import DetInferencer
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env as collect_base_env
from mmpose.apis import MMPoseInferencer
from mmpose_inferencer import MMPoseInferencer

# Show the structure of result dict
from rich.pretty import pprint
import warnings
warnings.filterwarnings('ignore')


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMDetection'] = f'{mmdet.__version__}+{get_git_hash()[:7]}'
    return env_info

def dis_env():
    """Display the current working environment."""
    for name, val in collect_env().items():
        print(f'{name}: {val}')

def print_all_supported_models():
    """models is the list of all supported models and automatically print all of them"""
    models = DetInferencer.list_models('mmdet')


# Load the model configuration and weights
model_config = '/home/tianchenguo/youfor2032-cv/DWPose_usage/yolox_config/yolox_l_8xb8-300e_coco.py'
model_weights = "/home/tianchenguo/youfor2032-cv/DWPose_usage/yolox_config/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth"


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_path, downsize_ratio=1):
        self.nFrames = None
        self.fps = None
        self.video_path = video_path
        # video will be BGR & [N, H, W, C]
        self.video = self.load_video(downsize_ratio)

    def __len__(self):
        return self.nFrames

    def get_video_frames(self):
        """
        Get() function returns the processed videos in the image form.
        That is to say, return a series of images extracted from video frames.
        :return: [N, H, W, C]
        N: the number of frames (at least fps * 5)
        H: the height of the video
        W: the width of the video
        C: the channel of the video, which equals 3
        """
        return self.video

    def get_fps(self):
        return self.fps

    def get_slice(self, st, ed):
        return self.video[max(st, 0): min(ed, self.nFrames)]

    def load_video(self, downsize_ratio=1):
        vid = cv2.VideoCapture(self.video_path)
        if not vid.isOpened():
            raise ValueError("Error opening video")

        Height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.nFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = vid.get(cv2.CAP_PROP_FPS)  # get video fps

        if self.nFrames <= 0:
            self.nFrames = self.fps * 5

        if downsize_ratio != 1:
            Height = int(Height * downsize_ratio)
            Width = int(Width * downsize_ratio)

        video = []
        count = 0
        while vid.isOpened():
            success, frame = vid.read()
            if not success:
                break
            if downsize_ratio != 1:
                frame = cv2.resize(frame, dsize=(Width, Height), interpolation=cv2.INTER_LINEAR)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video.append(frame)
            count += 1

        vid.release()
        return video[:count]


def get_bboxes(res):
    # Initialize list to store best bboxes
    best_bboxes = []

    for predictions in res['predictions']:
        labels = np.array(predictions['labels'])
        scores = np.array(predictions['scores'])
        bboxes = np.array(predictions['bboxes'])

        # Mask to select only the bboxes with label 0
        label_mask = labels == 0

        # If no label 0 is found, append None and continue
        if not np.any(label_mask):
            best_bboxes.append([0, 0, 0, 0])
            continue

        # Use masked arrays to directly select the highest score and corresponding bbox
        filtered_scores = np.where(label_mask, scores, -np.inf)
        max_index = np.argmax(filtered_scores)
        best_bbox = np.round(bboxes[max_index]).astype(int)
        best_bboxes.append(best_bbox)

    return best_bboxes


def crop_images(imgs, bboxes):
    """
    Crop the images based on the bounding boxes. Only save the area inside the bounding boxes.

    :param imgs: numpy array of shape [N, H, W, C]
    :param bboxes: numpy array of shape [N, 4] where each row is [x_min, y_min, x_max, y_max]
                   representing the bounding box for the corresponding image.
    :return: List of cropped images, each of shape [H', W', C] where H' and W' depend on the bounding box.
    """
    cropped_images = []

    for img, bbox in zip(imgs, bboxes):
        x_min, y_min, x_max, y_max = bbox

        # Ensure bounding box is within image dimensions
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img.shape[1], x_max)
        y_max = min(img.shape[0], y_max)

        # Crop the image
        cropped_img = img[y_min:y_max, x_min:x_max]

        cropped_images.append(cropped_img)

    return cropped_images


if __name__ == "__main__":

    video_path = "/home/tianchenguo/youfor2032-cv/data/Unitest_Data/pushup/blob_2024-07-18_11-22-28.mov"
    video_data = VideoDataset(video_path)
    video_images = video_data.get_video_frames()  # [N, H, W, C]
    N, H, W, C = np.array(video_images).shape
    video_fps = video_data.get_fps()
    print(f"video frame shape: {np.array(video_images).shape}")

    #cv2.imwrite("/home/tianchenguo/youfor2032-cv/tmp_test/video_image.jpg", video_imgs[0])

    # load the model, yolox_l_8xb8-300e_coco.py
    # inferencer = DetInferencer(model="yolox_l_8x8_300e_coco", device='cuda:0')

    # document: https://mmdetection.readthedocs.io/zh-cn/latest/user_guides/inference.html
    # result = inferencer(video_images, out_dir='/home/tianchenguo/youfor2032-cv/tmp_test/', pred_score_thr=0.8, batch_size=128)

    # print the first 4 results
    # pprint(result, max_length=4)

    # bboxes = get_bboxes(result)  # [N, 4] 4:[x_min, y_min, x_max, y_max]
    # bboxes = []
    # print("The bbox of label 0 with the highest score:", np.array(bboxes).shape)
    #
    # video_cropped_images = crop_images(video_images, bboxes)



    # models = MMPoseInferencer.list_models('mmpose')

    # pose estimation
    inference_topdown = MMPoseInferencer(
        pose2d="/home/tianchenguo/youfor2032-cv/DWPose_usage/dwpose_config/dwpose-l_384x288.py",
        pose2d_weights="/home/tianchenguo/youfor2032-cv/DWPose_usage/dwpose_config/dw-ll_ucoco_384.pth",
        # det_model="/home/tianchenguo/youfor2032-cv/DWPose_usage/yolox_config/yolox_l_8xb8-300e_coco.py",
        # det_weights="/home/tianchenguo/youfor2032-cv/DWPose_usage/yolox_config/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth",
        det_cat_ids=[0],
    )
    print(f"video length: {np.array(video_images).shape}")
    key_points_result_generator = inference_topdown(video_images, device="cuda:0", batch_size=64,
                                                    out_dir="/home/tianchenguo/youfor2032-cv/data_test_demo", show_progress=False)
    key_points_result = next(key_points_result_generator)

    pprint(key_points_result, max_length=4)

    output_video_path = "/home/tianchenguo/youfor2032-cv/data_test_demo/0.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, video_fps, (H, W))



    for frame_idx, frame in enumerate(video_images):
        # Convert frame to the correct color space if needed (e.g., RGB to BGR for OpenCV)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        predictions = key_points_result['predictions'][frame_idx]  # Assuming predictions correspond to frames
        for prediction in predictions:
            first_prediction = prediction
            keypoints = first_prediction['keypoints'][:17]
            keypoint_scores = first_prediction['keypoint_scores'][:17]
            bbox = first_prediction['bbox'][0]
            bbox_score = first_prediction['bbox_score']

            # print(f"bbox: {bbox}, keypoints: {keypoints}")
            # print(f"frame: {frame.shape}")
            x1, y1, x2, y2 = map(int, bbox)

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for kp in keypoints:
                x, y = map(int, kp)
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        # Write frame to the video file
        video_writer.write(frame)

    # Release the VideoWriter object
    video_writer.release()

    print(f"Video saved to {output_video_path}")

