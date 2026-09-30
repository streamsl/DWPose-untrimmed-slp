import os
import cv2
import torch
import numpy as np

from DWPose import Wholebody, DWProcessor
from PIL import Image


def save2npy(data, save_name, o_save_path):
    o_save_path = o_save_path
    save_path = os.path.join(o_save_path, save_name + ".npy")
    data = data.reshape(-1, 133, 3)
    data = data.astype(np.float16)
    np.save(save_path, data)
    print("FINISH", save_path)

def save_images_as_png(detected_images, save_path):
    """
    将 detected_images 列表中的图像按顺序保存为 PNG 格式。

    :param detected_images: list, 存放多个 (H, W, 3) 图像的列表 (numpy 数组)
    :param save_path: str, 保存图像的目录路径
    """
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)

    for i, img_array in enumerate(detected_images):
        # 将 numpy 数组转换为 PIL 图像
        img = Image.fromarray(img_array.astype(np.uint8))

        # 生成文件名
        img_filename = os.path.join(save_path, f"{i:04d}.png")  # 例如 0000.png, 0001.png, ...

        # 保存图像
        img.save(img_filename, format="PNG")



def get_video_info(input_path):
    """
    获取视频的分辨率、帧数和音频信息。
    :param input_path: 输入视频路径
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 获取视频基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频高度
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
    cap.release()
    # 输出视频信息
    print(f"分辨率: {width}x{height}")
    print(f"帧率: {fps} FPS")
    print(f"总帧数: {total_frames}")


def progress_video_dwpose(video_path, o_save_path):
    dwdetector = Wholebody()
    dwprocessor = DWProcessor()

    batch_size_1 = 128  # For bounding box detection, 195
    batch_size_2 = 256  # For pose estimation, 385

    # DWPose
    print(f"-------------video name:{video_path}, DWPose-------------")

    # Track video loading time
    video_images = VideoDataset(video_path).get_video_frames()  # [N, H, W, C]
    N, H, W, C = np.array(video_images).shape


    bboxes_list = []
    detected_images = []
    pose_sequences = []
    real_pose_sequences = []
    pos_score = []
    first_flag = False

    # Get bounding boxes from images (YOLOX)
    for i in range(0, len(video_images), batch_size_1):
        image_batch = video_images[i:min(i + batch_size_1, len(video_images))]
        bboxes_list.extend(dwdetector.get_bboxes(image_batch))

    # Get pose sequences from bounding boxes and images (DWPose)
    for i in range(0, len(video_images), batch_size_2):
        image_batch = video_images[i:min(i + batch_size_2, len(video_images))]
        bbox_batch = bboxes_list[i:min(i + batch_size_2, len(video_images))]
        keypoints_list_i, score_list_i = dwdetector.get_pose_sequence(image_batch, bbox_batch)
        detected_images_i, pose_sequences_i, real_sequences_i = dwprocessor(image_batch, keypoints_list_i, score_list_i,
                                                          bbox_batch)
        detected_images.extend(detected_images_i)

        if not first_flag:
            pose_sequences = pose_sequences_i.copy()
            real_pose_sequences = real_sequences_i.copy()
            first_flag = True
        else:
            pose_sequences = np.concatenate((pose_sequences, pose_sequences_i), axis=0)
            real_pose_sequences = np.concatenate((real_pose_sequences, real_sequences_i), axis=0)

    SHOW = False
    if SHOW:
        show_path = "./tests/detected_images"
        save_images_as_png(detected_images, show_path)

    s_name = video_path.split("/")[-1].split(".")[0]
    save2npy(real_pose_sequences, s_name, o_save_path)


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_path, frame_extraction_ratio=0):
        self.nFrames = None
        self.fps = None
        self.video_path = video_path
        # video will be BGR & [N, H, W, C]
        self.video = self.load_video(frame_extraction_ratio)
        # frame extraction ratio: positive: every i frames, extract 1 frame
        # negative: every frame, extract i frames

    def __len__(self):
        return self.nFrames

    def resize_with_padding(self, image, target_size=(640, 640), color=(114, 114, 114), stride=64):
        if isinstance(target_size, int):
            target_size = (target_size, target_size)

        h, w = image.shape[:2]
        scale = min(target_size[0] / h, target_size[1] / w)  # Scale to fit within target size
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize the original image
        resized_image = cv2.resize(image, (new_w, new_h))

        # Compute padding dimensions
        dw, dh = target_size[1] - new_w, target_size[0] - new_h  # Width and height padding

        # Divide padding into two sides (left-right and top-bottom)
        dw /= 2
        dh /= 2

        # Create a new image with the padding color
        padded_image = np.full((target_size[0], target_size[1], 3), color, dtype=np.uint8)

        # Offset to place the resized image in the center of the padded image
        x_offset = int(dw)
        y_offset = int(dh)

        # Place the resized image in the center of the new padded image
        padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

        return padded_image

    def remove_black_borders(self, image_):
        image = image_ / 255.0  # convert from 0 to 1
        H, W, C = image.shape
        row_sum = np.sum(np.sum(image, axis=1), axis=1)  # find the sum of all rows, sum three channels as well
        column_sum = np.sum(np.sum(image, axis=0), axis=1)

        top = 0
        bottom = H - 1
        left = 0
        right = W - 1

        # find top
        for i in range(H):
            if row_sum[i] <= 30:
                top += 1
            else:
                break

        # find bottom
        for i in range(H-1, 0, -1):
            if row_sum[i] <= 30:
                bottom -= 1
            else:
                break

        # find left
        for i in range(W):
            if column_sum[i] <= 30:
                left += 1
            else:
                break

        # find right
        for i in range(W - 1, 0, -1):
            if column_sum[i] <= 30:
                right -= 1
            else:
                break

        #print(top, bottom, left, right)

        cropped_image = image_[top:bottom+1, left:right+1]

        # Apply the sharpening kernel to the image
        # sharpened_image = cv2.filter2D(cropped_image, -1, kernel)
        #enhanced_image = cv2.convertScaleAbs(cropped_image, alpha=1.5, beta=20)
        #denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        return cropped_image, [top, bottom+1, left, right+1]

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
        print(f"resized videos: {np.array(self.video).shape}")
        return self.video

    def get_fps(self):
        return self.fps

    def get_slice(self, st, ed):
        return self.video[max(st, 0): min(ed, self.nFrames)]

    def load_video(self, frame_extraction_ratio=0):
        vid = cv2.VideoCapture(self.video_path)
        if not vid.isOpened():
            raise ValueError("Error opening video")

        Height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.nFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = vid.get(cv2.CAP_PROP_FPS)  # get video fps
        print(f"original info: N: {self.nFrames}, H: {Height}, W: {Width}, fps: {self.fps}")

        if self.nFrames <= 0:
            self.nFrames = self.fps * 5

        video = []
        count = 0
        remove_black_boarder_flag = False
        first_frame_flag = True
        cropped_shape = []
        while vid.isOpened():
            success, frame = vid.read()
            if not success:
                break
            count += 1

            # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]  # quality between 0 (worst) and 100 (best)
            # _, buffer = cv2.imencode('.jpg', frame, encode_param)
            # frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)  # Decode the compressed frame

            if first_frame_flag:
                first_frame_flag = False
                cropped_image, cropped_shape = self.remove_black_borders(frame)
                cond = (abs(cropped_image.shape[0] - frame.shape[0]) > 10) or (abs(cropped_image.shape[1] - frame.shape[1]) > 10)
                if cond:
                    remove_black_boarder_flag = True

            target_size = (640, 640)  # (new_h, new_w)

            if self.fps > 55:
                if count % 2 == 0:
                    if remove_black_boarder_flag:
                        cropped_top = cropped_shape[0]
                        cropped_bottom = cropped_shape[1]
                        cropped_left = cropped_shape[2]
                        cropped_right = cropped_shape[3]
                        frame = self.resize_with_padding(frame[cropped_top:cropped_bottom, cropped_left:cropped_right], target_size)
                    else:
                        frame = self.resize_with_padding(np.array(frame), target_size)
                    video.append(frame)
            else:
                if remove_black_boarder_flag:
                    cropped_top = cropped_shape[0]
                    cropped_bottom = cropped_shape[1]
                    cropped_left = cropped_shape[2]
                    cropped_right = cropped_shape[3]

                    if frame[cropped_top:cropped_bottom, cropped_left:cropped_right].shape == (0,0,3):
                        cropped_top = 1
                        cropped_bottom = Height
                        cropped_left = 1
                        cropped_right = Width
                    frame = self.resize_with_padding(frame[cropped_top:cropped_bottom, cropped_left:cropped_right], target_size)
                else:
                    frame = self.resize_with_padding(np.array(frame), target_size)
                video.append(frame)

        vid.release()
        if frame_extraction_ratio == 0:
            return video

        # frame extraction techniques
        extracted_video = []
        if frame_extraction_ratio < 0:
            # negative, extract i frames for each frame
            extracted_video = [video[i] for i in range(0, len(video), -frame_extraction_ratio)]
        if frame_extraction_ratio > 0:
            # positive, extract 1 frames for i frame
            extracted_video = [video[i] for i in range(len(video)) if i % frame_extraction_ratio != 0]

        return extracted_video

if __name__ == "__main__":
    # print("Hello, World!")
    # ppp = "/media/xins/xinS1/Dataset/WLASL/WLASL-100-MTV/wlasl_test/s1"
    # # ppp = "/media/xins/xinS1/Cross_View_ISLR/TESTSET/Processed_Test"
    # # get_video_info("/media/xins/xinS1/Cross_View_ISLR/TESTSET/Processed_Test/web7_1734.mp4")
    # # progress_video_dwpose("/media/xins/xinS1/Cross_View_ISLR/TESTSET/Processed_Test/web7_1734.mp4")
    # # print(kkk)
    # # o_save_path = "/media/xins/xinS1/Cross_View_ISLR/Code/XV-SLR-main/data/MM_WLAuslan/test/TED/kf/2d_skeleton"
    # o_save_path = "/media/xins/xinS1/Dataset/WLASL/WLASL-100-MTV/wlasl_test_dwpose/s1"
    #
    # cnt = 0
    # for name in os.listdir(ppp):
    #     save_name = name.split(".")[0]
    #     save_path = os.path.join(o_save_path, save_name + ".npy")
    #
    #     if os.path.exists(save_path) == True:
    #         print("Finish", save_path)
    #     else:
    #         cnt = cnt + 1
    #         video_name = os.path.join(ppp, name)
    #         # get_video_info(video_name)
    #         progress_video_dwpose(video_name, o_save_path)
    # print(cnt)

    # print("Hello, World!")
    # for i in [1,2,3,4]:
    #     ppp = "/media/xins/xinS1/Dataset/WLASL/WLASL-100-MTV/wlasl_test/s" + str(i)
    #     # ppp = "/media/xins/xinS1/Cross_View_ISLR/TESTSET/Processed_Test"
    #     # get_video_info("/media/xins/xinS1/Cross_View_ISLR/TESTSET/Processed_Test/web7_1734.mp4")
    #     # progress_video_dwpose("/media/xins/xinS1/Cross_View_ISLR/TESTSET/Processed_Test/web7_1734.mp4")
    #     # print(kkk)
    #     # o_save_path = "/media/xins/xinS1/Cross_View_ISLR/Code/XV-SLR-main/data/MM_WLAuslan/test/TED/kf/2d_skeleton"
    #     o_save_path = "/media/xins/xinS1/Dataset/WLASL/WLASL-100-MTV/wlasl_test_dwpose/s" + str(i)
    #
    #     cnt = 0
    #     for name in os.listdir(ppp):
    #         save_name = name.split(".")[0]
    #         save_path = os.path.join(o_save_path, save_name + ".npy")
    #
    #         if os.path.exists(save_path) == True:
    #             print("Finish", save_path)
    #         else:
    #             cnt = cnt + 1
    #             video_name = os.path.join(ppp, name)
    #             # get_video_info(video_name)
    #             progress_video_dwpose(video_name, o_save_path)
    #     print(cnt)

# ===============================================================================================================================
# Infer a folder
    # print("Hello, World!")
    # # ppp = "/media/xins/xinS1/Dataset/WLASL/WLASL100/"
    # ppp = "/media/xins/xinS1/Download/"
    # # get_video_info("/media/xins/xinS1/Cross_View_ISLR/TESTSET/Processed_Test/web7_1734.mp4")
    # # progress_video_dwpose("/media/xins/xinS1/Cross_View_ISLR/TESTSET/Processed_Test/web7_1734.mp4")
    # # print(kkk)
    # # o_save_path = "/media/xins/xinS1/Cross_View_ISLR/Code/XV-SLR-main/data/MM_WLAuslan/test/TED/kf/2d_skeleton"
    # o_save_path = "/media/xins/xinS1/Download/"
    #
    # cnt = 0
    # for name in os.listdir(ppp):
    #     save_name = name.split(".")[0]
    #     save_path = os.path.join(o_save_path, save_name + ".npy")
    #
    #     if os.path.exists(save_path) == True:
    #         print("Finish", save_path)
    #     else:
    #         cnt = cnt + 1
    #         video_name = os.path.join(ppp, name)
    #         # get_video_info(video_name)
    #         progress_video_dwpose(video_name, o_save_path)
    # print(cnt)

# Infer one sample
    ppp = "BOBSL/original_data/videos/mp4/5085344787448740525.mp4"
    o_save_path = "./tests"
    progress_video_dwpose(ppp, o_save_path)