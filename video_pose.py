import os
from datetime import datetime

import cv2
import numpy as np

from utils.file_path import FilePath
from utils.image_processing import ImageProcessing
from utils.keypoint_tracker import KeypointTracker
from utils.log import Log
from utils.pose_classifier import PoseClassifier
from utils.pose_estimator import PoseEstimator


class VideoPose:

    def __init__(self):
        self.__pose_estimator = PoseEstimator(face=False, hand=False)
        self.__keypoint_tracker = None

    def analyze(self,
                video_path,
                fps=None,
                max_frame=np.inf,
                start_frame=1,
                show_skeleton=True,
                show_num_of_people=True,
                classify_pose=False,
                show_pose=False,
                track_pose_id=False,
                frames_to_disappear=None,
                show_pose_id=False,
                display_image=True,
                wait_key=1,
                log=False,
                export_frame=False):

        if start_frame < 1:
            raise Exception("argument 'start_frame' in VideoPose.analyze() cannot be less than 1")

        fp = FilePath(video_path)
        dir_name = fp.getDirectory() + "\\" + fp.getFileName() + "\\"
        time_identifier = datetime.now().strftime("%Y%m%d-%H%M%S-")

        if export_frame is True or log is True:
            try:
                os.mkdir(dir_name)
                print("Directory", dir_name, "Created ")
            except FileExistsError:
                print("Directory", dir_name, "already exists")

        if log is True:
            log_file = os.path.abspath(dir_name + time_identifier + fp.getFileName() + ".log")
            logger = Log.setup_logger(fp.getFileName(), log_file)
            logger.info('Processing Video: ' + str(video_path) + "\n")
        else:
            logger = None

        try:
            cap = cv2.VideoCapture(video_path)
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if fps is None:
                fps = original_fps
            if track_pose_id is True:
                if frames_to_disappear is None:
                    self.__keypoint_tracker = KeypointTracker(
                        frame_width=frame_width,
                        frame_height=frame_height,
                        logger=logger
                    )
                else:
                    self.__keypoint_tracker = KeypointTracker(
                        frame_width=frame_width,
                        frame_height=frame_height,
                        max_disappeared=frames_to_disappear,
                        logger=logger
                    )

            num_frame = start_frame
            for i in range(start_frame):
                cap.read()
                for j in range(int(original_fps / fps) - 1):
                    cap.read()

            while cap.isOpened() and (max_frame is None or num_frame <= max_frame):
                ret, frame = cap.read()
                for i in range(int(original_fps / fps) - 1):
                    ret, frame = cap.read()

                if ret is False or frame is None:
                    break

                print("Processing Frame:", num_frame)
                if log is True:
                    logger.info('Processing Frame: ' + str(num_frame))

                self.__pose_estimator.processImage(frame)
                if show_skeleton is True:
                    outputImage = self.__pose_estimator.getOutputImage()
                else:
                    outputImage = frame
                keypoints = self.__pose_estimator.getPoseKeypoints()
                if show_num_of_people is True:
                    ImageProcessing.outputNumberOfPeopleToImage(outputImage, len(keypoints))

                if classify_pose is True:
                    index = 1
                    for keypoint in keypoints:
                        print('Passenger', index)
                        if log is True:
                            logger.info('Passenger' + str(index))
                            logger.info('\n' + str(keypoint))
                        pose = PoseClassifier.predictPoseBody25(keypoint, logger)
                        text = str(index) + ': ' + pose.name
                        print(text + "\n")
                        if log is True:
                            logger.info('Passenger ' + text + '\n')
                        if show_pose is True:
                            ImageProcessing.outputIndividualPoseToImage(outputImage, text, keypoint)
                        index += 1

                if track_pose_id is True:
                    self.__keypoint_tracker.update(keypoints)
                    keypoints = self.__keypoint_tracker.getKeypointsInFrame()
                    if show_pose_id is True:
                        for (pose_id, keypoint) in zip(keypoints.keys(), keypoints.values()):
                            ImageProcessing.outputIndividualIdToImage(outputImage, pose_id, keypoint)

                if export_frame is True:
                    output_path = dir_name + time_identifier + 'Frame ' + str(num_frame) + ".jpg"
                    cv2.imwrite(output_path, outputImage)
                    print(output_path)
                    print('Successfully saved' + '\n')
                    if log is True:
                        logger.info(output_path)
                        logger.info('Successfully saved' + '\n')

                if display_image is True:
                    try:
                        cv2.imshow(video_path, outputImage)
                        cv2.waitKey(wait_key)
                    except Exception as e:
                        if log is True:
                            logger.error(e)
                        print(e)

                num_frame += 1

            print("Done")
            if log is True:
                logger.info("Done")

        except Exception as e:
            if log is True:
                logger.error(e)
            print(e)
            raise e


if __name__ == "__main__":

    video_path = "videos/00001.mp4"

    VideoPose().analyze(
        video_path,
        fps=4,
        max_frame=200,
        start_frame=1,
        show_skeleton=False,
        show_num_of_people=True,
        classify_pose=False,
        show_pose=False,
        track_pose_id=True,
        frames_to_disappear=3,
        show_pose_id=True,
        display_image=True,
        wait_key=1,
        log=True,
        export_frame=True
    )




