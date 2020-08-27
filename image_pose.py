import os
from datetime import datetime

import cv2

from utils.file_path import FilePath
from utils.image_processing import ImageProcessing
from utils.log import Log
from utils.pose_classifier import PoseClassifier
from utils.pose_estimator import PoseEstimator


class ImagePose:

    def __init__(self):
        self.__pose_estimator = PoseEstimator(face=False, hand=False)

    def analyze(self,
                image,
                image_id=None,
                show_skeleton=True,
                show_num_of_people=True,
                classify_pose=False,
                show_pose=False,
                display_image=True,
                wait_key=0,
                logger=None,
                export_path=None):

        directory, file = os.path.split(os.path.abspath(export_path))

        if export_path is not None:
            try:
                os.mkdir(directory)
                print("Directory", directory, " Created ")
            except FileExistsError:
                print("Directory", directory, " already exists")

        if logger is not None and image_id is not None:
            logger.info('Processing Image: ' + str(image_id))

        try:
            self.__pose_estimator.processImage(image)
            if show_skeleton is True:
                outputImage = self.__pose_estimator.getOutputImage()
            else:
                outputImage = image
            keypoints = self.__pose_estimator.getPoseKeypoints()
            if show_num_of_people is True:
                ImageProcessing.outputNumberOfPeopleToImage(outputImage, len(keypoints))

            if classify_pose is True:
                index = 1
                for keypoint in keypoints:
                    print('Passenger', index)
                    if logger is not None:
                        logger.info('Passenger' + str(index))
                        logger.info('\n' + str(keypoint))
                    # pose = PoseClassifier.predictPoseBody25(keypoint, logger)
                    pose = PoseClassifier.predictPoseModel(keypoint, logger)
                    text = str(index) + ': ' + pose.name
                    print(text + "\n")
                    if logger is not None:
                        logger.info('Passenger ' + text + '\n')
                    if show_pose is True:
                        ImageProcessing.outputIndividualPoseToImage(outputImage, text, keypoint)
                    index += 1

            if export_path is not None:
                cv2.imwrite(export_path, outputImage)
                print(export_path)
                print('Successfully saved')
                if logger is not None:
                    logger.info(export_path)
                    logger.info('Successfully saved' + '\n')

            if display_image is True:
                cv2.destroyAllWindows()
                cv2.imshow(image_id, outputImage)
                cv2.waitKey(wait_key)

        except Exception as e:
            if logger is not None:
                logger.error(e)
            print(e)
            raise e

        return outputImage


if __name__ == "__main__":

    image_path = "images/Subway/00006.jpg"
    fp = FilePath(image_path)
    dir_name = fp.getDirectory() + "\\processed\\"
    time_identifier = datetime.now().strftime("%Y%m%d-%H%M%S-")

    output_path = dir_name + time_identifier + fp.getFileName() + fp.getExtension()
    log_file = os.path.abspath(dir_name + time_identifier + fp.getFileName() + ".log")
    logger = Log.setup_logger(fp.getFileName(), log_file)
    imageToProcess = cv2.imread(image_path)

    ImagePose().analyze(
        image=imageToProcess,
        image_id=None,
        show_skeleton=True,
        show_num_of_people=True,
        classify_pose=True,
        show_pose=True,
        display_image=True,
        wait_key=0,
        logger=logger,
        export_path=output_path
    )




