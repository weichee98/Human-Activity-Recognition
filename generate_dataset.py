import os
import cv2
import pandas as pd
import numpy as np
import json
from datetime import datetime

from keras_preprocessing.image import ImageDataGenerator, load_img

from utils.body25 import Body25
from utils.file_path import FilePath
from utils.log import Log
from utils.pose_estimator import PoseEstimator


columns = ['KEYPOINTS', 'NORMALIZED_KEYPOINTS', 'POSE', 'FILENAME', 'SCORE']


def imageAugmentation(directory, export_directory=None, prefix="aug", extension="jpg", logger=None):
    aug = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    dir_path = os.path.abspath(directory)
    if export_directory is None:
        export_directory = dir_path + "\\augmented"
        try:
            os.mkdir(export_directory)
            print("Directory", export_directory, "Created ")
        except FileExistsError:
            print("Directory", export_directory, "already exists")
    extension = extension.lower()

    try:
        file_names = os.listdir(dir_path)
        counter = 1
        for file_name in file_names:
            if not file_name.lower().endswith(extension):
                continue
            fp = FilePath(dir_path + "\\" + file_name)
            try:
                image = np.expand_dims(load_img(fp.getAbsPath()), axis=0)
            except Exception:
                continue
            print("Processing Image " + str(counter) + ": " + file_name)
            if logger is not None:
                logger.info("Processing Image " + str(counter) + ": " + file_name)
            aug.fit(image)
            for x, val in zip(
                    aug.flow(image, save_to_dir=export_directory, save_format=extension, save_prefix=prefix),
                    range(10)
                    ):
                pass
                # if logger is not None:
                #     logger.info("...... saving augmented image " + str(val + 1) + " ......")
            counter += 1

        print("Done")
        if logger is not None:
            logger.info("Done")

    except Exception as e:
        if logger is not None:
            logger.info(e)
        print(e)


def generateDatasetFromDirectory(directory,
                                 pose,
                                 export_path,
                                 start=0,
                                 end=None,
                                 logger=None,
                                 df=None,
                                 config=False
                                 ):

    path = export_path.split('.')
    extension = path[-1]
    export_path = '.'.join(path[:-1])

    if df is None:
        try:
            df = pd.read_pickle(export_path + ".pkl")
        except Exception as e:
            df = pd.DataFrame([], columns=columns)
    else:
        df.columns = columns
    dir_path = os.path.abspath(directory)

    config_path = os.path.join(dir_path, "generate_dataset_config.json")

    try:

        file_names = sorted(os.listdir(dir_path))

        if config is True:
            config_dict = dict()
            try:
                with open(config_path, 'r') as openfile:
                    config_dict = json.load(openfile)
                    openfile.close()
            except Exception as e:
                pass
            for file_name in file_names:
                if file_name not in config_dict:
                    config_dict[file_name] = False

        if end is None:
            end = len(file_names) - 1

        pose_estimator = PoseEstimator(face=False, hand=False)

        for i, file_name in enumerate(file_names, start=1):

            fp = FilePath(dir_path + "\\" + file_name)
            try:
                if int(fp.getFileName()) < start or int(fp.getFileName()) > end:
                    print(file_name + " is not in range\n")
                    continue
            except Exception as e:
                continue

            if config is True:
                if config_dict[file_name] is True:
                    print(file_name + " has already been processed\n")
                    continue

            imageToProcess = cv2.imread(fp.getAbsPath())
            if imageToProcess is None:
                continue

            print("Processing " + file_name)
            if logger is not None:
                logger.info("Processing " + file_name)
            pose_estimator.processImage(imageToProcess)
            output = pose_estimator.getPoseKeypoints()

            person = 1
            for keypoint in output:
                score = Body25.getAverageScore(keypoint)
                if score < 0.5:
                    body_pose = "UNKNOWN"
                else:
                    body_pose = pose
                df = df.append(
                    {
                        columns[0]: keypoint,
                        columns[1]: Body25.normalizeKeypoint(keypoint),
                        columns[2]: body_pose,
                        columns[3]: file_name,
                        columns[4]: score
                    },
                    ignore_index=True
                )
                print("Person " + str(person) + ": " + body_pose)
                if logger is not None:
                    logger.info("Person " + str(person) + ": " + body_pose)
                person += 1

            if config is True:
                config_dict[file_name] = True

            print(file_name + " Done")
            print(str(df['POSE'].value_counts()) + "\n")
            if logger is not None:
                logger.info(file_name + " Done")
                logger.info(str(df['POSE'].value_counts()) + "\n")

            if i % 5 == 0:
                df.to_pickle(export_path + ".pkl", protocol=4)
                print(export_path + ".pkl Saved")
                if logger is not None:
                    logger.info(export_path + ".pkl Saved")
                if config is True:
                    with open(config_path, "w") as outfile:
                        json.dump(config_dict, outfile, sort_keys=True)
                        outfile.close()

    except Exception as e:
        if logger is not None:
            logger.error(e)
        print(e)

    df.to_pickle(export_path + ".pkl", protocol=4)
    df.to_csv(export_path + "." + extension, index=False, header=True)
    if config is True:
        with open(config_path, "w") as outfile:
            json.dump(config_dict, outfile, sort_keys=True)
            outfile.close()
    print(export_path + ".pkl Saved")
    print(export_path + "." + extension + " Saved" + "\n")
    print("Done")
    print(df['POSE'].value_counts())
    if logger is not None:
        logger.info(export_path + ".pkl Saved")
        logger.info(export_path + "." + extension + " Saved" + "\n")
        logger.info("Done")
        logger.info(df['POSE'].value_counts())


def combineDataset(export_path, data_frames, shuffle=False, shuffle_time=1):
    path = export_path.split('.')
    extension = path[-1]
    export_path = '.'.join(path[:-1])

    df = pd.DataFrame([], columns=columns)
    for i, data_frame in enumerate(data_frames):
        try:
            data_frame.columns = columns
            df = df.append(data_frame, ignore_index=True)
            print("Appended", i + 1)
        except Exception as e:
            print(e)
            continue

    if shuffle is True:
        for i in range(int(shuffle_time)):
            df = df.sample(frac=1).reset_index(drop=True)
            print("Shuffled", i + 1)

    print("Exporting to pickle:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    df.to_pickle(export_path + ".pkl", protocol=4)
    print("Exported:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print("Exporting to csv:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    df.to_csv(export_path + "." + extension, index=False, header=True)
    print("Exported:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    time_identifier = datetime.now().strftime("%Y%m%d-%H%M%S-")

    """ Sitting Dataset """
    # log_file = "E:/Human Activity Recognition/images/Sitting/" + "sitting-dataset.log"
    # csv_file = "E:/Human Activity Recognition/images/Sitting/" + "sitting-dataset.csv"
    # logger = Log.setup_logger("generate_dataset", log_file, mode='a')
    # generateDatasetFromDirectory(
    #     directory="E:/Human Activity Recognition/images/Sitting/augmented",
    #     pose="SITTING",
    #     export_path=csv_file,
    #     end=999,
    #     logger=logger,
    #     df=None,
    #     config=True
    # )

    """ Standing Dataset """
    # log_file = "E:/Human Activity Recognition/images/Standing/" + "standing-dataset.log"
    # csv_file = "E:/Human Activity Recognition/images/Standing/" + "standing-dataset.csv"
    # logger = Log.setup_logger("generate_dataset", log_file, mode='a')
    # generateDatasetFromDirectory(
    #     directory="E:/Human Activity Recognition/images/Standing/augmented",
    #     pose="STANDING",
    #     export_path=csv_file,
    #     end=999,
    #     logger=logger,
    #     df=None,
    #     config=True
    # )

    path = "dataset/" + time_identifier + "dataset.csv"
    files = [
        "E:/Human Activity Recognition/images/Standing/standing-dataset.pkl",
        "E:/Human Activity Recognition/images/Sitting/sitting-dataset.pkl",
    ]
    dfs = [pd.read_pickle(file) for file in files]
    combineDataset(path, dfs, shuffle=True, shuffle_time=10)

    # df = pd.read_pickle("E:/Human Activity Recognition/dataset/20200525-191257-dataset.pkl")
    # print(df['POSE'].value_counts())

    # dir = "E:/Human Activity Recognition/images/Standing/New"
    # log_file = dir + "/" + time_identifier + "augment.log"
    # logger = Log.setup_logger("augment_images", log_file)
    # imageAugmentation(dir, export_directory=None, prefix="aug", extension="jpg", logger=logger)

    # config_file = open("E:/Human Activity Recognition/images/Sitting/augmented/generate_dataset_config.json", "w+")
    # config_dict = json.load(config_file)
    # print(config_dict)


