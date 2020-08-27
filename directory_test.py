import os
from datetime import datetime

import cv2

from image_pose import ImagePose
from utils.file_path import FilePath
from utils.log import Log


try:
    dir_path = os.path.abspath("images/COCO")
    file_names = os.listdir(dir_path)
    dir_name = dir_path + "\\processed\\"
    IP = ImagePose()

    for file_name in file_names:

        fp = FilePath(dir_path + "\\" + file_name)
        imageToProcess = cv2.imread(fp.getAbsPath())
        if imageToProcess is None:
            break

        time_identifier = datetime.now().strftime("%Y%m%d-%H%M%S-")
        log_file = os.path.abspath(dir_name + time_identifier + fp.getFileName() + ".log")
        logger = Log.setup_logger(fp.getFileName(), log_file)
        output_path = dir_name + time_identifier + fp.getFileName() + fp.getExtension()

        IP.analyze(
            image=imageToProcess,
            image_id=fp.getFileName(),
            show_skeleton=True,
            show_num_of_people=True,
            classify_pose=True,
            show_pose=True,
            display_image=True,
            wait_key=1,
            logger=logger,
            export_path=output_path
        )

    print("Done")

except Exception as e:
    print(e)
    raise e
