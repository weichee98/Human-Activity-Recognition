import cv2
import numpy as np

from utils.openpose import OpenPose
from collections import OrderedDict

op = OpenPose.op


class PoseEstimator:

    def __init__(self, face=False, hand=False):
        self.__params = dict()
        self.__params["model_folder"] = OpenPose.model_path
        self.__params["face"] = face
        self.__params["hand"] = hand

        self.__opWrapper = self.__createWrapper(self.__params)
        self.__opWrapper.start()

        self.__datum = op.Datum()

    def __createWrapper(self, params):
        opWrapper = op.WrapperPython()
        opWrapper.configure(self.__params)
        return opWrapper

    def setFace(self, face):
        self.__params["face"] = face
        self.__opWrapper = self.__createWrapper(self.__params)
        self.__opWrapper.start()

    def setHand(self, hand):
        self.__params["hand"] = hand
        self.__opWrapper = self.__createWrapper(self.__params)
        self.__opWrapper.start()

    def setParams(self, face, hand):
        self.__params["face"] = face
        self.__params["hand"] = hand
        self.__opWrapper = self.__createWrapper(self.__params)
        self.__opWrapper.start()

    def processImage(self, image):
        self.__datum.cvInputData = image
        self.__opWrapper.emplaceAndPop([self.__datum])

    def getOutputImage(self):
        return self.__datum.cvOutputData

    def getPoseKeypoints(self):
        try:
            len(self.__datum.poseKeypoints)
            return self.__datum.poseKeypoints
        except TypeError:
            return np.array([])

    def getFaceKeypoints(self):
        try:
            len(self.__datum.faceKeypoints)
            return self.__datum.faceKeypoints
        except TypeError:
            return np.array([])

    def getLeftHandKeypoints(self):
        try:
            len(self.__datum.handKeypoints[0])
            return self.__datum.handKeypoints[0]
        except TypeError:
            return np.array([])

    def getRightHandKeypoints(self):
        try:
            len(self.__datum.handKeypoints[1])
            return self.__datum.handKeypoints[1]
        except TypeError:
            return np.array([])


if __name__ == "__main__":
    image_path = "../images/COCO/COCO_val2014_000000000192.jpg"
    imageToProcess = cv2.imread(image_path)
    pose_estimator = PoseEstimator(face=False, hand=False)
    pose_estimator.processImage(imageToProcess)
    outputImage = pose_estimator.getOutputImage()
    keypoints = pose_estimator.getPoseKeypoints()
    print(keypoints)

    text = "Number of People: " + str(len(keypoints))
    (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)[0]
    org = (30, 60)
    box_coordinates = ((org[0], org[1] + 5), (30 + text_width + 2, 60 - text_height - 10))
    cv2.rectangle(outputImage, box_coordinates[0], box_coordinates[1], (255, 255, 255), cv2.FILLED)
    cv2.putText(outputImage, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
    cv2.imshow(image_path, outputImage)
    cv2.waitKey(0)
