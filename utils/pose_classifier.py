import os

import numpy as np
import math
from enum import Enum

from scipy.stats import stats
from tensorflow import keras

from utils.utilities import Utilities as Ut
from utils.body25 import Body25

dir_path = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.abspath(dir_path + "/../model")

class PoseClassifier:

    class Pose(Enum):
        UNKNOWN = 0
        SITTING = 1
        STANDING = 2

    class Model:

        def __init__(self):
            self.__models = self.__loadModels()

        def __loadModels(self):
            models = []
            model_paths = os.listdir(model_dir)
            for model_path in model_paths:
                model = keras.models.load_model(model_dir + "\\" + model_path)
                model.trainable = False
                models.append(model)
            return models

        def getModels(self):
            return self.__models

    __model = Model()

    @staticmethod
    def predictPoseModel(keypoint, logger=None):
        keypoint = Body25.normalizeKeypoint(keypoint)
        poses = {pose: 0 for pose in PoseClassifier.Pose}
        for model in PoseClassifier.__model.getModels():
            pose_prediction = model.predict(np.array([keypoint]))
            for i in range(len(pose_prediction[0])):
                poses[PoseClassifier.Pose(i)] += pose_prediction[0][i]
        print(poses)
        if logger is not None:
            logger.info(str(poses))
        return max(poses, key=poses.get)

    class Scores:
        def __init__(self, logger=None):
            self.__logger = logger
            self.RBody_hip_score = np.nan
            self.RBody_hip_score = np.nan
            self.RKnee_hip_score = np.nan
            self.RKnee_score = np.nan
            self.LBody_hip_score = np.nan
            self.LKnee_hip_score = np.nan
            self.LKnee_score = np.nan
            self.left_score = np.nan
            self.right_score = np.nan

        def print(self):
            print('{\n\tScores: {')
            print('\t\tRBody_hip_score: ', self.RBody_hip_score)
            print('\t\tRKnee_hip_score: ', self.RKnee_hip_score)
            print('\t\tRKnee_score: ', self.RKnee_score)
            print('\t\tLBody_hip_score: ', self.LBody_hip_score)
            print('\t\tLKnee_hip_score: ', self.LKnee_hip_score)
            print('\t\tLKnee_score: ', self.LKnee_score)
            print('\t}')
            print('\tRight_score: ', self.right_score)
            print('\tLeft_score: ', self.left_score)
            print('}')

        def log(self):
            self.__logger.info('Scores:')
            self.__logger.info('=======')
            self.__logger.info('RBody_hip_score: ' + str(self.RBody_hip_score))
            self.__logger.info('RKnee_hip_score: ' + str(self.RKnee_hip_score))
            self.__logger.info('RKnee_score: ' + str(self.RKnee_score))
            self.__logger.info('LBody_hip_score: ' + str(self.LBody_hip_score))
            self.__logger.info('LKnee_hip_score: ' + str(self.LKnee_hip_score))
            self.__logger.info('LKnee_score: ' + str(self.LKnee_score))
            self.__logger.info('Right_score: ' + str(self.right_score))
            self.__logger.info('Left_score: ' + str(self.left_score))

    @staticmethod
    def __yDifference(p1, p2):
        """
        Calculate the difference of y-coordinates between 2 points
        @param p1: Point 1
        @param p2: Point 2
        @return: Difference of y-coordinates
        """
        if np.count_nonzero(p1) == 0 or np.count_nonzero(p2) == 0:
            return np.nan
        return abs(p1[1] - p2[1])

    @staticmethod
    def __angleScore(angle):
        """
        Calculate the score of sitting from the angle obtained
        @param angle: Angle in radians
        @return: Score of sitting
        """
        if angle < 2.416:
            return math.sin(0.5 * (angle + math.pi / 2))
        else:
            return math.sin(angle) - 0.25 * math.sin(2 * angle)

    @staticmethod
    def __distanceScore(y_dist, dist):
        return 1 - (y_dist / dist)

    @staticmethod
    def __finalScore(scores, weights):
        if len(scores) != len(weights):
            raise Exception('length of scores and weights are different')
        else:
            sum_score = 0
            sum_weight = 0
            for (score, weight) in zip(scores, weights):
                if np.isnan(score):
                    continue
                sum_score += score * weight
                sum_weight += weight
            if sum_score == 0 and sum_weight == 0:
                return np.nan
            else:
                return sum_score / sum_weight

    @staticmethod
    def __determinePose(left_score, right_score, threshold=0.5):
        if np.isnan(right_score) and np.isnan(left_score):
            return PoseClassifier.Pose.UNKNOWN
        elif np.isnan(right_score) and left_score >= threshold:
            return PoseClassifier.Pose.SITTING
        elif np.isnan(left_score) and right_score >= threshold:
            return PoseClassifier.Pose.SITTING
        else:
            probability_sitting = left_score * right_score
            probability_standing = (1 - left_score) * (1 - right_score)
            if probability_sitting >= threshold ** 2:
                return PoseClassifier.Pose.SITTING
            elif probability_standing >= threshold ** 2:
                return PoseClassifier.Pose.STANDING
            else:
                if probability_sitting > probability_standing and \
                        (left_score > threshold or right_score > threshold):
                    return PoseClassifier.Pose.SITTING
                elif probability_sitting > probability_standing and \
                        ((1 - left_score) > threshold or (1 - right_score) > threshold):
                    return PoseClassifier.Pose.STANDING
                return PoseClassifier.Pose.UNKNOWN

    @staticmethod
    def predictPoseBody25(keypoint, logger=None):
        """
        @param log_file: The path of log file
        @param keypoint: 3D array, 25 keypoints x 3 (x, y, score)
        @return:
        """
        keypoint = np.array(keypoint)
        Neck = Body25.getCoordinates(keypoint, Body25.Parts.NECK)
        MidHip = Body25.getCoordinates(keypoint, Body25.Parts.MID_HIP)
        RHip = Body25.getCoordinates(keypoint, Body25.Parts.R_HIP)
        RKnee = Body25.getCoordinates(keypoint, Body25.Parts.R_KNEE)
        RAnkle = Body25.getCoordinates(keypoint, Body25.Parts.R_ANKLE)
        LHip = Body25.getCoordinates(keypoint, Body25.Parts.L_HIP)
        LKnee = Body25.getCoordinates(keypoint, Body25.Parts.L_KNEE)
        LAnkle = Body25.getCoordinates(keypoint, Body25.Parts.L_ANKLE)

        # define vectors of body parts
        MidHip_Neck = Ut.pointToVector(MidHip, Neck, ignore_zero_vector=True)
        MidHip_RKnee = Ut.pointToVector(MidHip, RKnee, ignore_zero_vector=True)
        MidHip_LKnee = Ut.pointToVector(MidHip, LKnee, ignore_zero_vector=True)
        RKnee_RHip = Ut.pointToVector(RKnee, RHip, ignore_zero_vector=True)
        RKnee_RAnkle = Ut.pointToVector(RKnee, RAnkle, ignore_zero_vector=True)
        LKnee_LHip = Ut.pointToVector(LKnee, LHip, ignore_zero_vector=True)
        LKnee_LAnkle = Ut.pointToVector(LKnee, LAnkle, ignore_zero_vector=True)

        # find angle in radians between body parts
        RBody_hip_angle = Ut.angleBetween(MidHip_Neck, -RKnee_RHip)
        RKnee_hip_angle = Ut.angleBetween(MidHip_Neck, MidHip_RKnee)
        RKnee_angle = Ut.angleBetween(RKnee_RHip, RKnee_RAnkle)
        LBody_hip_angle = Ut.angleBetween(MidHip_Neck, -LKnee_LHip)
        LKnee_hip_angle = Ut.angleBetween(MidHip_Neck, MidHip_LKnee)
        LKnee_angle = Ut.angleBetween(LKnee_LHip, LKnee_LAnkle)

        # find the score of sitting
        scores = PoseClassifier.Scores(logger)
        scores.RBody_hip_score = PoseClassifier.__angleScore(RBody_hip_angle)
        scores.RKnee_hip_score = PoseClassifier.__angleScore(RKnee_hip_angle)
        scores.RKnee_score = PoseClassifier.__angleScore(RKnee_angle)
        scores.LBody_hip_score = PoseClassifier.__angleScore(LBody_hip_angle)
        scores.LKnee_hip_score = PoseClassifier.__angleScore(LKnee_hip_angle)
        scores.LKnee_score = PoseClassifier.__angleScore(LKnee_angle)

        # find final score
        scores.left_score = PoseClassifier.__finalScore(
            [scores.LBody_hip_score, scores.LKnee_score, scores.LKnee_hip_score], (1, 1, 2))
        scores.right_score = PoseClassifier.__finalScore(
            [scores.RBody_hip_score, scores.RKnee_score, scores.RKnee_hip_score], (1, 1, 2))
        scores.print()
        if logger is not None:
            scores.log()

        # Comparing left and right score
        return PoseClassifier.__determinePose(scores.left_score, scores.right_score)


if __name__ == "__main__":
    zeroes = np.zeros((25, 3))

    kpsit = np.array([
        [4.42390076e+02, 1.42055908e+02, 9.31057751e-01],
        [4.73108246e+02, 1.76704758e+02, 8.55732441e-01],
        [4.18393311e+02, 1.74087585e+02, 7.22459793e-01],
        [4.21043182e+02, 2.58171295e+02, 7.65351832e-01],
        [4.53163239e+02, 1.87442978e+02, 8.13761950e-01],
        [5.23877197e+02, 1.78072769e+02, 7.46139526e-01],
        [5.51937683e+02, 2.67498383e+02, 8.12307060e-01],
        [4.95846954e+02, 2.92851898e+02, 8.49543452e-01],
        [4.78472076e+02, 2.84907715e+02, 6.46864474e-01],
        [4.49085541e+02, 2.84876923e+02, 6.36026323e-01],
        [4.11716949e+02, 2.83555817e+02, 8.71465325e-01],
        [3.78382233e+02, 4.29104858e+02, 7.43569255e-01],
        [5.09182556e+02, 2.86229889e+02, 5.98113239e-01],
        [5.69236389e+02, 3.15548248e+02, 8.75916243e-01],
        [5.38580383e+02, 4.59809021e+02, 4.78815258e-01],
        [4.41069611e+02, 1.29969437e+02, 8.78573596e-01],
        [4.57085266e+02, 1.35340851e+02, 9.38320279e-01],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        [4.89181396e+02, 1.43406219e+02, 8.67179275e-01],
        [5.30567322e+02, 4.77151215e+02, 1.23407543e-01],
        [5.43909973e+02, 4.87838837e+02, 1.73968494e-01],
        [5.29224792e+02, 4.71822449e+02, 2.13540137e-01],
        [3.54326416e+02, 4.62431793e+02, 6.26299977e-01],
        [3.50283020e+02, 4.55789062e+02, 6.02773786e-01],
        [3.86339966e+02, 4.39743530e+02, 5.85921109e-01],
    ])

    kpstand = np.array([
        [4.11885315e+02, 3.90554077e+02, 9.08873439e-01],
        [4.82883087e+02, 3.97675751e+02, 6.94850504e-01],
        [5.14727417e+02, 3.90667145e+02, 6.93315387e-01],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        [4.68591736e+02, 4.04796509e+02, 9.05950427e-01],
        [4.50879333e+02, 5.21866943e+02, 6.65256977e-01],
        [3.90712646e+02, 4.82909851e+02, 7.84260571e-01],
        [4.89994324e+02, 6.24703918e+02, 6.81913018e-01],
        [5.11208069e+02, 6.14037720e+02, 5.85318208e-01],
        [5.11277130e+02, 7.70067017e+02, 6.71832383e-01],
        [5.25365479e+02, 9.40246643e+02, 7.92349756e-01],
        [4.72255646e+02, 6.28303223e+02, 6.84913397e-01],
        [4.68632324e+02, 7.98429077e+02, 6.73719764e-01],
        [5.04047089e+02, 9.93633728e+02, 8.46730828e-01],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        [4.08392944e+02, 3.76354156e+02, 9.59921718e-01],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
        [4.36781921e+02, 3.48028015e+02, 9.65771973e-01],
        [4.33117950e+02, 1.01837164e+03, 8.61553252e-01],
        [4.47467743e+02, 1.02898376e+03, 8.76815796e-01],
        [5.21970520e+02, 1.00415997e+03, 8.63886356e-01],
        [4.68623779e+02, 9.43982849e+02, 4.17909473e-01],
        [4.68790588e+02, 9.40424622e+02, 4.09848332e-01],
        [5.39665955e+02, 9.50938538e+02, 7.58430064e-01],
    ])

    print(PoseClassifier.predictPoseBody25(kpsit).name)
    print(PoseClassifier.predictPoseModel(kpsit).name)
