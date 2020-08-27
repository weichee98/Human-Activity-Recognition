import math
from enum import Enum
import numpy as np


class Body25:

    class Parts(Enum):
        NOSE = 0
        NECK = 1
        R_SHOULDER = 2
        R_ELBOW = 3
        R_WRIST = 4
        L_SHOULDER = 5
        L_ELBOW = 6
        L_WRIST = 7
        MID_HIP = 8
        R_HIP = 9
        R_KNEE = 10
        R_ANKLE = 11
        L_HIP = 12
        L_KNEE = 13
        L_ANKLE = 14
        R_EYE = 15
        L_EYE = 16
        R_EAR = 17
        L_EAR = 18
        L_BIG_TOE = 19
        L_SMALL_TOE = 20
        L_HEEL = 21
        R_BIG_TOE = 22
        R_SMALL_TOE = 23
        R_HEEL = 24

    class Keypoint(Enum):
        X = 0
        Y = 1
        SCORE = 2

        @staticmethod
        def getCoordinatesShape():
            return Body25.Keypoint.SCORE.value

        @staticmethod
        def getKeypointShape():
            return 25, 3

    @staticmethod
    def getCoordinates(keypoint, part):
        return keypoint[part.value][:Body25.Keypoint.SCORE.value]

    @staticmethod
    def getScore(keypoint, part):
        return keypoint[part.value][Body25.Keypoint.SCORE.value]

    @staticmethod
    def getCentroid(keypoint, to_int=False):
        sum_coord = np.zeros(shape=Body25.Keypoint.getCoordinatesShape())
        num = 0
        for part in Body25.Parts:
            coord = Body25.getCoordinates(keypoint, part)
            if np.count_nonzero(coord) == 0:
                continue
            if sum_coord is None:
                sum_coord = coord
            else:
                sum_coord += coord
            num += 1
        if num == 0:
            return sum_coord
        if to_int is True:
            return (sum_coord / num).astype(int)
        else:
            return sum_coord / num

    @staticmethod
    def getFrameCoordinates(keypoint, to_int=False):
        # identify rows that are all zeros
        delete_index = list()
        for part in Body25.Parts:
            coord = Body25.getCoordinates(keypoint, part)
            if np.count_nonzero(coord) == 0:
                delete_index.append(part.value)

        # delete score column
        keypoint = np.delete(keypoint, Body25.Keypoint.SCORE.value, axis=1)
        # delete rows that are all zeros
        keypoint = np.delete(keypoint, delete_index, axis=0)
        # get the minimum and maximum values of each remaining column
        min_coord = np.amin(keypoint, axis=0)
        max_coord = np.amax(keypoint, axis=0)

        # return min_x, min_y, max_x, max_y
        if to_int is True:
            return tuple(np.ndarray.flatten(np.array((min_coord, max_coord))).astype(int))
        else:
            return tuple(np.ndarray.flatten(np.array((min_coord, max_coord))))

    @staticmethod
    def normalizeKeypoint(keypoint):
        min_x, min_y, max_x, max_y = Body25.getFrameCoordinates(keypoint)
        kp = keypoint.copy()
        for part in Body25.Parts:
            if np.count_nonzero(kp[part.value]) == 0:
                continue
            kp[part.value, Body25.Keypoint.X.value] = \
                (kp[part.value, Body25.Keypoint.X.value] - min_x) / (max_x - min_x)
            kp[part.value, Body25.Keypoint.Y.value] = \
                (kp[part.value, Body25.Keypoint.Y.value] - min_y) / (max_y - min_y)
        return kp

    @staticmethod
    def getAverageScore(keypoint):
        num_nonzero = np.count_nonzero(keypoint)
        weight = [1, 1]
        mean_score = math.sin(np.mean(keypoint, axis=0)[Body25.Keypoint.SCORE.value] * 0.5 * math.pi)
        nonzero_score = num_nonzero / keypoint.size
        scores = [mean_score, nonzero_score]
        return np.dot(scores, weight) / sum(weight)


if __name__ == "__main__":
    kp = np.array([
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

    print(Body25.getFrameCoordinates(kp, to_int=False))
    print(Body25.getFrameCoordinates(kp, to_int=True))
    print(Body25.normalizeKeypoint(kpstand))
    print(kpstand)


