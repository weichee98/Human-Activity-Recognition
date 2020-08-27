import numpy as np
import math

from utils.file_path import FilePath


class Utilities:

    @staticmethod
    def pointToVector(p1, p2, ignore_zero_vector=False):
        """
        Convert 2 points into a vector
        @param ignore_zero_vector: If either point 1 or point 2 is zero vector, return np.nan
        @param p1: Point 1
        @param p2: Point 2
        @return: Vector
        """
        if ignore_zero_vector:
            if np.count_nonzero(p1) == 0 or np.count_nonzero(p2) == 0:
                return np.nan
        return np.subtract(p1, p2)

    @staticmethod
    def distance(p1, p2, ignore_zero_vector=False):
        """
        Calculate distance between 2 points
        @param ignore_zero_vector: If either point 1 or point 2 is zero vector, return np.nan
        @param p1: Point 1
        @param p2: Point 2
        @return: Distance
        """
        if ignore_zero_vector:
            if np.count_nonzero(p1) == 0 or np.count_nonzero(p2) == 0:
                return np.nan
        return np.linalg.norm(Utilities.pointToVector(p1, p2))

    @staticmethod
    def unitVector(vector):
        """
        Calculate unit vector of a vector
        @param vector: Original vector
        @return: Unit vector
        """
        return vector / np.linalg.norm(vector)

    @staticmethod
    def angleBetween(v1, v2, radians=True, reflex=False):
        """
        Calculate angle between 2 vectors
        @param v1: Vector 1
        @param v2: Vector 2
        @param radians: Return angle in radians
        @param reflex: Return reflex angle
        @return: Angle
        """
        if np.isnan(v1).any() or np.isnan(v2).any():
            return np.nan
        else:
            uv1 = Utilities.unitVector(v1)
            uv2 = Utilities.unitVector(v2)
            dot_product = np.dot(uv1, uv2)
            angle = abs(np.arccos(dot_product))

            if reflex is False and angle > math.pi:
                angle = 2 * math.pi - angle

            if radians is True:
                return angle
            else:
                angle = angle / math.pi * 180
                return angle


if __name__ == "__main__":
    path = "image/processed/image.jpg"
    fp = FilePath(path)
    print(fp.getDirectory())
    print(fp.getFile())
    print(fp.getFileName())
    print(fp.getExtension())
