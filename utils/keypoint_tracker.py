from collections import OrderedDict
import numpy as np

from utils.body25 import Body25
from utils.utilities import Utilities as Ut


class KeypointTracker:

    def __init__(self, frame_width, frame_height, max_disappeared=3, logger=None):
        self.__next_ID = 0
        self.__frame_width = frame_width
        self.__frame_height = frame_height
        self.__logger = logger
        self.__keypoints = OrderedDict()
        self.__in_frame = OrderedDict()
        self.__disappeared = OrderedDict()
        self.__max_disappeared = max_disappeared

    def setMaxDisappeared(self, max_disappeared):
        self.__max_disappeared = max_disappeared

    def __register(self, keypoint):
        self.__keypoints[self.__next_ID] = keypoint
        self.__in_frame[self.__next_ID] = True
        self.__disappeared[self.__next_ID] = 0
        self.__next_ID += 1

    def __unregister(self, object_id):
        del self.__keypoints[object_id]
        del self.__in_frame[object_id]
        del self.__disappeared[object_id]

    def getKeypointsInFrame(self):
        keypoints_in_frame = OrderedDict()
        for (object_id, keypoint) in zip(self.__keypoints.keys(), self.__keypoints.values()):
            if self.__in_frame[object_id] is True:
                keypoints_in_frame[object_id] = keypoint
        return keypoints_in_frame

    def __calculateError(self, keypoint1, keypoint2):

        min_x1, min_y1, max_x1, max_y1 = Body25.getFrameCoordinates(keypoint1)
        min_x2, min_y2, max_x2, max_y2 = Body25.getFrameCoordinates(keypoint2)
        width1 = max_x1 - min_x1
        width2 = max_x2 - min_x2
        height1 = max_y1 - min_y1
        height2 = max_y2 - min_y2
        total_area1 = width1 * height1
        total_area2 = width2 * height2

        frame = np.full(shape=Body25.Keypoint.getCoordinatesShape(), fill_value=np.inf)
        frame[Body25.Keypoint.X.value] = self.__frame_width
        frame[Body25.Keypoint.Y.value] = self.__frame_height

        def distanceError():
            sum_dist = 0
            num = 0
            for part in Body25.Parts:
                coord1 = Body25.getCoordinates(keypoint1, part)
                coord2 = Body25.getCoordinates(keypoint2, part)
                dist = Ut.distance(coord1 / frame, coord2 / frame, ignore_zero_vector=True)
                if np.isnan(dist):
                    continue
                else:
                    sum_dist += dist
                num += 1
            if num == 0:
                return np.inf
            else:
                return sum_dist / num

        def centroidError():
            centroid1 = Body25.getCentroid(keypoint1)
            centroid2 = Body25.getCentroid(keypoint2)
            dist = Ut.distance(centroid1 / frame, centroid2 / frame, ignore_zero_vector=True)
            if np.isnan(dist):
                return np.inf
            return dist

        def areaError():
            return abs(total_area1 - total_area2) / max(total_area1, total_area2)

        def overlapError():
            width = min(max_x1, max_x2) - max(min_x1, min_x2)
            height = min(max_y1, max_y2) - max(min_y1, min_y2)
            overlapped_area = width * height
            if abs(width) > width1 or abs(width) > width2 or \
                    abs(height) > height1 or abs(height) > height2 or \
                    abs(overlapped_area) > total_area1 or abs(overlapped_area) > total_area2:
                return np.inf
            return 1 - overlapped_area / (total_area1 + total_area2)

        errors = [distanceError(), centroidError(), areaError(), overlapError()]
        weight = [15, 3, 1, 3]
        return np.dot(errors, weight) / sum(weight)

    def update(self, frame_keypoints):
        if len(frame_keypoints) == 0:
            objectIDs = self.__disappeared.keys()
            for objectID in objectIDs:
                self.__disappeared[objectID] += 1
                self.__in_frame[objectID] = False
                if self.__disappeared[objectID] >= self.__max_disappeared:
                    self.__unregister(objectID)
            return

        # if the object dictionary is empty, register all tracked objects
        if len(self.__keypoints) == 0:
            for keypoint in frame_keypoints:
                self.__register(keypoint)

        # if there are existing objects, try to match the input keypoints to existing keypoints
        else:
            objectID_list = list(self.__keypoints.keys())
            objectKeypoints_list = list(self.__keypoints.values())

            # comparing object keypoints and input keypoints
            errorMatrix = np.empty(shape=(len(objectKeypoints_list), len(frame_keypoints)), dtype=object)
            for (i, keypoint) in enumerate(objectKeypoints_list):
                for (j, inputKeypoint) in enumerate(frame_keypoints):
                    err = self.__calculateError(keypoint, inputKeypoint)
                    errorMatrix[i, j] = err

            row_index, col_index = np.unravel_index(np.argsort(errorMatrix, axis=None), errorMatrix.shape)

            used_row = set()
            used_col = set()
            existing_id = list()
            for (row, col) in zip(row_index, col_index):
                if row in used_row or col in used_col:
                    continue
                if np.isinf(errorMatrix[row, col]):
                    continue
                # otherwise take the object ID for the current row and update its new keypoints
                # row number represents index of the current object
                objectID = objectID_list[row]
                existing_id.append(objectID)
                # assign input keypoints which has shortest distance from to the existing object
                # col number represents index of the input keypoint
                self.__keypoints[objectID] = frame_keypoints[col]
                self.__disappeared[objectID] = 0
                self.__in_frame[objectID] = True
                # update used rows and cols
                used_row.add(row)
                used_col.add(col)

            unused_row = set(range(0, errorMatrix.shape[0])).difference(used_row)
            unused_col = set(range(0, errorMatrix.shape[1])).difference(used_col)
            disappeared_id = list()
            for row in unused_row:
                objectID = objectID_list[row]
                disappeared_id.append(objectID)
                self.__disappeared[objectID] += 1
                self.__in_frame[objectID] = False
                if self.__disappeared[objectID] >= self.__max_disappeared:
                    self.__unregister(objectID)
            for col in unused_col:
                self.__register(frame_keypoints[col])

            print('Error Matrix:\n' + str(errorMatrix))
            print('Object ID List: ' + str(objectID_list))
            print('Row Index: ' + str(row_index))
            print('Col Index: ' + str(col_index))
            print('Existing IDs: ' + str(sorted(existing_id)))
            print('Disappeared IDs: ' + str(sorted(disappeared_id)))
            print('All IDs: ' + str(sorted(list(self.__keypoints.keys()))))
            if self.__logger is not None:
                self.__logger.info('Error Matrix:\n' + str(errorMatrix))
                self.__logger.info('Object ID List: ' + str(objectID_list))
                self.__logger.info('Row Index: ' + str(row_index))
                self.__logger.info('Col Index: ' + str(col_index))
                self.__logger.info('Existing IDs: ' + str(sorted(existing_id)))
                self.__logger.info('Disappeared IDs: ' + str(sorted(disappeared_id)))
                self.__logger.info('All IDs: ' + str(sorted(list(self.__keypoints.keys()))))
