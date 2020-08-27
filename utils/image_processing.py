import cv2
from utils.body25 import Body25
from utils.pose_estimator import PoseEstimator


class ImageProcessing:

    @staticmethod
    def outputIndividualPoseToImage(image, text, keypoint):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 2
        (text_width, text_height) = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        position = Body25.getCentroid(keypoint, to_int=True)
        position[0] -= text_width // 2
        box = ((position[0] - 2, position[1] + 2), (position[0] + text_width + 2, position[1] - text_height - 2))
        cv2.rectangle(image, box[0], box[1], (0, 0, 0), cv2.FILLED)
        cv2.putText(image, text, tuple(position), font, font_scale, (255, 255, 0), font_thickness)

    @staticmethod
    def outputNumberOfPeopleToImage(image, num_of_people):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text = "Number of People: " + str(num_of_people)
        (text_width, text_height) = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        org = (15, 15 + text_height)
        box_coordinates = ((org[0] - 5, org[1] + 5), (org[0] + text_width + 5, org[1] - text_height - 5))
        cv2.rectangle(image, box_coordinates[0], box_coordinates[1], (255, 255, 255), cv2.FILLED)
        cv2.putText(image, text, org, font, font_scale, (0, 0, 255), font_thickness)

    @staticmethod
    def outputIndividualIdToImage(image, id, keypoint):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 2
        text = "ID: " + str(id)
        text_width, text_height = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        min_x, min_y, max_x, max_y = Body25.getFrameCoordinates(keypoint, to_int=True)
        box_coordinates = ((min_x - 5, min_y - text_height - 10), (max_x + 5, max_y + 5))
        org = (min_x, min_y - 5)
        # text_box_coordinates = ((org[0] - 2, org[1] + 2), (org[0] + text_width + 2, org[1] - text_height - 2))
        cv2.rectangle(image, box_coordinates[0], box_coordinates[1], (0, 255, 0), 2)
        # cv2.rectangle(image, text_box_coordinates[0], text_box_coordinates[1], (0, 0, 0), cv2.FILLED)
        cv2.putText(image, text, org, font, font_scale, (0, 255, 0), font_thickness)


if __name__ == "__main__":
    image_path = "../images/COCO/COCO_val2014_000000000328.jpg"
    imageToProcess = cv2.imread(image_path)
    pose_estimator = PoseEstimator(face=False, hand=False)
    image_id = pose_estimator.processImage(imageToProcess, image_path)
    outputImage = pose_estimator.getOutputImage()
    keypoints = pose_estimator.getPoseKeypoints()

    id = 0
    for keypoint in keypoints:
        ImageProcessing.outputIndividualIdToImage(outputImage, id, keypoint)
        id += 1

    cv2.imshow(image_id, outputImage)
    cv2.waitKey(0)
