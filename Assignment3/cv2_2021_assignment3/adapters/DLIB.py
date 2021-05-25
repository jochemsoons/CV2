import cv2
import openface
import numpy as np

class Adapter:
    def __init__(self, dlib_weight_path):
        self.align = openface.AlignDlib(dlib_weight_path)

    def get_landmarks(self, image, d):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        landmarks = self.align.findLandmarks(gray_image, d)

        confidence = np.ones(68, dtype=np.float32)

        return landmarks, confidence
