# from <https://sefiks.com/2019/07/21/face-recognition-with-openface-in-keras/>

import cv2
import os
import numpy as np
from Utility import Utility


class plsplsplsWork():
    def __init__(self):
        pass

    def faceDetect(self, image):
        # Load the cascade
        face_cascade = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')

        # Read the input image
        # img = cv2.imread(image)
        # Convert into grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_color = image[y:y + h, x:x + w]
            return roi_color

    def tryingThis(self, image):
        dst = cv2.dct(np.float32(image))
        # cv2.idct(image, dst)
