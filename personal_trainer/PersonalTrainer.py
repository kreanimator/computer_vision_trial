__author__ = "Valentin Bakin"

import cv2
import time
import numpy as np

from pose_estimation import PoseModule as pm


cap = cv2.VideoCapture(0)
w_cam, h_cam = 1280, 720

while True:
    success, img = cap.read()

    img = cv2.resize(img, (w_cam, h_cam))
    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break
    # Check if the window is closed
    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break
