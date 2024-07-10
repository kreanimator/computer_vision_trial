__author__ = "Valentin Bakin"

import cv2
import time
import numpy as np

from pose_estimation import PoseModule as pm

cap = cv2.VideoCapture(0)
w_cam, h_cam = 1280, 720

detector = pm.PoseDetector()

while True:
    success, img = cap.read()

    img = detector.find_pose(img)
    lm_list = detector.get_all_landmarks()
    # print(lm_list)
    left_arm = detector.find_angle(img, 11, 13, 15, True)
    right_arm = detector.find_angle(img, 12, 14, 16, True)
    if len(lm_list) != 0:
        pass
    img = cv2.resize(img, (w_cam, h_cam))
    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break
    # Check if the window is closed
    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break
