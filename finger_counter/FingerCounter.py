__author__ = "Valentin Bakin"

import cv2
import time
import math
import numpy as np
import os

from hand_tracking import HandTrackingModule as htm

w_cam, h_cam = 1024, 768

cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)
p_time = 0

detector = htm.HandDetector(detection_confidence=0.7)

while True:
    success, img = cap.read()
    cv2.imshow("Image", img)
    cv2.waitKey(1)
