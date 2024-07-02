__author__ = "Valentin Bakin"

import cv2
import time
import numpy as np


w_cam, h_cam = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)

while True:
    success, img = cap.read()

    cv2.imshow("Image", img)
    cv2.waitKey(1)


    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break
    # Check if the window is closed by looking if any windows are still open
    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break