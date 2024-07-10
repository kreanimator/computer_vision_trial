__author__ = "Valentin Bakin"

import cv2
import numpy as np
import time
from hand_tracking import HandTrackingModule as htm


menu = cv2.imread('./menu.png')

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, img = cap.read()

    cv2.imshow("Image", img)
    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break
    # Check if the window is closed
    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break
