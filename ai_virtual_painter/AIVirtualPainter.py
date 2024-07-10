__author__ = "Valentin Bakin"

import cv2
import numpy as np
import time
from hand_tracking import HandTrackingModule as htm

menu = cv2.imread('./menu.png')
width = 1280
height = 720
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

detector = htm.HandDetector(detection_confidence=0.85)

while True:
    #  1.Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    #  2.  Find hand landmarks
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, False)

    if len(lm_list) != 0:
        #  Tip of index and middle finger
        x1, y1 = lm_list[8][1:]  # Index
        x2, y2 = lm_list[12][1:]  # Middle

    #  3. Check which fingers are up
    #  4. If selection mode - Two fingers are up
    #  5. If draw mode - Index finger is up

    img[0:120, 0:width] = menu
    cv2.imshow("Image", img)
    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break
    # Check if the window is closed
    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break
