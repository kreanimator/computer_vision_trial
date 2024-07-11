__author__ = "Valentin Bakin"

import cv2
import numpy as np
from hand_tracking import HandTrackingModule as htm
import pynput

width = 640
height = 480

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

detector = htm.HandDetector(detection_confidence=0.85)

while True:
    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find hand landmarks
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, draw=False)  # Use draw=False to avoid drawing landmarks

    if len(lm_list) != 0:

        # Tip of index and middle finger
        x1, y1 = lm_list[8][1:]  # Index finger
        x2, y2 = lm_list[12][1:]  # Middle finger

        # 3. Check which fingers are up
        fingers = detector.detect_which_finger_is_up()

        # 4. If moving mode - Index finger is up
        if fingers[1] and not fingers[2]:
            pass

        #  5. Convert coordinates
        #  6. Smoothen values
        #  7. Move mouse
        #  8. If clicking mode - Two fingers are up
        if fingers[1] and fingers[2]:
            pass
        #  9. Find distance

    cv2.imshow("Image", img)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break

    # Check if the window is closed
    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
