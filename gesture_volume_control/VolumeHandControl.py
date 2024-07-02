__author__ = "Valentin Bakin"

import cv2
import time
import numpy as np

from hand_tracking import HandTrackingModule as htm

w_cam, h_cam = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)
p_time = 0

detector = htm.HandDetector(detection_confidence=0.7)

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, draw=False)
    if len(lm_list) != 0:
        print(lm_list[4], lm_list[8])
        x1, y1 = lm_list[4][1], lm_list[4][2]  # Thumb
        x2, y2 = lm_list[8][1], lm_list[8][2]  # Index finger

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  # Center of the line
        cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (0, 255, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (200, 255, 0), 3)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                1, (0, 255, 0), 1)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break
    # Check if the window is closed by looking if any windows are still open
    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break
