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

folder_path = "../finger_images"
my_list = os.listdir(folder_path)
overlay_list = []
for im_path in my_list:
    image = cv2.imread(f'{folder_path}/{im_path}')
    overlay_list.append(image)
detector = htm.HandDetector(detection_confidence=0.7)

while True:
    success, img = cap.read()

    h, w, c = overlay_list[0].shape
    img[0:h, 0:w] = overlay_list[0]

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.putText(img, f"FPS: {int(fps)}", (400, 70), cv2.FONT_HERSHEY_PLAIN,
                1.5, (0, 255, 0), 1)

    cv2.imshow("Image", img)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break
    # Check if the window is closed
    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break
