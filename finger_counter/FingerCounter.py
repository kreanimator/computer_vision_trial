__author__ = "Valentin Bakin"

import cv2
import time
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
tip_ids = [4, 8, 12, 16, 20]


while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, draw=False)

    if len(lm_list) != 0:
        fingers = []
        #  Thumb
        if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Rest fingers (only for right hand for now)
        for f_id in range(1,5):
            if lm_list[tip_ids[f_id]][2] < lm_list[tip_ids[f_id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        total_fingers = fingers.count(1)
        print(total_fingers)

        h, w, c = overlay_list[total_fingers -1].shape
        img[0:h, 0:w] = overlay_list[total_fingers -1]

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
