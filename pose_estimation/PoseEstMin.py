__author__ = "Valentin Bakin"

from typing import List
import cv2
import mediapipe as mp
import time

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.holistic
pose = mp_pose.Holistic()

cap = cv2.VideoCapture(0)
p_time = 0
while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    c_time = time.time()
    fps = 1/(c_time-p_time)
    p_time = c_time

    # Display FPS on the frame
    cv2.putText(img, "FPS: " + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
