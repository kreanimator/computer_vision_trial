__author__ = "Valentin Bakin"

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
p_time = 0

mp_faceMesh_detection = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
face_detection = mp_faceMesh_detection.FaceMesh()

while True:
    success, img = cap.read()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break
    # Check if the window is closed by looking if any windows are still open
    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break
