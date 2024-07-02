__author__ = "Valentin Bakin"

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
p_time = 0

mp_faceMesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
face_detection = mp_faceMesh.FaceMesh(max_num_faces=2)
draw_specs = mp_draw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

while True:
    success, img = cap.read()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)

    if results.multi_face_landmarks:
        for face_lms in results.multi_face_landmarks:
            mp_draw.draw_landmarks(img, face_lms, mp_faceMesh.FACEMESH_CONTOURS, landmark_drawing_spec=draw_specs)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break
    # Check if the window is closed by looking if any windows are still open
    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break
