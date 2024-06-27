__author__ = "Valentin Bakin"

from typing import List
import cv2
import mediapipe as mp
import time

# Initialize Mediapipe Holistic model
mp_draw = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

cap = cv2.VideoCapture(0)
p_time = 0

while True:
    success, img = cap.read()
    if not success:
        break

    # Convert the BGR image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Process the RGB image with the Holistic model
    results = holistic.process(img_rgb)

    # Draw pose landmarks
    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        for lm_id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 3, (255, 0, 0), cv2.FILLED)
    # Draw face landmarks with smaller circle radius (adjust the value as needed)
    if results.face_landmarks:
        mp_draw.draw_landmarks(img, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                               landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
        for lm_id, lm in enumerate(results.face_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 1, (0, 255, 0), cv2.FILLED)
    # Draw left hand landmarks
    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        for lm_id, lm in enumerate(results.left_hand_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 3, (255, 255, 0), cv2.FILLED)
    # Draw right hand landmarks
    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        for lm_id, lm in enumerate(results.right_hand_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
    # Calculate and display FPS
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, "FPS: " + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    # Display the image
    cv2.imshow("Image", img)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
