import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
p_time = 0

while True:
    success, img = cap.read()
    cv2.imshow("Image", img)
    c_time = time.time()
    fps = 1/(c_time/p_time)
    p_time = c_time
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 2)

    cv2.waitKey(1)
