import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Capture hands on the screen
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    # Display hand landmarks on displayed image
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Get id and landmark information
            for id, lm in enumerate(handLms.landmark):
                # Height, width, center
                h, w, c = img.shape
                # Center x and center y
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                # if id ==0:
                #     cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    # Display FPS on the screen
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

    # Display img from the webcam
    cv2.imshow("Image", img)
    cv2.waitKey(1)
