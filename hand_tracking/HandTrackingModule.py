from typing import List

import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self,
                 mode=False,
                 num_hands=2,
                 complexity=1,
                 detection_confidence=0.5,
                 tracking_confidence=0.5
                 ):
        self.results = None
        self.mode = mode
        self.num_hands = num_hands
        self.complexity = complexity
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.num_hands, self.complexity,
                                        self.detection_confidence, self.tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Capture hands on the screen
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        # Display hand landmarks on displayed image
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, hand_no=0, draw=True) -> List:

        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            # Get id and landmark information
            for lm_id, lm in enumerate(my_hand.landmark):
                # Height, width, center
                h, w, c = img.shape
                # Center x and center y
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([lm_id, cx, cy])
                # if id ==0:
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return lm_list


def main():

    p_time = 0
    c_time = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position(img, 0, False)
        if len(lm_list) != 0:
            print(lm_list[4])
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        # Display FPS on the screen
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

        # Display img from the webcam
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
