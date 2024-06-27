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
                 tracking_confidence=0.5):
        # Initialize variables and MediaPipe hand detection module
        self.results = None
        self.mode = mode
        self.num_hands = num_hands
        self.complexity = complexity
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        # Set up MediaPipe hands module
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.num_hands, self.complexity,
                                        self.detection_confidence, self.tracking_confidence)
        # Set up drawing utilities for hand landmarks
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        # Convert the image from BGR to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the RGB image to detect hands
        self.results = self.hands.process(imgRGB)

        # Draw hand landmarks if any are detected and draw is True
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0, draw=True, point_radius=5) -> List:
        # Initialize list to hold landmark positions
        lm_list = []
        # Check if hand landmarks are detected
        if self.results and self.results.multi_hand_landmarks:
            # Select the specified hand
            my_hand = self.results.multi_hand_landmarks[hand_no]
            h, w, c = img.shape  # Get image dimensions
            # Iterate through each landmark in the hand
            for lm_id, lm in enumerate(my_hand.landmark):
                # Convert normalized coordinates to pixel coordinates
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([lm_id, cx, cy])
                # Draw a circle at the landmark if draw is True
                if draw:
                    cv2.circle(img, (cx, cy), point_radius, (255, 0, 255), cv2.FILLED)
        return lm_list


def main():
    # Initialize variables for calculating FPS
    p_time = 0
    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)
    # Initialize hand detector
    detector = HandDetector()

    while True:
        # Read a frame from the webcam
        success, img = cap.read()
        if not success:
            break
        # Find and draw hands on the frame
        img = detector.find_hands(img)
        # Get the positions of hand landmarks
        lm_list = detector.find_position(img, 0, False)
        if lm_list:
            print(lm_list[4])  # Print the position of landmark 4 (tip of the thumb)
        # Calculate FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        # Display FPS on the frame
        cv2.putText(img, "FPS: " + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        # Show the frame with hand landmarks
        cv2.imshow("Image", img)

        # Break the loop if 'Esc' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break
        # Check if the window is closed by looking if any windows are still open
        if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()