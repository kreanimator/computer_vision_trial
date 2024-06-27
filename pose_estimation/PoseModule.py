__author__ = "Valentin Bakin"

from typing import List
import cv2
import mediapipe as mp
import time


class PoseDetector:
    def __init__(self, mode=False,
                 complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 refine_face_landmarks=False,
                 detection_confidence=0.5,
                 tracking_confidence=0.5):

        self.mode = mode
        self.complexity = complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.refine_face_landmarks = refine_face_landmarks
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        # Initialize Mediapipe Holistic model
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(self.mode, self.complexity, self.smooth_landmarks,
                                                  self.enable_segmentation, self.smooth_segmentation,
                                                  self.refine_face_landmarks, self.detection_confidence,
                                                  self.tracking_confidence)

    def find_pose(self, img, draw=True):
        # Convert the BGR image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the RGB image with the Holistic model
        results = self.holistic.process(img_rgb)

        # Draw pose landmarks
        if draw:
            if results.pose_landmarks:
                self.draw_landmarks(img, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS, (255, 0, 0), draw)

    def find_face(self, img, draw=True):
        # Convert the BGR image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the RGB image with the Holistic model
        results = self.holistic.process(img_rgb)
        if draw:
            if results.face_landmarks:
                self.draw_landmarks(img, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION, (0, 255, 0), draw, radius=2)

    def find_left_hand(self, img, draw=True):
        # Convert the BGR image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the RGB image with the Holistic model
        results = self.holistic.process(img_rgb)
        # Draw left hand landmarks
        if draw:
            if results.left_hand_landmarks:
                self.draw_landmarks(img, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, (0, 0, 255), draw)

    def find_right_hand(self, img, draw=True):
        # Convert the BGR image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the RGB image with the Holistic model
        results = self.holistic.process(img_rgb)
        # Draw right hand landmarks
        if draw:
            if results.right_hand_landmarks:
                self.draw_landmarks(img, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, (255, 0, 0), draw)

    def draw_landmarks(self, img, landmarks, connections, color, draw=True, radius=4):
        if draw:
            self.mp_draw.draw_landmarks(img, landmarks, connections,
                                        landmark_drawing_spec=self.mp_draw.DrawingSpec(color=color, thickness=1,
                                                                                       circle_radius=radius))

def main():
    cap = cv2.VideoCapture(0)
    p_time = 0

    # Initialize detectors
    pose_detector = PoseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        # Detect pose and draw landmarks
        pose_detector.find_pose(img)
        # Detect face and draw landmarks
        pose_detector.find_face(img)
        # Detect and draw left hand landmarks
        pose_detector.find_left_hand(img)
        # Detect and draw right hand landmarks
        pose_detector.find_right_hand(img)

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
        # Check if the window is closed by looking if any windows are still open
        if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
