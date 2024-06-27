__author__ = "Valentin Bakin"

from typing import List
import csv
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

        self.results = None
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

        # Initialize dictionaries to store landmark positions
        self.body_landmarks = []
        self.face_landmarks = []
        self.left_hand_landmarks = []
        self.right_hand_landmarks = []

    def find_pose(self, img, draw=True):
        # Convert the BGR image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the RGB image with the Holistic model
        self.results = self.holistic.process(img_rgb)

        # Draw pose landmarks
        if draw:
            if self.results.pose_landmarks:
                self.draw_landmarks(img, self.results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS, (255, 0, 0), draw)
                self.store_landmarks(img, self.results.pose_landmarks, 'body')

        return img

    def find_face(self, img, draw=True):
        # Process the RGB image with the Holistic model (removed redundant code)
        results = self.holistic.process(img)  # Use the processed image directly
        if draw:
            if results.face_landmarks:
                self.draw_landmarks(img, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION, (0, 255, 0), draw, radius=2)
                self.store_landmarks(img, results.face_landmarks, 'face')

        return img

    def find_left_hand(self, img, draw=True):
        # Process the RGB image with the Holistic model (removed redundant code)
        results = self.holistic.process(img)  # Use the processed image directly
        if draw:
            if results.left_hand_landmarks:
                self.draw_landmarks(img, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, (0, 0, 255), draw)
                self.store_landmarks(img, results.left_hand_landmarks, 'left_hand')

        return img

    def find_right_hand(self, img, draw=True):
        # Process the RGB image with the Holistic model (removed redundant code)
        results = self.holistic.process(img)  # Use the processed image directly
        if draw:
            if results.right_hand_landmarks:
                self.draw_landmarks(img, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, (255, 0, 0), draw)
                self.store_landmarks(img, results.right_hand_landmarks, 'right_hand')

        return img

    def draw_landmarks(self, img, landmarks, connections, color, draw=True, radius=4):
        if draw:
            self.mp_draw.draw_landmarks(img, landmarks, connections,
                                        landmark_drawing_spec=self.mp_draw.DrawingSpec(color=color, thickness=1,
                                                                                       circle_radius=radius))

    def store_landmarks(self, img, landmarks, part) -> List:
        lm_list = []
        h, w, c = img.shape  # Get image dimensions

        for lm_id, lm in enumerate(landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append([lm_id, cx, cy])

        # Store landmarks based on part
        if part == 'body':
            self.body_landmarks.append(lm_list)
        elif part == 'face':
            self.face_landmarks.append(lm_list)
        elif part == 'left_hand':
            self.left_hand_landmarks.append(lm_list)
        elif part == 'right_hand':
            self.right_hand_landmarks.append(lm_list)

        return lm_list

    def get_all_landmarks(self):
        return {
            'body': self.body_landmarks,
            'face': self.face_landmarks,
            'left_hand': self.left_hand_landmarks,
            'right_hand': self.right_hand_landmarks
        }

    @staticmethod
    def write_landmarks_to_csv(landmarks, filename='landmarks.csv'):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Part', 'ID', 'X', 'Y'])
            for part, lm_lists in landmarks.items():
                for lm_list in lm_lists:
                    for lm in lm_list:
                        writer.writerow([part, lm[0], lm[1], lm[2]])


def main():
    cap = cv2.VideoCapture(0)
    p_time = 0

    # Initialize detectors
    pose_detector = PoseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        # Process the image once and reuse the processed image
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect pose and draw landmarks
        pose_detector.find_pose(processed_img)
        # Detect face and draw landmarks
        pose_detector.find_face(processed_img)
        # Detect and draw left hand landmarks
        pose_detector.find_left_hand(processed_img)
        # Detect and draw right hand landmarks
        pose_detector.find_right_hand(processed_img)

        lm_list = pose_detector.get_all_landmarks()

        # Convert processed image back to BGR for display
        img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)

        # Calculate and display FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, "FPS: " + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        # Display the image
        cv2.imshow("Image", img)

        # Break the loop if 'Esc' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            PoseDetector.write_landmarks_to_csv(lm_list)
            break
        # Check if the window is closed by looking if any windows are still open
        if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            PoseDetector.write_landmarks_to_csv(lm_list)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
