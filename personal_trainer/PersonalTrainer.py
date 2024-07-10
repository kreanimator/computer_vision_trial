__author__ = "Valentin Bakin"

import cv2
import time
import numpy as np

from pose_estimation import PoseModule as pm

# Set up webcam
cap = cv2.VideoCapture(0)
w_cam, h_cam = 1280, 720

# Initialize the pose detector
detector = pm.PoseDetector()
count_lh = 0
count_rh = 0
direction_lh = 0
direction_rh = 0

while True:
    success, img = cap.read()
    if not success:
        break

    # Flip the image horizontally for a mirror view
    # img = cv2.flip(img, 1)
    # Resize the image for the webcam display
    img = cv2.resize(img, (w_cam, h_cam))

    # Detect pose landmarks without drawing them
    img = detector.find_pose(img, draw=False)

    # Retrieve all landmarks
    lm_dict = detector.get_all_landmarks()

    if lm_dict['body']:
        # Left arm angle up 340 bottom 180
        angle_lh = detector.find_angle(img, 11, 13, 15, draw=True)
        # Right arm angle up 30 bottom 180
        angle_rh = detector.find_angle(img, 12, 14, 16, draw=True)

        per_lh = np.interp(angle_lh, (30, 180), (100, 0))
        # print(f'Left hand angle: {angle_lh} percentage: {per_lh}')

        per_rh = np.interp(angle_rh, (180, 330), (0, 100))
        # print(f'Right hand angle: {angle_rh} percentage: {per_rh}')
        if per_lh == 100:
            if direction_lh == 0:
                count_lh += 0.5
        if per_rh == 100:
            if direction_rh == 0:
                count_rh += 0.5
    # Display the image
    cv2.imshow("Image", img)

    # Check for ESC key to break the loop
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    # Check if the window is closed
    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
