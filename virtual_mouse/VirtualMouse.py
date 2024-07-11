__author__ = "Valentin Bakin"

import cv2
from hand_tracking import HandTrackingModule as htm
import numpy as np
from pynput.mouse import Controller, Button
from screeninfo import get_monitors
import time

width = 640
height = 480

frame_red = 100

monitor = get_monitors()[0]
width_screen, height_screen = monitor.width, monitor.height

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

mouse = Controller()

prev_x, prev_y = 0, 0
smooth_factor = 5

detector = htm.HandDetector(detection_confidence=0.85)
p_time = 0
while True:
    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find hand landmarks
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, draw=False)  # Use draw=False to avoid drawing landmarks

    if len(lm_list) != 0:

        # Tip of index and middle finger
        x1, y1 = lm_list[8][1:]  # Index finger
        x2, y2 = lm_list[12][1:]  # Middle finger
        x4, y4 = lm_list[20][1:] #  Pinky finger

        # 3. Check which fingers are up
        fingers = detector.detect_which_finger_is_up()
        cv2.rectangle(img, (frame_red, frame_red), (width - frame_red, height - frame_red), (0, 255, 0), 2)
        # 4. If moving mode - Index finger is up
        if fingers[1] and not fingers[2]:
            print("Moving mode")
            #  5. Convert coordinates

            x3 = np.interp(x1, (frame_red, width-frame_red), (0, width_screen))
            y3 = np.interp(y1, (frame_red, height-frame_red), (0, height_screen))

        #  6. Smoothen values
            x_smooth = prev_x + (x3 - prev_x) / smooth_factor
            y_smooth = prev_y + (y3 - prev_y) / smooth_factor
        #  7. Move mouse
            mouse.position = (x_smooth, y_smooth)
            prev_x, prev_y = x_smooth, y_smooth
        #  8. If clicking mode - Two fingers are up
        if fingers[1] and fingers[2]:
            print("Clicking mode")
            length, img, _ = detector.find_distance(8,12, img)
            if length <= 40:
                mouse.click(Button.left, 1)

        if fingers[1] and fingers[4] and not fingers[2]:
            print("Right mouse click")
            mouse.click(Button.right, 1)
        #  9. Find distance between fingers

        #  10. Click if distance short

    #  11.  Frame rate
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    cv2.putText(img, f"FPS: {int(fps)}", (400, 70), cv2.FONT_HERSHEY_PLAIN,
                1.5, (0, 255, 0), 1)
    cv2.imshow("Image", img)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break

    # Check if the window is closed
    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
