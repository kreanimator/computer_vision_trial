import cv2
import numpy as np
from hand_tracking import HandTrackingModule as htm

menu = cv2.imread('./menu.png')
width = 1280
height = 720
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

detector = htm.HandDetector(detection_confidence=0.85)

# Initialize cursor position variables
cursor_x = None
cursor_y = None
selected_color = None


def draw_cursor(image, x, y, color):
    cv2.circle(image, (x, y), 20, color, cv2.FILLED)
    cv2.circle(image, (x, y), 30, color, 3)


while True:
    # 1. Import image
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img[0:120, 0:width] = menu

    # 2. Find hand landmarks
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, draw=False)  # Use draw=False to avoid drawing landmarks

    if len(lm_list) != 0:
        # Tip of index and middle finger
        x1, y1 = lm_list[8][1:]  # Index finger
        x2, y2 = lm_list[12][1:]  # Middle finger

        # 3. Check which fingers are up
        fingers = detector.detect_which_finger_is_up()

        # 4. If selection mode - Two fingers are up
        if fingers[1] and fingers[2]:
            if y1 < 120:
                if 200 < x1 < 350:
                    cursor_x, cursor_y = 245, 60
                    selected_color = (255, 255, 255)  # Purple
                elif 350 < x1 < 450:
                    cursor_x, cursor_y = 435, 60
                    selected_color = (255, 255, 255)  # Green
                elif 550 < x1 < 750:
                    cursor_x, cursor_y = 625, 60
                    selected_color = (255, 255, 255)  # Blue
                elif 750 < x1 < 850:
                    cursor_x, cursor_y = 815, 60
                    selected_color = (0, 0, 0)  # White
                elif 950 < x1 < 1050:
                    cursor_x, cursor_y = 1005, 60
                    selected_color = (255, 255, 255)  # Black
                elif 1150 < x1 < 1280:
                    cursor_x, cursor_y = 1195, 60
                    selected_color = (0, 0, 0)  # Eraser

        # 5. If draw mode - Index finger is up
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, (255, 255, 255), cv2.FILLED)
            print("Drawing mode")

    # Draw the cursor based on the selected color
    if selected_color is not None and cursor_x is not None and cursor_y is not None:
        draw_cursor(img, cursor_x, cursor_y, selected_color)

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


