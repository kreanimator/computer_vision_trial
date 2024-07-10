import cv2
import numpy as np
from hand_tracking import HandTrackingModule as htm

menu = cv2.imread('./menu.png')
width = 1280
height = 720
brush_thickness = 15
eraser_thickness = 50
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

detector = htm.HandDetector(detection_confidence=0.65)

# Initialize cursor position variables
cursor_x = None
cursor_y = None
cursor_color = None
selected_color = (255, 255, 255)
xp, yp = 0, 0


def draw_cursor(image, x, y, color):
    cv2.circle(image, (x, y), 20, color, cv2.FILLED)
    cv2.circle(image, (x, y), 30, color, 3)


img_canvas = np.full((720, 1280, 3), (100, 100, 100), np.uint8)

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
                # Purple
                if 200 < x1 < 350:
                    cursor_x, cursor_y = 245, 60
                    cursor_color = (255, 255, 255)
                    selected_color = (255, 0, 255)
                # Green
                elif 350 < x1 < 450:
                    cursor_x, cursor_y = 435, 60
                    cursor_color = (255, 255, 255)
                    selected_color = (0, 255, 0)
                # Blue
                elif 550 < x1 < 750:
                    cursor_x, cursor_y = 625, 60
                    cursor_color = (255, 255, 255)
                    selected_color = (255, 0, 0)
                # White
                elif 750 < x1 < 850:
                    cursor_x, cursor_y = 815, 60
                    cursor_color = (0, 0, 0)
                    selected_color = (255, 255, 255)
                # Black
                elif 950 < x1 < 1050:
                    cursor_x, cursor_y = 1005, 60
                    cursor_color = (255, 255, 255)
                    selected_color = (0, 0, 0)
                # Eraser
                elif 1150 < x1 < 1280:
                    cursor_x, cursor_y = 1195, 60
                    cursor_color = (0, 0, 0)
                    selected_color =(100, 100, 100)
            cv2.circle(img, (x1, y1), 30, selected_color, 3)

        # 5. If draw mode - Index finger is up
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, selected_color, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if selected_color == (100,100,100):
                cv2.line(img, (xp, yp), (x1, y1), selected_color, eraser_thickness)
                cv2.line(img_canvas, (xp, yp), (x1, y1), selected_color, eraser_thickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), selected_color, brush_thickness)
                cv2.line(img_canvas, (xp, yp), (x1, y1), selected_color, brush_thickness)
            xp, yp = x1, y1
    # Draw the cursor based on the selected color
    if selected_color is not None and cursor_x is not None and cursor_y is not None:
        draw_cursor(img, cursor_x, cursor_y, cursor_color)

    cv2.imshow("Image", img)
    cv2.imshow("Canvas", img_canvas)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break

    # Check if the window is closed
    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
