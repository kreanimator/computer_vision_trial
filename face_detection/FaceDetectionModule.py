__author__ = "Valentin Bakin"

import cv2
import mediapipe as mp
import time


class FaceDetector:
    def __init__(self, min_detection_con=0.5):

        self.min_detection_con = min_detection_con
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_con)

    def find_faces(self, img, draw=True):

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(img_rgb)
        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bbox_c = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bbox_c.xmin * iw), int(bbox_c.ymin * ih), \
                    int(bbox_c.width * iw), int(bbox_c.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.modified_drawing(img, bbox)

                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            2, (255, 0, 255), 2)

        return img, bboxs

    def modified_drawing(self, img, bbox, length=30, thickness=3, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        # Top left corner x,y
        cv2.line(img, (x, y), (x + length, y), (255, 0, 255), thickness)
        cv2.line(img, (x, y), (x, y+length), (255, 0, 255), thickness)
        # Top right corner x1,y
        cv2.line(img, (x1, y), (x1 - length, y), (255, 0, 255), thickness)
        cv2.line(img, (x1, y), (x1, y+length), (255, 0, 255), thickness)
        # Bottom left corner x,y1
        cv2.line(img, (x, y1), (x + length, y1), (255, 0, 255), thickness)
        cv2.line(img, (x, y1), (x, y1 - length), (255, 0, 255), thickness)
        # Bottom right corner x1,y1
        cv2.line(img, (x1, y1), (x1 - length, y1), (255, 0, 255), thickness)
        cv2.line(img, (x1, y1), (x1, y1 - length), (255, 0, 255), thickness)

        return img


def main():
    cap = cv2.VideoCapture(0)
    p_time = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()

        img, bboxs = detector.find_faces(img)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

        # Break the loop if 'Esc' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break
        # Check if the window is closed by looking if any windows are still open
        if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
            break


if __name__ == "__main__":
    main()
