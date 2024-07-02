__author__ = "Valentin Bakin"

import cv2
import time
import math
import numpy as np

from hand_tracking import HandTrackingModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

w_cam, h_cam = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, w_cam)
cap.set(4, h_cam)
p_time = 0

detector = htm.HandDetector(detection_confidence=0.7)

#  Audio init

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volume_range = volume.GetVolumeRange()

min_vol = volume_range[0]
max_vol = volume_range[1]
vol = 0
vol_bar = 400
vol_per = 0

while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, draw=False)
    if len(lm_list) != 0:
        # print(lm_list[4], lm_list[8])
        x1, y1 = lm_list[4][1], lm_list[4][2]  # Thumb
        x2, y2 = lm_list[8][1], lm_list[8][2]  # Index finger

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center of the line
        cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (0, 255, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (200, 255, 0), 3)
        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)

        #  Hand range 30 -> 210
        #  Volume range -65 -> 0

        vol = np.interp(length, [30, 210], [min_vol, max_vol])
        vol_bar = np.interp(length, [30, 210], [400, 150])
        vol_per = np.interp(length, [30, 210], [0, 100])
        print(volume)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 30:
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (0,255 , 0), 3)
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0,255 , 0), cv2.FILLED)
    cv2.putText(img, f': {int(vol_per)} %', (40, 450), cv2.FONT_HERSHEY_PLAIN,
                1, (0, 255, 0), 1)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                1, (0, 255, 0), 1)

    cv2.imshow("Image", img)
    # cv2.waitKey(1)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break
    # Check if the window is closed
    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break
