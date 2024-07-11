__author__ = "Valentin Bakin"

import cv2
import numpy as np
from hand_tracking import HandTrackingModule as htm

width = 1280
height = 720

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

detector = htm.HandDetector(detection_confidence=0.85)
