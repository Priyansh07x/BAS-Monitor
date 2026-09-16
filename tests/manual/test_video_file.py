import cv2
import sys
cap = cv2.VideoCapture("data/videos/test_avc1.mp4")
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("Successfully read a frame from video file!")
    else:
        print("Failed to read frame")
else:
    print("Failed to open video file")
