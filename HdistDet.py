import cv2
import numpy as np
import time
import module as hmod
import hand_utils as hutil

# Camera setup
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("http://192.168.1.100:4747/video")
w, h = 1280, 720
cap.set(3, w)
cap.set(4, h)

x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)  # y = Ax^2 + Bx + C
A, B, C = coff

pTime = 0
hDetector = hmod.handDetector()

while True:
    success, fr = cap.read()
    fr = cv2.flip(fr, 1)
    fr = hDetector.findHands(fr, draw=False)
    lmList, bbox = hDetector.findPosition(fr, draw=True)

    if len(lmList) != 0:
        ln, fr, line = hDetector.findDistance(5, 17, fr, draw=True)  # base to middle tip
        distanceCM = A * ln ** 2 + B * ln + C
        hutil.draw_text(fr, f"Dist: {round(distanceCM, 2)}", 1100, 30)

    # FPS calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    hutil.draw_text(fr, f"{round(fps, 2)}", 10, 30)

    # Show the camera feed
    cv2.imshow("cam", fr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
