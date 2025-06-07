import cv2
import numpy as np
#import math
#import mediapipe as mp
import module as hmod
import autopy
import time
import sys


screen_width, screen_height = autopy.screen.size()
frame_width, frame_height = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)

pTime = 0

hDetector = hmod.handDetector()

prev_x, prev_y = 0, 0
smoothing = 5  # smaller is smoother
autopy.mouse.toggle(down = False)

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)

    img = hDetector.findHands(img, draw=True)
    lmList, bbox = hDetector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print("lmList: ", lmList)
        xT, yT = lmList[4][1], lmList[4][2]
        xP, yP = lmList[8][1], lmList[8][2]
        xI, yI = lmList[20][1], lmList[20][2]
        cv2.circle(img, (int(xP), int(yP)), 10, (255, 0, 0), cv2.FILLED)
        cv2.putText(img, "L-Click / Drag", (xT-15, yT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, "Mouse", (xP - 15, yP - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, "R-Click", (xI - 15, yI - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    else:
        xT, yT, xP, yP, xM, yM = None, None, None, None, None, None
    #print(xT, yT, xP, yP, xM, yM)
    f, hSide = hDetector.fingersUp()
    cv2.rectangle(img, (100, 100), (frame_width - 100, frame_height - 100), (255, 255, 0), 2)
    #print(f, hSide)
    if len(lmList) != 0:
        if f[1] == 1 and 100<=xP<= frame_width-100 and 100<=yP<= frame_height-100 and f[2] != 1:
            mapped_x = np.interp(xP, (100, frame_width-100), (0, screen_width))
            mapped_y = np.interp(yP, (100, frame_height-100), (0, screen_height))
            mapped_x = max(0, min(mapped_x, screen_width - 1))
            mapped_y = max(0, min(mapped_y, screen_height - 1))
            mapped_x = prev_x + (mapped_x - prev_x) / smoothing
            mapped_y = prev_y + (mapped_y - prev_y) / smoothing
            autopy.mouse.move(mapped_x, mapped_y)
            prev_x, prev_y = mapped_x, mapped_y
            #print(f)
            if f[0] == 1:  #  Correctly checking if thumb is up
                autopy.mouse.toggle(down=True)  # Start drag
                time.sleep(0.2)
                autopy.mouse.move(mapped_x, mapped_y)  # Continue drag
            else:
                autopy.mouse.toggle(down=False)  # Release drag
        if f[4] == 1:
            sys.exit(0)
        if f[1] == 1 and 100<=xP<= frame_width-100 and 100<=yP<= frame_height-100 and f[2] == 1:
            autopy.mouse.toggle(down=False)
            length, img, lineInfo = hDetector.findDistance(8, 12, img)
            #print(length)
            if length < 30:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"fps: {round(fps, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
