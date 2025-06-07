import cv2
import numpy as np
import math
import module as hmod
import sys
import time

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("http://192.168.1.100:4747/video")
cap.set(3, 1280)
cap.set(4, 720)

flag = 0
exit_start_time = 0
exit_delay = 3  # seconds
hDetector = hmod.handDetector()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    if not success:
        break

    img = hDetector.findHands(img)
    lmList, bbox = hDetector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers_up, label = hDetector.fingersUp()
        gesture = hDetector.recognizeGesture(fingers_up)

        print(f"You have {sum(fingers_up)} fingers up.")
        cv2.putText(img, label, (1200, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
        cv2.putText(img, f'Gesture: {gesture}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        if gesture == "Fuck You" and flag == 0:
            print("Gesture recognized: Fuck You. Exiting in 3 seconds...")
            flag = 1
            exit_start_time = time.time()

    # Check exit condition outside gesture block
    if flag == 1 and (time.time() - exit_start_time >= exit_delay):
        print("Exiting now.")
        sys.exit(80085)

    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
