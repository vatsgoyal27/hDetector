import cv2
import numpy as np
import time
import os
import sys
import subprocess
import module as hmod


def is_notepad_running():
    result = subprocess.run("tasklist", capture_output=True, text=True)
    return "notepad.exe" in result.stdout.lower()


cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("http://192.168.1.100:4747/video")
w = 1280
h = 720
cap.set(3, w)
cap.set(4, h)
pTime = 0
hDetector = hmod.handDetector()

prev_state = None  # To track the previous gesture state

while True:
    success, fr = cap.read()
    fr = hDetector.findHands(fr, draw=True)
    lmList, bbox = hDetector.findPosition(fr, draw=False)

    if len(lmList) != 0:
        ln, fr, line = hDetector.findDistance(4, 8, fr, draw=True)

        # Determine current gesture state
        state = None
        if ln < 50:
            state = "close"
        elif ln > 300:
            state = "open"
        else:
            state = "idle"

        # Execute only if state has changed
        if state != prev_state:
            if state == "close":
                if is_notepad_running():
                    print("Notepad is running. Closing it...")
                    os.system("taskkill /f /im notepad.exe")
                else:
                    print("Notepad is not running.")
            elif state == "open":
                if not is_notepad_running():
                    print("Notepad is opening...")
                    os.system("start notepad")
                else:
                    print("Notepad is already running.")
            prev_state = state  # Update previous state

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(fr, f'FPS: {int(fps)}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("cam", fr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
