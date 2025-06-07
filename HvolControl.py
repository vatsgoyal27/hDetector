import cv2
import numpy as np
import time
import module as hmod
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

# Initialize Pycaw to control system volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()  # e.g., (-65.25, 0.0, 0.03125)
minVol = volRange[0]
maxVol = volRange[1]

# Camera setup
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("http://192.168.1.100:4747/video")
w, h = 1280, 720
cap.set(3, w)
cap.set(4, h)

pTime = 0
hDetector = hmod.handDetector()

while True:
    success, fr = cap.read()
    fr = hDetector.findHands(fr, draw=True)
    lmList, bbox = hDetector.findPosition(fr, draw=False)

    if len(lmList) != 0:
        ln, fr, line = hDetector.findDistance(4, 8, fr, draw=True)  # Thumb tip & index tip
        if ln < 55:
            ln = 0
        elif ln > 280:
            ln = 300
        # Map ln (distance) to volume range
        vol = np.interp(ln, [50, 300], [minVol, maxVol])  # Adjust 20–250 as needed for your reach
        volume.SetMasterVolumeLevel(vol, None)

        # Draw volume bar
        volBar = np.interp(ln, [20, 250], [400, 150])  # Y-position
        cv2.rectangle(fr, (50, 150), (85, 400), (0, 255, 0), 2)
        cv2.rectangle(fr, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)

        volPercent = np.interp(ln, [20, 250], [0, 100])
        cv2.putText(fr, f'{int(volPercent)} %', (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # FPS calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(fr, f'FPS: {int(fps)}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Show the camera feed
    cv2.imshow("cam", fr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
