import cv2
import numpy as np
import time
import mediapipe as mp
import hand_utils as det

# Initialize webcam
frameWidth = 640
frameHeight = 480
fr = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("http://192.168.1.100:4747/video")
fr.set(3, frameWidth)
fr.set(4, frameHeight)
fr.set(10, 150)

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2)
mpDraw = mp.solutions.drawing_utils
handConnections = mpHands.HAND_CONNECTIONS

# Initialize previous time
pTime = 0

#Initialize tracked landmarks
hand1_landmark_idx = 8
hand2_landmark_idx = 8

#Initialise tagged landmark
hindex = "Right"
lindex = 0

while True:
    success, img = fr.read()
    if not success:
        print("Failed to read frame.")
        break

    img = cv2.flip(img, 1)
    black_img = np.zeros((frameHeight, frameWidth, 3), dtype=np.uint8)
    imgdet = img.copy()

    # Detect hands and get landmarks, and hand labels
    landmarks, hand_ids = det.detect_hands(imgdet, hands, frameWidth, frameHeight)

    # Draw landmarks on both copies
    imgdet = det.draw_hands(imgdet, landmarks, hand_ids)
    black_img = det.draw_hands(black_img, landmarks, hand_ids)

    # Draw line between hands based on specific landmarks
    imgline = imgdet.copy()
    imgline, dist = det.between_points(imgline, landmarks, hand_ids, hand1_landmark_idx, hand2_landmark_idx, color=(0, 255, 255), thickness=3, drawit = True)

    # Calculate and overlay FPS
    pTime, fps = det.fpscalc(pTime)

    imgdet = det.draw_text(imgdet, f"{int(fps)}", 10, 30)
    imgline = det.draw_text(imgline, f"{int(fps)}", 10, 30)

    # Track tagged landmark
    imgdet, xloc, yloc = det.loc(imgdet, landmarks, hindex, lindex, hand_ids, draw=True)
    print("hand:", hindex, f" landmark: {lindex}, at: ({xloc}, {yloc})")

    # Stack views
    imgStack = det.stackImages(0.8, ([img, imgdet], [imgline, black_img]))
    cv2.imshow("Detected Hands", imgStack)

    key = cv2.waitKey(1) & 0xFF
    # Change landmark indices with keys:

    # keys 1-5 and ` change left hand landmark index
    if key == ord('1'):
        hand1_landmark_idx = 4  # thumb tip
    elif key == ord('2'):
        hand1_landmark_idx = 8  # index tip
    elif key == ord('3'):
        hand1_landmark_idx = 12  # middle tip
    elif key == ord('4'):
        hand1_landmark_idx = 16  # ring tip
    elif key == ord('5'):
        hand1_landmark_idx = 20  # pinky tip
    elif key == ord('`'):
        hand1_landmark_idx = 0   # palm base

    # keys 6-0 and - change right hands landmark index
    elif key == ord('6'):
        hand2_landmark_idx = 4
    elif key == ord('7'):
        hand2_landmark_idx = 8
    elif key == ord('8'):
        hand2_landmark_idx = 12
    elif key == ord('9'):
        hand2_landmark_idx = 16
    elif key == ord('0'):
        hand2_landmark_idx = 20
    elif key == ord('-'):
        hand2_landmark_idx = 0

    # r corresponds to right hand, l corresponds to left hand for the tagged landmark
    if key == ord('r'):
        hindex = "Right"
    elif key == ord('l'):
        hindex = "Left"

    if key == ord('.'):
        if lindex == 20:
            lindex = 0
        else:
            lindex += 1
    elif key == ord(','):
        if lindex == 0:
            lindex = 20
        else:
            lindex -= 1

    # s takes a screenshot
    if key == ord('s'):
        timestamp = int(time.time())
        filename = f"screenshot_{timestamp}.png"
        cv2.imwrite(filename, imgStack)
        print(f"[Saved] {filename}")

    # q quits the process
    if key == ord('q'):
        break

fr.release()
cv2.destroyAllWindows()
