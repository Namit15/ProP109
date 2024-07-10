import numpy as np
import pyautogui
import imutils
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]
thumb_tip = 4
screenshot_taken = False

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            # Accessing the landmarks by their position
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            # Array to hold true or false if finger is folded    
            finger_fold_status = []
            for tip in finger_tips:
                # Getting the landmark tip position and drawing blue circle
                x, y = int(lm_list[tip].x * w), int(lm_list[tip].y * h)
                cv2.circle(img, (x, y), 15, (255, 0, 0), cv2.FILLED)

                # Writing condition to check if finger is folded
                # If finger folded changing color to green
                if lm_list[tip].x < lm_list[tip - 3].x:
                    cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            # Checking if thumb is folded and all other fingers are extended
            thumb_folded = lm_list[thumb_tip].x < lm_list[thumb_tip - 2].x
            fingers_extended = all([not status for status in finger_fold_status])

            if thumb_folded and fingers_extended and not screenshot_taken:
                pyautogui.screenshot("screenshot.png")
                screenshot_taken = True
                print("Screenshot taken")

            # Reset screenshot_taken if gesture is not detected
            if not (thumb_folded and fingers_extended):
                screenshot_taken = False

            mp_draw.draw_landmarks(
                img, hand_landmark, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec((0, 0, 255), 2, 2),
                mp_draw.DrawingSpec((0, 255, 0), 4, 2)
            )

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Exit when 'ESC' is pressed
        break

cap.release()
cv2.destroyAllWindows()