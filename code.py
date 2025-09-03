import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe and webcam
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

screen_width, screen_height = pyautogui.size()
frame_width, frame_height = 640, 480

cap.set(3, frame_width)
cap.set(4, frame_height)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip and convert frame to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark

            # Index finger tip (landmark 8)
            x = int(landmarks[8].x * frame_width)
            y = int(landmarks[8].y * frame_height)

            # Convert to screen coordinates
            screen_x = int((landmarks[8].x) * screen_width)
            screen_y = int((landmarks[8].y) * screen_height)
            pyautogui.moveTo(screen_x, screen_y)

            # Check for click gesture (thumb tip and index tip close)
            index_tip = landmarks[8]
            thumb_tip = landmarks[4]
            distance = ((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2) ** 0.5

            if distance < 0.03:
                pyautogui.click()
                pyautogui.sleep(1)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
