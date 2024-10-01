import cv2
import mediapipe as mp
import pyautogui

pyautogui.FAILSAFE = False
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Function to improve lighting conditions
def adjust_lighting(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value = 50  # increase brightness
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

# Function to calculate distance between two landmarks
def calculate_distance(landmark1, landmark2):
    return ((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2) ** 0.5

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    # Adjust lighting to improve hand detection in backlight
    frame = adjust_lighting(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = hand_landmarks.landmark
            
            # Calculate distances for checking if hand is open
            thumb_index_distance = calculate_distance(landmarks[mp_hands.HandLandmark.THUMB_TIP], landmarks[mp_hands.HandLandmark.THUMB_MCP])
            index_base_distance = calculate_distance(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP], landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP])
            middle_base_distance = calculate_distance(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP], landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP])
            ring_base_distance = calculate_distance(landmarks[mp_hands.HandLandmark.RING_FINGER_TIP], landmarks[mp_hands.HandLandmark.RING_FINGER_MCP])
            pinky_base_distance = calculate_distance(landmarks[mp_hands.HandLandmark.PINKY_TIP], landmarks[mp_hands.HandLandmark.PINKY_MCP])
            
            # Additional condition to check hand size for distance approximation
            wrist = landmarks[mp_hands.HandLandmark.WRIST]
            index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            hand_size = calculate_distance(wrist, index_finger_tip)
            
            # Filter based on hand size to approximate distance
            if hand_size > 0.15:  # Adjust based on real-time testing
                # More strict threshold for detecting fully open hand
                threshold = 0.2
                
                extended_fingers = sum([
                    thumb_index_distance > threshold,
                    index_base_distance > threshold,
                    middle_base_distance > threshold,
                    ring_base_distance > threshold,
                    pinky_base_distance > threshold
                ])
                
                if extended_fingers == 4:
                    pyautogui.press('space')
                    cv2.putText(frame, "JUMP", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
