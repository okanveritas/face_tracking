import cv2
import mediapipe as mp
import time
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

keyboard = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/', '<']  # Silme tuşunu ekledim
]

keyboard_top_left = (100, 100)

key_hitboxes = {}

for i, row in enumerate(keyboard):
    for j, key in enumerate(row):
        x = keyboard_top_left[0] + j * 100
        y = keyboard_top_left[1] + i * 100
        key_hitboxes[key] = [(x - 50, y - 50), (x + 50, y + 50)]

current_letter = ''
key_pressed = False
key_timer = 0
key_displayed = False
typed_text = ''

textbox_rect = [(400, 600), (880, 680)]

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

cv2.namedWindow('Virtual Keyboard', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Virtual Keyboard', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def close_window(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.destroyAllWindows()

cv2.setMouseCallback('Virtual Keyboard', close_window)

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(landmarks.landmark):
                h, w, c = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)

                if id == 8:
                    for key, (hitbox_min, hitbox_max) in key_hitboxes.items():
                        if hitbox_min[0] <= x <= hitbox_max[0] and hitbox_min[1] <= y <= hitbox_max[1]:
                            if calculate_distance((lm.x, lm.y), (landmarks.landmark[12].x, landmarks.landmark[12].y)) < 0.05:  # Eşik değeri burada ayarlayabilirsiniz
                                current_letter = key
                                key_pressed = True
                            break
                    else:
                        current_letter = ''
                        key_pressed = False

            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    if key_pressed and not key_displayed:
        key_timer = time.time()
        key_displayed = True

    for key, (hitbox_min, hitbox_max) in key_hitboxes.items():
        if current_letter == key and key_pressed:
            cv2.rectangle(frame, hitbox_min, hitbox_max, (0, 255, 0), -1)
            cv2.putText(frame, key, (hitbox_min[0] + 40, hitbox_min[1] + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.rectangle(frame, hitbox_min, hitbox_max, (0, 0, 255), 2)
            cv2.putText(frame, key, (hitbox_min[0] + 40, hitbox_min[1] + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if key_displayed and time.time() - key_timer >= 0.8:
        if current_letter == '<':
            typed_text = typed_text[:-1]
        else:
            typed_text += current_letter
        key_displayed = False

    cv2.rectangle(frame, textbox_rect[0], textbox_rect[1], (0, 0, 0), -1)
    cv2.putText(frame, typed_text, (textbox_rect[0][0] + 10, textbox_rect[0][1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    cv2.imshow('Virtual Keyboard', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
