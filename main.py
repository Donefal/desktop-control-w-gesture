import math

import cv2
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from enum import StrEnum, auto

from control import move_mouse_from_index

# -----------------------------------------------------------------------------------
# Constant & Enum
# -----------------------------------------------------------------------------------
PINCH_THRESHOLD = 0.05

class MODE(StrEnum):
    NAVIGATION = auto()
    KEY = auto()
    PRESENTATION = auto()
    STT = auto()
    IDLE = auto()

class FINGERS(StrEnum):
    THUMB = auto()
    INDEX = auto()
    MIDDLE = auto()
    RING = auto()
    PINKY = auto()

# -----------------------------------------------------------------------------------
# Fungsi-fungsi pendeteksi jari
# -----------------------------------------------------------------------------------

# Melihat knodisi masing-masing jari pada satu tangan, apakah extended atau tidak
def get_finger_extended(hand_landmarks):
    lm = hand_landmarks
    fingers = []

    # Four fingers: tip.y < knuckle.y means extended
    for tip, knuckle in [(8,5), (12,9), (16,13), (20,17)]:
        fingers.append(lm[tip].y < lm[knuckle].y)

    # Thumb: compare x axis instead
    thumb_extended = lm[4].x > lm[3].x  # flip if mirrored
    fingers.insert(0, thumb_extended)

    return fingers  # [thumb, index, middle, ring, pinky]

# Menghitung berapa jari yang extended
# EXCEPTION: thumb
def count_fingers(fingers):
    fingers = fingers[1:]
    return sum(fingers)

# Menghitung jarak antara 2 landmark
def landmark_distance(lm, id1, id2):
    dx = lm[id1].x - lm[id2].x
    dy = lm[id1].y - lm[id2].y
    return math.sqrt(dx**2 + dy**2)  # normalized, so ~0.05 is a pinch

def is_pointing(left_landmarks):
    fingers = get_finger_extended(left_landmarks)
    # Only index finger up, rest folded
    return fingers == [False, True, False, False, False]

def get_pinch(hand_landmarks):
    """
    Return jari yang pinching

    Return satu value dari enum FINGERS
    """
    thumb_tip = hand_landmarks[4]
    
    fingers = {
        FINGERS.INDEX:  hand_landmarks[8],
        FINGERS.MIDDLE: hand_landmarks[12],
        FINGERS.RING:   hand_landmarks[16],
        FINGERS.PINKY:  hand_landmarks[20],
    }
    
    for finger_name, finger_tip in fingers.items():
        dx = thumb_tip.x - finger_tip.x
        dy = thumb_tip.y - finger_tip.y
        dist = (dx**2 + dy**2) ** 0.5
        
        if dist < PINCH_THRESHOLD:
            return finger_name 
    
    return None  # no pinch detected
    
# -----------------------------------------------------------------------------------
# Mode-mode an
# -----------------------------------------------------------------------------------
def get_mode_from_left_hand(left_landmarks):
    """ Mendapatkan mode dari tangan kiri, return enum MODE"""
    if left_landmarks is None:
        return MODE.IDLE
    fingers = get_finger_extended(left_landmarks)
    count = count_fingers(fingers)

    if count == 1:
        return MODE.NAVIGATION
    elif count == 2:
        return MODE.KEY
    elif count == 3:
        return MODE.PRESENTATION
    elif count == 4:
        return MODE.STT
    return MODE.IDLE

def handle_navigation_action():
    """
        Aksi untuk MODE.NAVIGATION
        - Pinch telunjuk --> Klik Kiri
        - Pinch tengah --> Klik kanan
        - Pinch manis --> Zoom in (ctrl + '+')
        - Pinch kelingking --> Zoom out (Ctrl + '-')
    """
    global prev_pinch
    pinch_finger = get_pinch(right_landmarks)

    if(pinch_finger == FINGERS.INDEX):
        pyautogui.click();
    elif(pinch_finger == FINGERS.MIDDLE):
        pyautogui.rightClick()
    elif(pinch_finger == FINGERS.RING):
        pyautogui.hotkey('ctrl', '+')
    elif(pinch_finger == FINGERS.PINKY):
        pyautogui.hotkey('ctrl', '-')
    
    prev_pinch = pinch_finger

def handle_action(mode, right_landmarks, left_landmarks, frame_shape):
    """ Fungsi utama untuk handle aksi """
    if mode == MODE.IDLE:
        return

    fingers = get_finger_extended(left_landmarks)
    count = count_fingers(fingers)

    # TODO: Handle semua action disini
    if mode == MODE.NAVIGATION:
        if(is_pointing(left_landmarks)):
            # TODO: Move_mouse from index masih rada kaku (dia juga kyk cmn bisa setengah window aja gk bisa kekanan)
            move_mouse_from_index(left_landmarks, frame_shape)
        elif(right_landmarks is not None):
            handle_navigation_action()

    elif mode == MODE.KEY:
        pass  # keyboard shortcuts here
    elif mode == MODE.PRESENTATION:
        pass  # slide navigation here
    elif mode == MODE.STT:
        pass  # slide navigation here
    

# -----------------------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------------------

# Untuk menggambar Landmark (garis hijau & bintik merah)
def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
        
    # Draw connections manually
    connections = [
        (0,1),(1,2),(2,3),(3,4),         # thumb
        (0,5),(5,6),(6,7),(7,8),         # index
        (0,9),(9,10),(10,11),(11,12),    # middle
        (0,13),(13,14),(14,15),(15,16),  # ring
        (0,17),(17,18),(18,19),(19,20),  # pinky
        (5,9),(9,13),(13,17)             # palm
    ]

    for start, end in connections:
        x1, y1 = int(hand_landmarks[start].x * w), int(hand_landmarks[start].y * h)
        x2, y2 = int(hand_landmarks[end].x * w), int(hand_landmarks[end].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green line
    
    # Draw each landmark as a dot
    for lm in hand_landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)  # red dot


# -----------------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------------

latest_result = None
prev_pinch = None

def process_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

base_options = python.BaseOptions(model_asset_path='./models/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=process_result
)

recognizer = vision.GestureRecognizer.create_from_options(options)
cap = cv2.VideoCapture(0)
timestamp = 0

# -----------------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Use mp.Image directly instead of importing ImageFormat separately
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    timestamp += 1
    recognizer.recognize_async(mp_image, timestamp)

    if latest_result:
        left_landmarks = None
        right_landmarks = None

        current_result = latest_result
        if current_result and len(current_result.gestures) == len(current_result.handedness):
            for i, hand_gestures in enumerate(latest_result.gestures):
                handedness = latest_result.handedness[i][0].display_name
                lm = latest_result.hand_landmarks[i]

                # Flip the handedness label to match the mirrored frame
                handedness = "Right" if handedness == "Left" else "Left"

                if handedness == "Left":
                    left_landmarks = lm
                else:
                    right_landmarks = lm

                draw_landmarks(frame, lm)  

                gesture = hand_gestures[0]
                label = f"{handedness}: {gesture.category_name} ({gesture.score:.2f})"

            mode = get_mode_from_left_hand(left_landmarks)
            handle_action(mode, right_landmarks, left_landmarks, frame.shape)
            cv2.putText(frame, f"Mode: {mode}", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.flip(frame, 1)
    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
recognizer.close()