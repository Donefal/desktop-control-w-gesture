import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import argparse

import pickle

from gesture_detection import (
    MODE, FINGERS, 
    is_pointing, get_mode_from_left_hand
)
from gesture_actions import (
    handle_navigation_action,
    handle_presentation_action,
    handle_key_action,
    print_debug
)
from control import (
    move_mouse_from_index, 
    handle_stt, 
    move_mouse_relative, 
    rel_reset
)

SCREEN_W = 1920
SCREEN_H = 1080

CONFIDENCE_TRESHOLD = 0.7

MODE_MAPPING = {
    'Menggerakkan_Mouse':   MODE.NAVIGATION,
    'Shortcut_Paste':       MODE.KEY,
    'Key_Enter':            MODE.KEY,
    'Shortcut_Copy':        MODE.KEY,
    'Key_ESC':              MODE.KEY,
    'Idle':                 MODE.IDLE,
    'Key_Backspace':        MODE.IDLE,
    'Shortcut_Cut':         MODE.KEY,
    'Panah_Kiri':           MODE.PRESENTATION,
    'Klik_Kanan':           MODE.NAVIGATION,
    'Begin_STT':            MODE.STT,
    'Panah_Kanan':          MODE.PRESENTATION,
    'Klik_Kiri':            MODE.NAVIGATION,
    'Zoom_IN':              MODE.IDLE,
    'Zoom_Out':             MODE.IDLE,
}

MODE_MAPPING_NEW = {
    'key'       : MODE.KEY,
    'nav'       : MODE.NAVIGATION,
    'none'      : MODE.IDLE,
    'pres'      : MODE.PRESENTATION,
    'stt'       : MODE.STT
}

with open("models/gesture_classifier_new.pkl", "rb") as f:
    gesture_classifier = pickle.load(f)

print("Known classes:", gesture_classifier.classes_)

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

def fit_to_screen(frame, screen_w, screen_h):
    h, w = frame.shape[:2]
    scale = min(screen_w / w, screen_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    # Pad with black bars
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    x_offset = (screen_w - new_w) // 2
    y_offset = (screen_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

# -----------------------------------------------------------------------------------
# Action Handler
# -----------------------------------------------------------------------------------

def handle_action(mode, right_landmarks, classification):
    """Main action handler based on detected mode"""

    classified_mode = MODE_MAPPING_NEW.get(classification[0], MODE.IDLE)
    classified_confidence = classification[1]

    # print_debug(right_landmarks, mode)
    if mode == MODE.IDLE or right_landmarks is None:
        return

    # ONLY ACTIVATE IF THE DATA IS GOOD
    # if classified_mode == MODE.IDLE and classified_confidence < CONFIDENCE_TRESHOLD:
    #      return
    
    # if mode != classified_mode:
    #      return

    if mode == MODE.NAVIGATION:
        handle_navigation_action(right_landmarks)

    elif mode == MODE.PRESENTATION:
        handle_presentation_action(right_landmarks)

    elif mode == MODE.KEY:
        handle_key_action(right_landmarks)

    elif mode == MODE.STT:
        handle_stt(right_landmarks)


# -----------------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------------

latest_result = None

def process_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result
    
def classify_gesture_with_confidence(hand_landmarks):
    if hand_landmarks is None:
        return ['IDLE', 1]

    row = np.array([coord for point in hand_landmarks for coord in [point.x, point.y, point.z]]).reshape(1, -1)
    label = gesture_classifier.predict(row)[0]
    confidence = gesture_classifier.predict_proba(row).max()
    return [label, confidence]

base_options = python.BaseOptions(model_asset_path='./models/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=process_result
)

parser = argparse.ArgumentParser(description="Gesture Control")
parser.add_argument('--camera', type=int, default=0, help="Camera index to use")
args = parser.parse_args()

recognizer = vision.GestureRecognizer.create_from_options(options)
cap = cv2.VideoCapture(args.camera)
timestamp = 0

# -----------------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------------

# See what classes the model knows
print("Known classes:", gesture_classifier.classes_)

# See how many trees voted for each class on a test prediction
print("Number of estimators:", gesture_classifier.n_estimators)


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
        current_result = latest_result  # snapshot once
        
        try:
            gestures = current_result.gestures
            handedness_list = current_result.handedness
            landmarks_list = current_result.hand_landmarks

            # Verify all three lists are consistent
            if not (len(gestures) == len(handedness_list) == len(landmarks_list)):
                continue

            left_landmarks = None
            right_landmarks = None

            for i in range(len(gestures)):
                handedness = handedness_list[i][0].display_name
                handedness = "Right" if handedness == "Left" else "Left"
                lm = landmarks_list[i]

                if handedness == "Left":
                    left_landmarks = lm
                else:
                    right_landmarks = lm

                draw_landmarks(frame, lm)

            classification = classify_gesture_with_confidence(left_landmarks)
            mode = get_mode_from_left_hand(left_landmarks)
            handle_action(mode, right_landmarks, classification)
            cv2.putText(frame, f"Mode: {mode}", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        except (IndexError, AttributeError):
            pass  # skip this frame if result is malformed



    cv2.flip(frame, 1)
    # frame = fit_to_screen(frame, SCREEN_W, SCREEN_H)
    cv2.namedWindow("Gesture Recognition", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Gesture Recognition", cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()