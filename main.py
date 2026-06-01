import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

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

def handle_action(mode, right_landmarks, left_landmarks, frame_shape):
    """Main action handler based on detected mode"""

    # print_debug(right_landmarks, mode)
    if mode == MODE.IDLE or right_landmarks is None:
        return


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
    # frame = fit_to_screen(frame, SCREEN_W, SCREEN_H)
    cv2.namedWindow("Gesture Recognition", cv2.WINDOW_NORMAL)
    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()