import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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


# SETUP
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

# MAIN LOOP
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
        for i, hand_gestures in enumerate(latest_result.gestures):
            gesture = hand_gestures[0]
            handedness = latest_result.handedness[i][0].display_name
            label = f"{handedness}: {gesture.category_name} ({gesture.score:.2f})"
            cv2.putText(frame, label, (10, 40 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            draw_landmarks(frame, latest_result.hand_landmarks[i])  # add this

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
recognizer.close()