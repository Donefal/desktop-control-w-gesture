import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def draw_landmarks(image, hand_landmarks):
    h, w, _ = image.shape
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]
    for lm in hand_landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (px, py), 5, (0, 0, 255), -1)
    for start, end in connections:
        x1, y1 = int(hand_landmarks[start].x * w), int(hand_landmarks[start].y * h)
        x2, y2 = int(hand_landmarks[end].x * w), int(hand_landmarks[end].y * h)
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Label each landmark with its index
    for idx, lm in enumerate(hand_landmarks):
        px, py = int(lm.x * w), int(lm.y * h)
        cv2.putText(image, str(idx), (px + 5, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

base_options = python.BaseOptions(model_asset_path='models/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Change this to any image from your dataset
image_path = "datasets/nav/frame_0030.jpg"

frame = cv2.imread(image_path)
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
result = recognizer.recognize(mp_image)

if not result.hand_landmarks:
    print("No hand detected in this image")
else:
    for i, lm in enumerate(result.hand_landmarks):
        handedness = result.handedness[i][0].display_name
        print(f"Hand {i}: {handedness}")
        print(f"  Tip positions:")
        print(f"  Thumb  (4): x={lm[4].x:.3f} y={lm[4].y:.3f}")
        print(f"  Index  (8): x={lm[8].x:.3f} y={lm[8].y:.3f}")
        print(f"  Middle(12): x={lm[12].x:.3f} y={lm[12].y:.3f}")
        print(f"  Ring  (16): x={lm[16].x:.3f} y={lm[16].y:.3f}")
        print(f"  Pinky (20): x={lm[20].x:.3f} y={lm[20].y:.3f}")

        draw_landmarks(frame, lm)

cv2.imshow("Landmark Visualization", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
recognizer.close()