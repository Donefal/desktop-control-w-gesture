import os
import csv
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='models/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
)
recognizer = vision.GestureRecognizer.create_from_options(options)

dataset_path = "datasets"
output_csv = "train/landmarks.csv"

dataset_path = "datasets"

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    header = [f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]] + ["label"]
    writer.writerow(header)

    for mode in os.listdir(dataset_path):
        mode_dir = os.path.join(dataset_path, mode)
        if not os.path.isdir(mode_dir):
            continue

        for label in os.listdir(mode_dir):
            label_dir = os.path.join(mode_dir, label)
            if not os.path.isdir(label_dir):
                continue

            for img_file in os.listdir(label_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue

                img_path = os.path.join(label_dir, img_file)
                frame = cv2.imread(img_path)
                if frame is None:
                    print(f"Skipping unreadable file: {img_path}")
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = recognizer.recognize(mp_image)

                result = recognizer.recognize(mp_image)
                print(img_path)

                # Skip if no hand or more than one hand detected
                if not result.hand_landmarks or len(result.hand_landmarks) > 1:
                    print(f"Skipping (wrong hand count): {img_path}")
                    continue

                # Optionally filter by handedness — only keep left hand
                handedness = result.handedness[0][0].display_name
                if handedness != "Left":  # MediaPipe's perspective, so unmirrored
                    print(f"Skipping (wrong hand): {img_path}")
                    continue

                lm = result.hand_landmarks[0]
                row = [coord for point in lm for coord in [point.x, point.y, point.z]]
                row.append(label)
                writer.writerow(row)

print("Done — landmarks.csv created")
recognizer.close()