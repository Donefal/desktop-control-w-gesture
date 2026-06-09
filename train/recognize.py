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
    num_hands=2,
)
recognizer = vision.GestureRecognizer.create_from_options(options)

dataset_path = "datasets"
output_csv = "train/landmarks_two_hand.csv"

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)

    # 126 features: left hand (63) + right hand (63) + label
    left_header  = [f"L_{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]]
    right_header = [f"R_{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]]
    writer.writerow(left_header + right_header + ["label"])

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
                    print(f"Skipping unreadable: {img_path}")
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = recognizer.recognize(mp_image)

                # Must have exactly 2 hands
                if not result.hand_landmarks or len(result.hand_landmarks) != 2:
                    print(f"Skipping (need 2 hands): {img_path}")
                    continue

                # Separate into left and right
                left_lm = None
                right_lm = None
                for i, hand in enumerate(result.handedness):
                    h = hand[0].display_name
                    if h == "Left":
                        left_lm = result.hand_landmarks[i]
                    else:
                        right_lm = result.hand_landmarks[i]

                if left_lm is None or right_lm is None:
                    print(f"Skipping (missing one hand): {img_path}")
                    continue

                left_row  = [coord for point in left_lm  for coord in [point.x, point.y, point.z]]
                right_row = [coord for point in right_lm for coord in [point.x, point.y, point.z]]
                writer.writerow(left_row + right_row + [label])
                print(f"Saved: {img_path}")

print("Done — landmarks_two_hand.csv created")
recognizer.close()