import whisper
import sounddevice as sd
import threading
import pyperclip
import pyautogui
import numpy as np

from gesture_detection import get_finger_extended

# -----------------------------------------------------------------------------------
# Moving mouse
# -----------------------------------------------------------------------------------

# Screen dimensions
SCREEN_W, SCREEN_H = pyautogui.size()

# Smoothing — higher = smoother but more laggy, lower = more responsive but jittery
SMOOTHING = 0.5

# Store previous position for smoothing
prev_x, prev_y = 0, 0

# Only use the center portion of the frame as the active area
MARGIN = 0.2  # 20% margin on each side

# PyautoGUI settings
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

def move_mouse_from_index(hand_landmarks, frame_shape):
    """ 
        TIDAK TERPAKAI (hanya untuk percobaan)
        Menggerakkan mouse 
    """
    global prev_x, prev_y

    tip = hand_landmarks[8]

    # Remap from active zone to 0-1 range
    x_norm = (tip.x - MARGIN) / (1 - 2 * MARGIN)
    y_norm = (tip.y - MARGIN) / (1 - 2 * MARGIN)

    # Clamp to valid range
    x_norm = max(0, min(1, x_norm))
    y_norm = max(0, min(1, y_norm))

    raw_x = x_norm * SCREEN_W
    raw_y = y_norm * SCREEN_H

    smooth_x = prev_x + SMOOTHING * (raw_x - prev_x)
    smooth_y = prev_y + SMOOTHING * (raw_y - prev_y)

    prev_x, prev_y = smooth_x, smooth_y

    pyautogui.moveTo(smooth_x, smooth_y)



prev_finger_x, prev_finger_y = None, None
SENSITIVITY = 3.0  # increase to move faster, decrease to slow down
DEAD_ZONE = 0.012  # ignore movements smaller than this

def move_mouse_relative(hand_landmarks):
    global prev_finger_x, prev_finger_y

    tip = hand_landmarks[8]

    if prev_finger_x is None:
        prev_finger_x, prev_finger_y = tip.x, tip.y
        return

    dx = (tip.x - prev_finger_x) * SENSITIVITY * SCREEN_W
    dy = (tip.y - prev_finger_y) * SENSITIVITY * SCREEN_H

    if abs(dx) < DEAD_ZONE * SCREEN_W and abs(dy) < DEAD_ZONE * SCREEN_H:
        return  # too small, ignore
    
    prev_finger_x, prev_finger_y = tip.x, tip.y

    pyautogui.moveRel(dx, dy)

def rel_reset():
    global prev_finger_x, prev_finger_y
    prev_finger_x, prev_finger_y = None, None

# -----------------------------------------------------------------------------------
# STT
# -----------------------------------------------------------------------------------
print(sd.query_devices())

# Load once at startup — not inside the function
whisper_model = whisper.load_model("tiny")

is_recording = False
stt_thread = None

def record_and_type():
    global is_recording
    sample_rate = 16000
    duration = 5  # seconds to record

    print("Listening...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32',
        device=1
    )
    sd.wait()  # blocks until recording is done

    audio_np = audio.flatten()
    print(f"Audio max amplitude: {audio_np.max():.4f}")
    result = whisper_model.transcribe(audio_np, language="id")  # change to "en" for English
    text = result["text"].strip()

    print(f"Recognized: {text}")

    if text:
        pyperclip.copy(text)
        pyautogui.hotkey('ctrl', 'v')

    is_recording = False


def handle_stt(right_landmarks):
    global is_recording, stt_thread
    fingers = get_finger_extended(right_landmarks)

    # Open palm triggers recording
    if all(fingers) and not is_recording:
        is_recording = True
        stt_thread = threading.Thread(target=record_and_type, daemon=True)
        stt_thread.start()