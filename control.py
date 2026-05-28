import pyautogui
import numpy as np

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

