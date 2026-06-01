"""
Gesture action handlers.
Handles different control modes (navigation, key, presentation, STT).
"""

import pyautogui
import time
from enum import StrEnum, auto

from gesture_detection import (
    FINGERS, 
    get_pinch, 
    get_touching_fingers, 
    get_finger_extended,
    count_fingers
)
from control import move_mouse_relative, rel_reset

# -----------------------------------------------------------------------------------
# MODE: NAVIGATION
# -----------------------------------------------------------------------------------

class NAV_STATE(StrEnum):
    IDLE = auto()
    MOVING = auto()
    PINCHING = auto()

current_nav_state = NAV_STATE.IDLE
pinch_executed = False  # ensures click fires once per pinch
last_state_change = time.time()
STATE_DEBOUNCE = 0.2  # seconds before state can change again

def get_nav_state(right_landmarks):
    fingers = get_finger_extended(right_landmarks)
    touching = get_touching_fingers(right_landmarks)
    pinch = get_pinch(right_landmarks)

    if touching == {FINGERS.INDEX, FINGERS.MIDDLE}:
        return NAV_STATE.MOVING, None
    elif pinch is not None:
        return NAV_STATE.PINCHING, pinch
    else:
        return NAV_STATE.IDLE, None


def handle_navigation_action(right_landmarks):
    """
    Handle NAVIGATION mode actions:
    - Pinch index --> Left click
    - Pinch middle --> Right click
    - Pinch ring --> Zoom in (Ctrl + '+')
    - Pinch pinky --> Zoom out (Ctrl + '-')
    """
    global current_nav_state, pinch_executed, last_state_change

    new_state, pinch_finger = get_nav_state(right_landmarks)

    # Debounce — don't switch state too fast
    now = time.time()
    if new_state != current_nav_state:
        if now - last_state_change < STATE_DEBOUNCE:
            # Not enough time passed, stay in current state
            new_state = current_nav_state
        else:
            current_nav_state = new_state
            last_state_change = now
            pinch_executed = False  # reset on every state change

    # Execute based on committed state
    if current_nav_state == NAV_STATE.MOVING:
        rel_reset() if pinch_finger else move_mouse_relative(right_landmarks)

    elif current_nav_state == NAV_STATE.PINCHING and not pinch_executed:
        if pinch_finger == FINGERS.INDEX:
            print("Left Click")
            pyautogui.click()
        elif pinch_finger == FINGERS.MIDDLE:
            print("Right Click")
            pyautogui.rightClick()
        pinch_executed = True  # fire once per pinch

    elif current_nav_state == NAV_STATE.IDLE:
        rel_reset()


# -----------------------------------------------------------------------------------
# MODE: PRESENTATION
# -----------------------------------------------------------------------------------

def handle_presentation_action(right_landmarks):
    """
    Handle PRESENTATION mode actions:
    - Pinch index --> Left arrow key
    - Pinch middle --> Right arrow key
    """
    pinch_finger = get_pinch(right_landmarks)

    if pinch_finger == FINGERS.INDEX:
        print("Left Arrow")
        pyautogui.press('left')
    elif pinch_finger == FINGERS.MIDDLE:
        print("Right Arrow")
        pyautogui.press('right')


def handle_key_action(right_landmarks):
    """
    Handle KEY mode actions:
    - Pinch index --> Ctrl + C (Copy)
    - Pinch middle --> Ctrl + V (Paste)
    - Pinch ring --> Ctrl + X (Cut)
    - Touch index+middle --> Enter
    - Touch index+middle+ring --> Backspace
    - Touch all 4 fingers --> Esc
    """
    pinch_finger = get_pinch(right_landmarks)
    touch_finger = get_touching_fingers(right_landmarks)

    # Pinching actions
    if pinch_finger is not None:
        if pinch_finger == FINGERS.INDEX:
            print("Copy (ctrl c)")
            pyautogui.hotkey('ctrl', 'c')
        elif pinch_finger == FINGERS.MIDDLE:
            print("Paste (ctrl v)")
            pyautogui.hotkey('ctrl', 'v')
        elif pinch_finger == FINGERS.RING:
            print("Cut (ctrl x)")
            pyautogui.hotkey('ctrl', 'x')

    # Touching actions
    elif touch_finger is not None:
        if touch_finger == {FINGERS.INDEX, FINGERS.MIDDLE}:
            print("Enter")
            pyautogui.press('enter')
        elif touch_finger == {FINGERS.INDEX, FINGERS.MIDDLE, FINGERS.RING}:
            print("Backspace")
            pyautogui.press('backspace')
        elif touch_finger == {FINGERS.INDEX, FINGERS.MIDDLE, FINGERS.RING, FINGERS.PINKY}:
            print("ESC")
            pyautogui.press('esc')

def print_debug(right_landmarks, mode):
    if right_landmarks is None:
        return

    fingers = get_finger_extended(right_landmarks)
    count = count_fingers(fingers)
    pinch = get_pinch(right_landmarks)
    touching = get_touching_fingers(right_landmarks)

    # Debug print — shows everything about right hand state
    print(f"Mode: {mode} | Fingers: {fingers} | Count: {count} | Pinch: {pinch} | Touching: {touching}")
