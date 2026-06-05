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

class PRES_STATE(StrEnum):
    IDLE = auto()
    PINCHING = auto()

current_pres_state = PRES_STATE.IDLE
pres_pinch_executed = False
last_pres_state_change = time.time()
PRES_DEBOUNCE = 0.3  # slightly longer than navigation feels better for slide control

def handle_presentation_action(right_landmarks):
    global current_pres_state, pres_pinch_executed, last_pres_state_change

    pinch_finger = get_pinch(right_landmarks)
    new_state = PRES_STATE.PINCHING if pinch_finger is not None else PRES_STATE.IDLE

    now = time.time()
    if new_state != current_pres_state:
        if now - last_pres_state_change < PRES_DEBOUNCE:
            new_state = current_pres_state  # not enough time, stay put
        else:
            current_pres_state = new_state
            last_pres_state_change = now
            pres_pinch_executed = False

    if current_pres_state == PRES_STATE.PINCHING and not pres_pinch_executed:
        if pinch_finger == FINGERS.INDEX:
            print("Left Arrow")
            pyautogui.press('left')
        elif pinch_finger == FINGERS.MIDDLE:
            print("Right Arrow")
            pyautogui.press('right')
        elif pinch_finger == FINGERS.RING:
            print("Shift + Ctrl + Left Arrow")
            pyautogui.hotkey('shift', 'ctrl', 'left')
        elif pinch_finger == FINGERS.PINKY:
            print("Shift + Ctrl + Right Arrow")
            pyautogui.hotkey('shift', 'ctrl', 'right')
        pres_pinch_executed = True


# -----------------------------------------------------------------------------------
# MODE: KEY
# -----------------------------------------------------------------------------------

class KEY_STATE(StrEnum):
    IDLE = auto()
    PINCHING = auto()
    TOUCHING = auto()

current_key_state = KEY_STATE.IDLE
key_action_executed = False
last_key_state_change = time.time()
KEY_DEBOUNCE = 0.25

def handle_key_action(right_landmarks):
    global current_key_state, key_action_executed, last_key_state_change

    pinch_finger = get_pinch(right_landmarks)
    touch_fingers = get_touching_fingers(right_landmarks)

    # Pinch takes priority over touch
    if pinch_finger is not None:
        new_state = KEY_STATE.PINCHING
    elif touch_fingers:
        new_state = KEY_STATE.TOUCHING
    else:
        new_state = KEY_STATE.IDLE

    now = time.time()
    if new_state != current_key_state:
        if now - last_key_state_change < KEY_DEBOUNCE:
            new_state = current_key_state
        else:
            current_key_state = new_state
            last_key_state_change = now
            key_action_executed = False

    if key_action_executed:
        return

    if current_key_state == KEY_STATE.PINCHING:
        if pinch_finger == FINGERS.INDEX:
            print("Copy (ctrl+c)")
            pyautogui.hotkey('ctrl', 'c')
        elif pinch_finger == FINGERS.MIDDLE:
            print("Paste (ctrl+v)")
            pyautogui.hotkey('ctrl', 'v')
        elif pinch_finger == FINGERS.RING:
            print("Cut (ctrl+x)")
            pyautogui.hotkey('ctrl', 'x')
        key_action_executed = True

    # TODO: Bagian ini masih kurang enakeun pakenya
    elif current_key_state == KEY_STATE.TOUCHING:
        if touch_fingers == {FINGERS.INDEX, FINGERS.MIDDLE}:
            print("Enter")
            pyautogui.press('enter')
        elif touch_fingers == {FINGERS.INDEX, FINGERS.MIDDLE, FINGERS.RING}:
            print("Backspace")
            pyautogui.press('backspace')
        elif touch_fingers == {FINGERS.INDEX, FINGERS.MIDDLE, FINGERS.RING, FINGERS.PINKY}:
            print("ESC")
            pyautogui.press('esc')
        key_action_executed = True

# -----------------------------------------------------------------------------------
# MODE: HELPER FUNCTION
# -----------------------------------------------------------------------------------
def print_debug(right_landmarks, mode):
    if right_landmarks is None:
        return

    fingers = get_finger_extended(right_landmarks)
    count = count_fingers(fingers)
    pinch = get_pinch(right_landmarks)
    touching = get_touching_fingers(right_landmarks)

    # Debug print — shows everything about right hand state
    print(f"Mode: {mode} | Fingers: {fingers} | Count: {count} | Pinch: {pinch} | Touching: {touching}")
