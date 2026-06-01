"""
Gesture action handlers.
Handles different control modes (navigation, key, presentation, STT).
"""

import pyautogui
from gesture_detection import (
    FINGERS, 
    get_pinch, 
    get_touching_fingers, 
    get_finger_extended,
    count_fingers
)

from control import move_mouse_relative, rel_reset


def handle_navigation_action(right_landmarks):
    """
    Handle NAVIGATION mode actions:
    - Pinch index --> Left click
    - Pinch middle --> Right click
    - Pinch ring --> Zoom in (Ctrl + '+')
    - Pinch pinky --> Zoom out (Ctrl + '-')
    """
    pinch_finger = get_pinch(right_landmarks)
    touching_fingers = get_touching_fingers(right_landmarks)

    if touching_fingers == {FINGERS.INDEX, FINGERS.MIDDLE}:
        print("Move mouse")
        move_mouse_relative(right_landmarks)
        return
    else:
         rel_reset()

    if pinch_finger == FINGERS.INDEX:
        print("Left Click")
        pyautogui.click()
    elif pinch_finger == FINGERS.MIDDLE:
        print("Right Click")
        pyautogui.rightClick()
    elif pinch_finger == FINGERS.RING:
        print("Zoom in (ctrl +)")
        pyautogui.hotkey('ctrl', '+')
    elif pinch_finger == FINGERS.PINKY:
        print("Zoom out (ctrl -)")
        pyautogui.hotkey('ctrl', '-')




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
