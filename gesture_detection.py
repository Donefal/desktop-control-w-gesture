"""
Gesture and finger detection module.
Handles hand landmark analysis and gesture recognition.
"""

from enum import StrEnum, auto

# -----------------------------------------------------------------------------------
# Constants & Enums
# -----------------------------------------------------------------------------------

PINCH_THRESHOLD = 0.05
TOUCH_THRESHOLD = 0.05


class MODE(StrEnum):
    NAVIGATION = auto()
    KEY = auto()
    PRESENTATION = auto()
    STT = auto()
    IDLE = auto()


class FINGERS(StrEnum):
    THUMB = auto()
    INDEX = auto()
    MIDDLE = auto()
    RING = auto()
    PINKY = auto()


# -----------------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------------

def dist(a, b):
    """Calculate distance between 2 points"""
    return ((a.x - b.x)**2 + (a.y - b.y)**2) ** 0.5


# -----------------------------------------------------------------------------------
# Finger detection functions
# -----------------------------------------------------------------------------------

def get_finger_extended(hand_landmarks):
    """
    Check if each finger is extended or not.
    Returns list: [thumb, index, middle, ring, pinky]
    """
    lm = hand_landmarks
    fingers = []

    # Four fingers: tip.y < knuckle.y means extended
    for tip, knuckle in [(8, 5), (12, 9), (16, 13), (20, 17)]:
        fingers.append(lm[tip].y < lm[knuckle].y)

    # Thumb: compare x axis instead
    thumb_extended = lm[4].x > lm[3].x  # flip if mirrored
    fingers.insert(0, thumb_extended)

    return fingers  # [thumb, index, middle, ring, pinky]


def count_fingers(fingers):
    """Count number of extended fingers (excluding thumb)"""
    fingers = fingers[1:]
    return sum(fingers)


def is_pointing(left_landmarks):
    """Check if only index finger is up, rest folded"""
    fingers = get_finger_extended(left_landmarks)
    return fingers == [False, True, False, False, False]


def get_pinch(hand_landmarks):
    """
    Return which finger is pinching with the thumb.
    Returns enum FINGERS value (except FINGERS.THUMB) or None
    """
    thumb_tip = hand_landmarks[4]

    finger_tips = {
        FINGERS.INDEX: hand_landmarks[8],
        FINGERS.MIDDLE: hand_landmarks[12],
        FINGERS.RING: hand_landmarks[16],
        FINGERS.PINKY: hand_landmarks[20],
    }

    for finger_name, finger_tip in finger_tips.items():
        dx = thumb_tip.x - finger_tip.x
        dy = thumb_tip.y - finger_tip.y
        distance = (dx**2 + dy**2) ** 0.5

        if distance < PINCH_THRESHOLD:
            return finger_name

    return None  # no pinch detected


def get_touching_fingers(hand_landmarks):
    """
    Return set of fingers that are touching each other.
    Returns set of FINGERS enum values (except FINGERS.THUMB)
    """
    finger_tips = {
        FINGERS.INDEX: hand_landmarks[8],
        FINGERS.MIDDLE: hand_landmarks[12],
        FINGERS.RING: hand_landmarks[16],
        FINGERS.PINKY: hand_landmarks[20],
    }

    touching = set()
    finger_names = list(finger_tips.keys())

    for i in range(len(finger_names)):
        for j in range(i + 1, len(finger_names)):
            name_a = finger_names[i]
            name_b = finger_names[j]
            if dist(finger_tips[name_a], finger_tips[name_b]) < TOUCH_THRESHOLD:
                touching.add(name_a)
                touching.add(name_b)

    return touching  # set of finger names that are touching each other


# -----------------------------------------------------------------------------------
# Mode detection
# -----------------------------------------------------------------------------------

def get_mode_from_left_hand(left_landmarks):
    """Detect mode from left hand gesture. Returns MODE enum"""
    if left_landmarks is None:
        return MODE.IDLE

    fingers = get_finger_extended(left_landmarks)
    count = count_fingers(fingers)

    if count == 1:
        return MODE.NAVIGATION
    elif count == 2:
        return MODE.KEY
    elif count == 3:
        return MODE.PRESENTATION
    elif count == 4:
        return MODE.STT
    return MODE.IDLE
