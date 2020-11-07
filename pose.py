from enum import Enum

import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose


class Direction(Enum):
    UNKNOWN = 0
    LEFT = 1
    AWAY = 2
    RIGHT = 3
    TOWARD = 4


def direction(pose) -> Direction:
    if pose is None:
        return Direction.UNKNOWN
    nose = pose[mp_pose.PoseLandmark.NOSE]
    left = pose[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right = pose[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    vdist = nose.y - (left.y + right.y) * 0.5
    hdist = right.x - left.x
    if abs(hdist) < vdist:
        if right.z > left.z:
            return Direction.LEFT
        else:
            return Direction.RIGHT
    if hdist < 0:
        return Direction.AWAY
    else:
        return Direction.TOWARD


class InFrame(Enum):
    NOT = 0
    LEFT = 1
    RIGHT = 2
    HIGH = 3
    LOW = 4
    CLOSE = 5
    FAR = 6
    OK = 7


def where_in_frame(
    pose,
    horisontal_margin: float = 0.15,
    vertical_margin: float = 0.3,
    min_height: float = 0.2,
) -> InFrame:
    if pose is None:
        return InFrame.NOT
    left_shoulder = pose[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_eye = pose[mp_pose.PoseLandmark.LEFT_EYE]
    right_eye = pose[mp_pose.PoseLandmark.RIGHT_EYE]
    xmin = min(left_shoulder.x, right_shoulder.x)
    xmax = max(left_shoulder.x, right_shoulder.x)
    ymin = min(left_eye.y, right_eye.y)
    ymax = max(left_shoulder.y, right_shoulder.y)
    if xmax > 1 - horisontal_margin and xmin < horisontal_margin:
        return InFrame.CLOSE
    if ymax > 1 - vertical_margin and ymin < vertical_margin:
        return InFrame.CLOSE
    if ymax - ymin < min_height:
        return InFrame.FAR
    if xmax > 1 - horisontal_margin:
        return InFrame.LEFT
    if xmin < horisontal_margin:
        return InFrame.RIGHT
    if ymin < vertical_margin:
        return InFrame.HIGH
    if ymax > 1 - vertical_margin:
        return InFrame.LOW
    return InFrame.OK


def move_instruction(
    pose,
    horisontal_margin: float = 0.15,
    vertical_margin: float = 0.3,
    min_height: float = 0.2,
) -> str:
    if pose is None:
        return "I can not see you!"
    if direction(pose) != Direction.TOWARD:
        return "Look at me!"
    frame = where_in_frame(pose, horisontal_margin, vertical_margin, min_height)
    if frame == InFrame.CLOSE:
        return "You are too close!"
    if frame == InFrame.FAR:
        return "You are too far away!"
    if frame == InFrame.LEFT:
        return "You are too far to the left!"
    if frame == InFrame.RIGHT:
        return "You are too far to the right!"
    if frame == InFrame.HIGH:
        return "The phone is too low down!"
    if frame == InFrame.LOW:
        return "The phone is too high up!"
    return ""
