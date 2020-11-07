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


def direction(pose, min_shoulder_dist: float = 0.1) -> Direction:
    if pose is None:
        return Direction.UNKNOWN
    left = pose[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right = pose[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    hdist = right.x - left.x
    if abs(hdist) < min_shoulder_dist:
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
    horisontal_padding: float = 0.3,
    vertical_padding: float = 1.0,
    min_height: float = 0.15,
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
    vertical_padding *= ymax - ymin
    horisontal_padding *= xmax - xmin
    if xmax > 1 - horisontal_padding * 1.2 and xmin < horisontal_padding * 1.2:
        return InFrame.CLOSE
    if ymax > 1 - vertical_padding * 1.2 and ymin < vertical_padding * 1.2:
        return InFrame.CLOSE
    if ymax - ymin < min_height:
        return InFrame.FAR
    if xmax > 1 - horisontal_padding:
        return InFrame.LEFT
    if xmin < horisontal_padding:
        return InFrame.RIGHT
    if ymin < vertical_padding:
        return InFrame.HIGH
    if ymax > 1 - vertical_padding:
        return InFrame.LOW
    return InFrame.OK


def move_instruction(
    pose,
    horisontal_padding: float = 0.35,
    vertical_padding: float = 1.0,
    min_height: float = 0.2,
) -> str:
    if pose is None:
        return "I can not see you!"
    if direction(pose) != Direction.TOWARD:
        return "Look at me!"
    frame = where_in_frame(pose, horisontal_padding, vertical_padding, min_height)
    if frame == InFrame.CLOSE:
        return "You are too close!"
    if frame == InFrame.FAR:
        return "Come closer!"
    if frame == InFrame.LEFT:
        return "You are too far to the left!"
    if frame == InFrame.RIGHT:
        return "You are too far to the right!"
    if frame == InFrame.HIGH:
        return "The phone is too low down!"
    if frame == InFrame.LOW:
        return "The phone is too high up!"
    return ""


class EarFrame(Enum):
    NOT = 0
    FORWARD = 1
    BACK = 2
    UP = 3
    DOWN = 4
    CLOSE = 5
    FAR = 6
    NOT_STRAIGHT = 7
    OK = 8


def ear_in_frame(
    pose, radius: float = 0.2, dist: float = 0.1, eye: float = 0.3, margin: float = 0.2,
) -> EarFrame:
    if pose is None:
        return EarFrame.NOT
    left_ear = pose[mp_pose.PoseLandmark.LEFT_EAR]
    right_ear = pose[mp_pose.PoseLandmark.RIGHT_EAR]
    if abs(left_ear.x - right_ear.x) > dist or abs(left_ear.y - right_ear.y) > dist:
        return EarFrame.NOT_STRAIGHT
    x = (left_ear.x + right_ear.x) * 0.5
    y = (left_ear.y + right_ear.y) * 0.5
    if y > 0.5 + radius:
        return EarFrame.DOWN
    if y < 0.5 - radius:
        return EarFrame.UP
    if x < 0.5 - radius:
        if left_ear.z > right_ear.z:
            return EarFrame.FORWARD
        else:
            return EarFrame.BACK
    if x > 0.5 + radius:
        if left_ear.z > right_ear.z:
            return EarFrame.BACK
        else:
            return EarFrame.FORWARD
    left_shoulder = pose[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    if (left_shoulder.y + right_shoulder.y) * 0.5 > 1.0 - margin:
        return EarFrame.CLOSE
    left_eye = pose[mp_pose.PoseLandmark.LEFT_EYE]
    right_eye = pose[mp_pose.PoseLandmark.RIGHT_EYE]
    if min(left_eye.x + right_eye.x) < margin:
        return EarFrame.CLOSE
    if max(left_eye.x + right_eye.x) > 1.0 - margin:
        return EarFrame.CLOSE
    if abs(x - (left_eye.x + right_eye.x) * 0.5) < eye:
        return EarFrame.FAR
    return EarFrame.OK


def ear_instruction(
    pose, radius: float = 0.2, dist: float = 0.1, eye: float = 0.3, margin: float = 0.2,
) -> str:
    if pose is None:
        return "I can not see your ear!"
    frame = ear_in_frame(pose, radius, dist, eye, margin)
    if frame == EarFrame.FORWARD:
        return "Move your phone forwards!"
    if frame == EarFrame.BACK:
        return "Move your phone backwards!"
    if frame == EarFrame.UP:
        return "Move your phone up!"
    if frame == EarFrame.DOWN:
        return "Move your phone down!"
    if frame == EarFrame.CLOSE:
        return "Move your phone further away!"
    if frame == EarFrame.FAR:
        return "Move your phone closer!"
    if frame == EarFrame.NOT_STRAIGHT:
        return "Look straight ahead!"
    return ""
