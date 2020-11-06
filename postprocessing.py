from typing import Tuple, List

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def longest_stretch(arr: np.ndarray) -> Tuple[int, int]:
    longest_start = 0
    longest_length = 0
    start = 0
    length = 0
    for i, x in enumerate(arr):
        if np.isnan(x):
            if length > longest_length:
                longest_start = start
                longest_length = length
            start = i + 1
            length = 0
        else:
            length += 1
    if length > longest_length:
        longest_start = start
        longest_length = length
    return longest_start, longest_length


def extract_poses(video: str) -> List:
    pose = mp_pose.Pose(False, 0.5, 0.5)
    cap = cv2.VideoCapture(video)
    poses = list()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        if results.pose_landmarks:
            poses.append(results.pose_landmarks.landmark)
        else:
            poses.append(None)

    cap.release()
    pose.close()
    return poses


def smooth_curve(x: np.ndarray, window: int = 30, sigma: float = 5) -> np.ndarray:
    if window % 2 == 0:
        window += 1
    kernel = np.exp(-((np.arange(window) - window // 2) ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return np.convolve(x, kernel, "same")


def find_extremes(x: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            np.array([0]),
            np.where((x[:-2] - x[1:-1]) * (x[1:-1] - x[2:]) < 0)[0] + 1,
            np.array([len(x) - 1]),
        )
    )


def find_largest_gap(x: np.ndarray) -> float:
    x = np.sort(x)
    diff = np.abs(x[1:] - x[:-1])
    am = np.argmax(diff)
    return (x[am] + x[am + 1]) * 0.5


def find_largest_valley(x: np.ndarray, smoothing_window: int = 15) -> Tuple[int, int]:
    # Trim nans
    start, end = longest_stretch(x)
    end += start
    # Smooth to find sparse extremes
    x = smooth_curve(x[start:end], smoothing_window)
    extremes = find_extremes(x)
    gap = find_largest_gap(x[extremes])
    peaks = np.where(x[extremes] > gap)[0]
    # Avoid "one-sided" walleys
    while len(peaks) == 1:
        extremes = np.delete(extremes, peaks)
        gap = find_largest_gap(x[extremes])
        peaks = np.where(x[extremes] > gap)[0]
    while len(peaks) == len(extremes) - 1:
        extremes = extremes[peaks]
        gap = find_largest_gap(x[extremes])
        peaks = np.where(x[extremes] > gap)[0]
    peaks = extremes[peaks]
    # This is the largest walley
    left = np.argmax(peaks[1:] - peaks[:-1])
    right = peaks[left + 1]
    left = peaks[left]
    # Remove other walleys:
    start2 = 0
    end2 = len(x)
    for i in range(right + 1, end2):
        if x[i] < gap:
            end2 = i
            break
    for i in range(left - 1, start2, -1):
        if x[i] < gap:
            start2 = i + 1
            break
    # Find matching peaks on both sides of the walley
    mathes = [
        (i, j, (x[i] + x[j]) * 0.5 - gap - abs(x[i] - x[j]) * 0.5)
        for i in range(start2, left + 1)
        for j in range(right, end2)
    ]
    am = np.argmax([x[2] for x in mathes])
    return start + mathes[am][0], start + mathes[am][1]


def trim_video(input: str, output: str, start: int, end: int, rotate: bool = True):
    cap = cv2.VideoCapture(input)
    size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    if rotate:
        size = (size[1], size[0])
    out = cv2.VideoWriter(
        output, cv2.VideoWriter_fourcc(*"XVID"), cap.get(cv2.CAP_PROP_FPS), size,
    )
    for i in range(start):
        cap.read()
    for i in range(start, end):
        success, image = cap.read()
        if not success:
            break
        if rotate:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        out.write(image)
    cap.release()
    out.release()


if __name__ == "__main__":
    poses = extract_poses("turn_around_full.mp4")
    face_z = [
        np.nan
        if p is None
        else p[mp_pose.PoseLandmark.LEFT_EYE].z + p[mp_pose.PoseLandmark.RIGHT_EYE].z
        for p in poses
    ]
    start, end = find_largest_valley(face_z)
    trim_video(
        "turn_around_full.mp4", "output2.avi", start - 30 // 5, end + 30 // 5, True,
    )
