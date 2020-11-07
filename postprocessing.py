import os
from typing import Tuple, List

import cv2
import mediapipe as mp
import numpy as np

from pose import InFrame, where_in_frame
from segmentation import load_model, predict_person, blur_background
from utils import max_diff, smooth_curve

mp_pose = mp.solutions.pose


def render_video_vertical(input: str, output: str):
    cap = cv2.VideoCapture(input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rotate = width > height
    if rotate:
        width, height = height, width
    out = cv2.VideoWriter(
        output,
        cv2.VideoWriter_fourcc(*"XVID"),
        cap.get(cv2.CAP_PROP_FPS),
        (width, height),
    )
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        if rotate:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        out.write(image)
    out.release()
    cap.release()


def longest_stretch(arr: List) -> Tuple[int, int]:
    longest_start = 0
    longest_length = 0
    start = 0
    length = 0
    for i, x in enumerate(arr):
        if x:
            length += 1
        else:
            if length > longest_length:
                longest_start = start
                longest_length = length
            start = i + 1
            length = 0
    if length > longest_length:
        longest_start = start
        longest_length = length
    return longest_start, longest_start + longest_length


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


def find_extremes(x: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            np.array([0]),
            np.where((x[:-2] - x[1:-1]) * (x[1:-1] - x[2:]) < 0)[0] + 1,
            np.array([len(x) - 1]),
        )
    )


def find_largest_valley(x: np.ndarray, smoothing_window: int = 15) -> Tuple[int, int]:
    # Smooth to find sparse extremes
    x = smooth_curve(x, smoothing_window)
    extremes = find_extremes(x)
    gap = (x[extremes].max() + x[extremes].min()) * 0.5
    peaks = extremes[np.where(x[extremes] > gap)[0]]
    # This is the largest walley
    left = np.argmax(peaks[1:] - peaks[:-1])
    right = peaks[left + 1]
    left = peaks[left]
    # Remove other walleys:
    start = 0
    end = len(x)
    for i in range(right + 1, end):
        if x[i] < gap:
            end = i
            break
    for i in range(left - 1, start, -1):
        if x[i] < gap:
            start = i + 1
            break
    # Find matching peaks on both sides of the walley
    mathes = [
        (i, j, (x[i] + x[j]) * 0.5 - gap - abs(x[i] - x[j]) * 0.5)
        for i in range(start, left + 1)
        for j in range(right, end)
    ]
    am = np.argmax([x[2] for x in mathes])
    return mathes[am][0], mathes[am][1]


def find_in_frame(
    poses: List, horisontal_margin=0.05, vertical_margin=0.5, min_height=0.1,
) -> Tuple[int, int]:
    in_frame = [
        where_in_frame(p, horisontal_margin, vertical_margin, min_height) == InFrame.OK
        for p in poses
    ]
    return longest_stretch(in_frame)


def crop_regions(
    poses,
    smoothing: float = 3.0,
    horisontal_padding: float = 0.35,
    vertical_padding: float = 0.8,
    horisontal_marks: List = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_EAR,
        mp_pose.PoseLandmark.RIGHT_EAR,
    ],
    vertical_marks: List = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_EYE,
        mp_pose.PoseLandmark.RIGHT_EYE,
        mp_pose.PoseLandmark.LEFT_EAR,
        mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.MOUTH_LEFT,
        mp_pose.PoseLandmark.MOUTH_RIGHT,
    ],
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    # Find the horisontal center
    hcenter = np.array(
        [sum([p[l].x for l in horisontal_marks]) / len(horisontal_marks) for p in poses]
    )
    hwidth = max([max_diff([p[l].x for l in horisontal_marks]) for p in poses]) * (
        1 + horisontal_padding * 2
    )
    hcenter = np.maximum(np.minimum(hcenter, 1 - hwidth * 0.5), hwidth * 0.5)
    # Find the vertical center
    vcenter = np.array(
        [sum([p[l].y for l in vertical_marks]) / len(vertical_marks) for p in poses]
    )
    vheight = max([max_diff([p[l].y for l in vertical_marks]) for p in poses]) * (
        1 + vertical_padding * 2
    )
    vcenter = np.maximum(np.minimum(vcenter, 1 - vheight * 0.5), vheight * 0.5)
    # Smooth out the movements
    hcenter = smooth_curve(hcenter, smoothing * 4, smoothing)
    vcenter = smooth_curve(vcenter, smoothing * 4, smoothing)
    return (hcenter, hwidth, vcenter, vheight)


def process_video(
    input: str,
    output: str,
    start: int,
    crops: Tuple[np.ndarray, float, np.ndarray, float],
    blur_strength: float = 0.03,
):
    cap = cv2.VideoCapture(input)
    fwidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fheight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    iwidth = int(fwidth * crops[1])
    iheight = int(fheight * crops[3])
    out = cv2.VideoWriter(
        output,
        cv2.VideoWriter_fourcc(*"XVID"),
        cap.get(cv2.CAP_PROP_FPS),
        (iwidth, iheight),
    )
    model = load_model()
    for i in range(start):
        cap.read()
    last_mask = 1.0
    for x, y in zip(crops[0], crops[2]):
        success, image = cap.read()
        if not success:
            break
        x = int(fwidth * x) - iwidth // 2
        y = int(fheight * y) - iheight // 2
        image = image[y : y + iheight, x : x + iwidth]
        mask = predict_person(model, image)
        last_mask = last_mask * 0.5 + mask * 0.5
        image = blur_background(image, last_mask, blur_strength)
        out.write(image)
    cap.release()
    out.release()


def postprocess(input_video, output_video):
    print("Extracting poses")
    poses = extract_poses(input_video)
    print("Trimming poses")
    start, end = find_in_frame(poses)
    face_z = [
        p[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
        - p[mp_pose.PoseLandmark.LEFT_SHOULDER].x
        for p in poses[start:end]
    ]
    start2, end2 = find_largest_valley(face_z)
    start, end = start + max(0, start2 - 30 // 3), min(start + end2 + 30 // 3, end)
    crops = crop_regions(poses[start:end])
    print("Post-processing video")
    process_video(input_video, output_video, start, crops)


if __name__ == "__main__":
    if not os.path.exists("turn_around_full.avi"):
        print("Rotating video")
        render_video_vertical("turn_around_full.mp4", "turn_around_full.avi")
    postprocess("turn_around_full.avi", "output2.avi")
