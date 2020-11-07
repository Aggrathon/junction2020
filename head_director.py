import time

import cv2
import mediapipe as mp

from pose import move_instruction
from postprocessing import postprocess
from utils import camera_show, camera_flush, get_camera, get_tts, tts_say

mp_pose = mp.solutions.pose


def direct_to_spot(cap, tts, show: bool = False) -> bool:
    pose = mp_pose.Pose(True, 0.5, 0.5)
    first_iter = True
    while True:
        camera_flush(cap)
        success, image = cap.read()
        if not success:
            break
        if show:
            camera_show(cap, image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        if results.pose_landmarks:
            ins = move_instruction(results.pose_landmarks.landmark)
        else:
            ins = move_instruction(None)
        if ins == "":
            if not first_iter:
                tts_say(tts, "Stand still!")
            break
        else:
            tts_say(tts, ins)
        first_iter = False
    pose.close()
    return first_iter


def intro_speech(tts, cap, show: bool = False):
    tts_say(tts, "Hi and welcome to the guided head scanning.")
    if show:
        camera_flush(cap)
        camera_show(cap)
    tts_say(
        tts,
        "Place your phone in an upright position at shoulder height with the camera pointing towards you.",
    )
    if show:
        camera_flush(cap)
        camera_show(cap)
    tts_say(tts, "Then take two steps back and follow my instructions.")
    if show:
        camera_flush(cap)
        camera_show(cap)


def instruction_speech(tts, cap, show: bool = False):
    tts_say(tts, "Listen to the instructions, and do not move until I say begin!")
    if show:
        camera_flush(cap)
        camera_show(cap)
    tts_say(
        tts,
        "Without moving your head too much you will slowly, very slowly, turn around.",
    )
    if show:
        camera_flush(cap)
        camera_show(cap)
    tts_say(
        tts,
        "When you have turned for a full circle, walk somewhere where I cannot see you.",
    )
    if show:
        camera_flush(cap)
        camera_show(cap)


def record_turning(cap, tts, output_file: str = "head_raw.avi", show: bool = False):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*"XVID"),
        cap.get(cv2.CAP_PROP_FPS) * 0.5,
        (width, height),
    )
    pose = mp_pose.Pose(False, 0.5, 0.5)
    camera_flush(cap)
    tts_say(tts, "Begin!")
    while cap.isOpened():
        cap.grab()  # Skip each other frame due to processing delays
        success, image = cap.read()
        if not success:
            break
        if show:
            camera_show(cap, image)
        image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image2.flags.writeable = False
        results = pose.process(image2)
        if results.pose_landmarks is None:
            break
        out.write(image)
    tts_say(tts, "Stopping the recording!")
    out.release()
    pose.close()


if __name__ == "__main__":
    SHOW = True
    tts = get_tts()
    cap = get_camera()
    intro_speech(tts, cap, SHOW)
    direct_to_spot(cap, tts, SHOW)
    instruction_speech(tts, cap, SHOW)
    # Repeat directions if the subject has moved
    while not direct_to_spot(cap, tts, SHOW):
        instruction_speech(tts, cap, SHOW)
    record_turning(cap, tts, "head_raw.avi", SHOW)
    cap.release()
    tts_say(tts, "Processing the video")
    postprocess("head_raw.avi", "head_processed.avi")
    tts_say(tts, "Come look at the results")
    print(
        "The raw video is in 'head_raw.avi' and the processed video is in 'head_processed.avi'"
    )
