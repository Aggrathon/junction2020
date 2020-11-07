import time

import cv2
import mediapipe as mp

from pose import move_instruction
from postprocessing import postprocess
from utils import Camera, TTS

mp_pose = mp.solutions.pose


def direct_to_spot(cam, tts, show: bool = False) -> bool:
    pose = mp_pose.Pose(True, 0.5, 0.5)
    first_iter = True
    while True:
        cam.flush()
        success, image = cam.read()
        if not success:
            break
        if show:
            cam.show(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.get_tts, tts_saywriteable = False
        results = pose.process(image)
        if results.pose_landmarks:
            ins = move_instruction(results.pose_landmarks.landmark)
        else:
            ins = move_instruction(None)
        if ins == "":
            if not first_iter:
                tts.say("Stand still!")
            break
        else:
            tts.say(ins)
        first_iter = False
    pose.close()
    return first_iter


def intro_speech(tts, cam, show: bool = False):
    tts.say("Hi and welcome to the guided head scanning.")
    if show:
        cam.flush()
        cam.show()
    tts.say(
        "Place your phone in an upright position at shoulder height with the camera pointing towards you.",
    )
    if show:
        cam.flush()
        cam.show()
    tts.say("Then take two steps back and follow my instructions.")
    if show:
        cam.flush()
        cam.show()


def instruction_speech(tts, cam, show: bool = False):
    tts.say("Listen to the instructions, and do not move until I say begin!")
    if show:
        cam.flush()
        cam.show()
    tts.say(
        "Without moving your head too much you will slowly, very slowly, turn around.",
    )
    if show:
        cam.flush()
        cam.show()
    tts.say(
        "When you have turned for a full circle, walk somewhere where I cannot see you.",
    )
    if show:
        cam.flush()
        cam.show()


def record_turning(cam, tts, output_file: str = "head_raw.avi", show: bool = False):
    width = int(cam.get(cv2.cam_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.cam_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*"XVID"),
        cam.get(cv2.cam_PROP_FPS) * 0.5,
        (width, height),
    )
    pose = mp_pose.Pose(False, 0.5, 0.5)
    cam.flush()
    tts.say("Begin!")
    while cam.isOpened():
        cam.grab()  # Skip each other frame due to processing delays
        success, image = cam.read()
        if not success:
            break
        if show:
            cam.show(image)
        image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image2.flags.writeable = False
        results = pose.process(image2)
        if results.pose_landmarks is None:
            break
        out.write(image)
    tts.say("Stopping the recording!")
    out.release()
    pose.close()


if __name__ == "__main__":
    SHOW = True
    tts = TTS()
    cam = Camera()
    intro_speech(tts, cam, SHOW)
    direct_to_spot(cam, tts, SHOW)
    instruction_speech(tts, cam, SHOW)
    # Repeat directions if the subject has moved
    while not direct_to_spot(cam, tts, SHOW):
        instruction_speech(tts, cam, SHOW)
    record_turning(cam, tts, "head_raw.avi", SHOW)
    cam.release()
    tts.say("Processing the video")
    postprocess("head_raw.avi", "head_processed.avi")
    tts.say("Come look at the results")
    print(
        "The raw video is in 'head_raw.avi' and the processed video is in 'head_processed.avi'"
    )
