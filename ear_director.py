from timeit import default_timer as timer

import cv2
import mediapipe as mp

from pose import ear_instruction, ear_in_frame, EarFrame
from postprocessing import postprocess_ear
from utils import Camera, TTS

mp_pose = mp.solutions.pose


def direct_to_spot(cam, tts, still_frames: int = 3, show: bool = False) -> bool:
    pose = mp_pose.Pose(False, 0.5, 0.5)
    first_iter = True
    counter = still_frames
    while True:
        cam.flush()
        success, image = cam.read()
        if not success:
            break
        if show:
            cam.show(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        ins = ear_instruction(
            results.pose_landmarks.landmark if results.pose_landmarks else None
        )
        if ins == "":
            if not first_iter and counter == still_frames:
                tts.say("Stay still!")
            still_frames -= 1
            if still_frames <= 0:
                break
            else:
                continue
        else:
            tts.say(ins)
        first_iter = False
        counter = still_frames
    pose.close()
    return first_iter


def intro_speech(tts, cam, left: bool = True, show: bool = False):
    if show:
        cam.flush()
        cam.show()
    tts.say("Hi and welcome to the guided ear scanning.")
    if show:
        cam.flush()
        cam.show()
    if left:
        tts.say(
            "Hold your phone in your left hand, pointing the camera towards your left ear."
        )
    else:
        tts.say(
            "Hold your phone in your right hand, pointing the camera towards your right ear."
        )
    if show:
        cam.flush()
        cam.show()


def instruction_speech(tts, cam, show: bool = False):
    if show:
        cam.flush()
        cam.show()
    tts.say("Lets show the camera different sides of your ear!")


def record_ear(
    cam, tts, output_file: str = "ear_raw.avi", duration: float = 10, show: bool = False
):
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*"XVID"),
        cam.get(cv2.CAP_PROP_FPS),
        (width, height),
    )
    tts.say("Slowly turn your head from side to side.")
    cam.flush()
    time = timer() + duration
    step = 0
    while cam.isOpened():
        success, image = cam.read()
        if not success:
            break
        if show:
            cam.show(image)
        out.write(image)
        if timer() > time:
            if step == 0:
                tts.say("Slowly tilt your head left and right.")
            elif step == 1:
                tts.say("Slowly roll your head around.")
            else:
                break
            step += 1
            time = timer() + duration
    tts.say("Stopping the recording!")
    out.release()
    if show:
        cam.flush()
        cam.show()


if __name__ == "__main__":
    SHOW = True
    LEFT = True
    tts = TTS()
    cam = Camera(True)
    cam.show()
    intro_speech(tts, cam, LEFT, SHOW)
    direct_to_spot(cam, tts, 3, SHOW)
    instruction_speech(tts, cam, SHOW)
    record_ear(cam, tts, "ear_raw.avi", 10, SHOW)
    cam.release()
    tts.say("Processing the video")
    postprocess_ear("ear_raw.avi", "ear_processed.avi")
    tts.say("Come look at the results")
    print(
        "The raw video is in 'ear_raw.avi' and the processed video is in 'ear_processed.avi'"
    )
