import cv2
import mediapipe as mp

from pose import ear_instruction, ear_in_frame, EarFrame
from postprocessing import postprocess
from utils import Camera, TTS

mp_pose = mp.solutions.pose


def direct_to_spot(cam, tts, still_frames: int = 3, show: bool = False) -> bool:
    pose = mp_pose.Pose(True, 0.5, 0.5)
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


if __name__ == "__main__":
    SHOW = True
    tts = TTS()
    cam = Camera(True)
    cam.show()
    # intro_speech(tts, cam, SHOW)
    direct_to_spot(cam, tts, SHOW)
    # instruction_speech(tts, cam, SHOW)
    # Repeat directions if the subject has moved
    # while not direct_to_spot(cam, tts, SHOW):
    #     instruction_speech(tts, cam, SHOW)
    # record_turning(cam, tts, "head_raw.avi", SHOW)
    cam.release()
    # tts.say("Processing the video")
    # postprocess("head_raw.avi", "head_processed.avi")
    # tts.say("Come look at the results")
    # print(
    #     "The raw video is in 'head_raw.avi' and the processed video is in 'head_processed.avi'"
    # )
