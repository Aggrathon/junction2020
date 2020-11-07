import time

import cv2
import mediapipe as mp
import pyttsx3

from pose import move_instruction

mp_pose = mp.solutions.pose


def get_tts():
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    voices = engine.getProperty("voices")
    # Prioritise certain voices
    voice_ids = (
        [
            p.id
            for p in engine.getProperty("voices")
            if p.name == "Microsoft Hazel Desktop - English (Great Britain)"
        ]
        + [
            p.id
            for p in engine.getProperty("voices")
            if p.name == "Microsoft Zira Desktop - English (United States)"
        ]
        + [p.id for p in engine.getProperty("voices") if p.name == "default"]
        + [p.id for p in engine.getProperty("voices") if p.name == "english"]
        + [p.id for p in engine.getProperty("voices") if p.name == "english-us"]
    )
    engine.setProperty("voice", voice_ids[0])
    return engine


def tts_say(tts: pyttsx3.engine.Engine, text: str):
    tts.say(text)
    tts.runAndWait()


def show_camera():
    cap = cv2.VideoCapture(0)
    _, img = cap.read()
    cap.release()
    cv2.imshow("Camera", img)
    cv2.waitKey(0)


def direct_to_spot(cap, tts, flush: int = 10) -> bool:
    pose = mp_pose.Pose(True, 0.5, 0.5)
    first_iter = True
    while True:
        for i in range(10):
            cap.grab()
        success, image = cap.read()
        if not success:
            break
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        if results.pose_landmarks:
            ins = move_instruction(results.pose_landmarks.landmark)
            if ins == "":
                if not first_iter:
                    tts_say(tts, "Stand still!")
                break
            else:
                tts_say(tts, ins)
        first_iter = False
    pose.close()
    return first_iter


if __name__ == "__main__":
    tts = get_tts()
    tts_say(tts, "Hi and welcome to the guided head scanning.")
    tts_say(
        tts,
        "Place your phone in an upright position at shoulder height with the camera pointing towards you.",
    )
    tts_say(tts, "Then take two steps back and follow my instructions.")
    time.sleep(3)
    cap = cv2.VideoCapture(0)
    direct_to_spot(cap, tts)
    cap.release()

