from typing import List

import cv2
import numpy as np
import pyttsx3


def smooth_curve(x: np.ndarray, window: int = 30, sigma: float = 5) -> np.ndarray:
    if window % 2 == 0:
        window += 1
    kernel = np.exp(-((np.arange(window) - window // 2) ** 2) / (2 * sigma ** 2))
    return np.convolve(x, kernel, "same") / np.convolve(np.ones_like(x), kernel, "same")


def max_diff(x: List) -> float:
    return max(x) - min(x)


def get_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10_000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10_000)
    return cap


def get_tts():
    engine = pyttsx3.init()
    engine.setProperty("rate", 120)
    voices = engine.getProperty("voices")
    # Prioritise certain voices
    voice_ids = (
        [
            p.id
            for p in voices
            if p.name == "Microsoft Hazel Desktop - English (Great Britain)"
        ]
        + [
            p.id
            for p in voices
            if p.name == "Microsoft Zira Desktop - English (United States)"
        ]
        + [p.id for p in voices if p.name == "default"]
        + [p.id for p in voices if p.name == "english"]
        + [p.id for p in voices if p.name == "english-us"]
    )
    engine.setProperty("voice", voice_ids[0])
    return engine


def tts_say(tts: pyttsx3.engine.Engine, text: str):
    tts.say(text)
    print("[TTS]:", text)
    tts.runAndWait()


def camera_flush(cap, amount: int = 10):
    for _ in range(amount):
        cap.grab()


def camera_show(cap, image=None):
    if image is None:
        _, image = cap.read()
    cv2.imshow("Camera", image)
    cv2.waitKey(1)
