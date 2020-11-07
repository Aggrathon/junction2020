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


class Camera:
    def __init__(self, rotate: bool = False):
        self.cam = cv2.VideoCapture(0)
        self.rotate = rotate
        # Try set the maximum resolution
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 10_000)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 10_000)

    def flush(self, amount: int = 5):
        for _ in range(amount):
            self.cam.grab()

    def read(self):
        succ, img = self.cam.read()
        if self.rotate:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return succ, img

    def isOpened(self):
        return self.cam.isOpened()

    def release(self):
        self.cam.release()

    def get(self, prop):
        if self.rotate and prop == cv2.CAP_PROP_FRAME_WIDTH:
            prop = cv2.CAP_PROP_FRAME_HEIGHT
        elif self.rotate and prop == cv2.CAP_PROP_FRAME_HEIGHT:
            prop = cv2.CAP_PROP_FRAME_WIDTH
        return self.cam.get(prop)

    def show(self, image=None):
        if image is None:
            image = self.read()[1]
        cv2.imshow("Camera", image)
        cv2.waitKey(1)

    def grab(self) -> bool:
        return self.cam.grab()


class TTS:
    def __init__(self):
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
        self.engine = engine

    def say(self, text: str):
        self.engine.say(text)
        print("[TTS]:", text)
        self.engine.runAndWait()
