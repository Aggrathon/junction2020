import cv2
import mediapipe as mp
import numpy as np

from utils import Camera
from segmentation import load_model, predict_person

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

if __name__ == "__main__":
    pose = mp_pose.Pose(False, 0.5, 0.5)
    cam = Camera(False)
    mod = load_model()

    while cam.isOpened():
        succ, image = cam.read()
        if not succ:
            break
        image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image2.flags.writeable = False
        results = pose.process(image2)
        mask = predict_person(mod, image)
        image2 = image.astype(np.float32)
        image2 = image2 * mask * 0.5 + mask * np.array([[[0.0, 0.0, 0.5]]])
        image = image2.astype(np.uint8)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
        cam.show(image)
    pose.close()
    cam.release()
