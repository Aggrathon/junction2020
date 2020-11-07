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
    last_mask = 1.0
    while cam.isOpened():
        succ, image = cam.read()
        if not succ:
            break
        image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image2.flags.writeable = False
        results = pose.process(image2)
        mask = predict_person(mod, image)
        last_mask = np.minimum(1.0, last_mask * 0.8 + mask * 0.4)
        image = (image.astype(np.float32) * last_mask).astype(np.uint8)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
        cam.show(image)
    pose.close()
    cam.release()
