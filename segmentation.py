import os
import urllib.request

import cv2
import numpy as np
import tensorflow as tf


def load_model():
    file = "deeplabv3.tflite"
    if not os.path.exists(file):
        urllib.request.urlretrieve(
            "https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/default/1?lite-format=tflite",
            file,
        )
    mod = tf.lite.Interpreter(file)
    mod.allocate_tensors()
    # Labels:
    # 'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    # 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    # 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
    return mod


def predict_person(model, image):
    image2 = cv2.resize(image, (257, 257), interpolation=cv2.INTER_AREA)
    image2.shape = (1, 257, 257, 3)
    image2 = image2.astype(np.float32) / 255
    model.set_tensor(model.get_input_details()[0]["index"], image2)
    model.invoke()
    output_data = model.get_tensor(model.get_output_details()[0]["index"])
    # Not quite softmax (we don't want any blurring on 'person'):
    mask = np.exp(output_data[0, ..., 15]) / np.exp(output_data[0].max(-1))
    mask = cv2.resize(mask, image.shape[1::-1])
    mask.shape = mask.shape + (1,)
    return mask


def blur_background(image, mask, kernel: float = 0.05):
    image2 = image.astype(np.float32)
    kernel = np.sqrt(np.mean(image.shape[:-1]) * kernel)
    blurred = cv2.GaussianBlur(image, (0, 0), kernel)
    return (image * mask + blurred * (1 - mask)).astype(image.dtype)


if __name__ == "__main__":
    load_model()
