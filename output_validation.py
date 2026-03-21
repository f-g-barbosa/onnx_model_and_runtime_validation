import cv2
import numpy as np

image_path = fr".\data\test.jpg"


def load_image(img_path):
    loaded_image = cv2.imread(img_path)
    return loaded_image

def load_model_outputs():
    boxes = np.load(".\outputs\output_0.npy")
    scores = np.load(".\outputs\output_1.npy")
    classes = np.load(".\outputs\output_2.npy")
    num_detections = np.load(".\outputs\output_3.npy")
    return  boxes, scores, classes, num_detections


def remove_batch_dimension():
    boxes = boxes[0]
    scores = scores[0]
    classes = classes[0]
    num_detections = inteiro
    