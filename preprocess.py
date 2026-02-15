import cv2
import numpy as np

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def extract_features(image):
    resized = cv2.resize(image, (64, 64))
    return resized.flatten()
