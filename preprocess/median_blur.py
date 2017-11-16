import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def median_blur(im):
    img = np.array(im)
    noise_reduce = cv2.medianBlur(img, 3)
    return Image.fromarray(noise_reduce)









