import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def enhance(im):
    img = np.array(im)
    r, g, b = cv2.split(img)
    er = cv2.equalizeHist(r)
    eg = cv2.equalizeHist(g)
    eb = cv2.equalizeHist(b)
    e_img = cv2.merge([er, eg, eb])
    return Image.fromarray(e_img)


