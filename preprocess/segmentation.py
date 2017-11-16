import cv2
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import os

def seg(im):
    img_path = './tmp_seg.jpg'
    im.save(img_path)
    # load an original image
    im_cv = cv2.imread(img_path)
    
    # color value range
    cRange = 256
    
    rows,cols,channels = im_cv.shape
    
    # convert color space from bgr to gray
    img = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # otsu method
    threshold,imgOtsu = cv2.threshold(imgGray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # erode and dilation
    kernel = np.ones((5,5),np.uint8)
    imgOtsu = cv2.dilate(imgOtsu,kernel,iterations = 1)
    
    imgOtsu,contours, hierarchy = cv2.findContours(
            imgOtsu,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
    )
    # cv2.drawContours(img,contours,-1,(0,0,255),3)
    nobj = 0
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        area = w*h
        if area > 400 and w<cols/2 and h<rows/2:
            origin_pic = cv2.rectangle(
                    im_cv, 
                    (x, y), 
                    (x+w, y+h), 
                    (0, 0, 255), 
                    2
            )
            nobj += 1
    
    cv2.imwrite(img_path, im_cv)
    
    return (Image.open(img_path), nobj)

