import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim

#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image

import sys

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
import visualization

class SSDLoader:
    '''
    SSD Net model loader
    '''
    def __init__(self, model_fn):
        self.isess = []
        self.image_4d = []
        self.predictions = []
        self.localisations = []
        self.bbox_img = []
        self.img_input = []
        self.ssd_anchors = []
        self.ssd_net = []
        self.graph = tf.Graph()

        with self.graph.as_default():
            gpu_options = tf.GPUOptions(
                    allow_growth = True
            )
            config = tf.ConfigProto(
                    log_device_placement = False, 
                    gpu_options = gpu_options
            )
            self.isess = tf.InteractiveSession(
                    config = config
            )
        
            # Input placeholder
            net_shape = (300, 300)
            data_format = 'NHWC'
            self.img_input = tf.placeholder(
                    tf.uint8, 
                    shape=(None, None, 3)
            )
            
            # resize to SSD net shape.
            image_pre, labels_pre, bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
                    self.img_input, 
                    None, 
                    None, 
                    net_shape, 
                    data_format, 
                    resize = ssd_vgg_preprocessing.Resize.WARP_RESIZE
            )
            self.image_4d = tf.expand_dims(image_pre, 0) 
            
            # Define the SSD model.
            #reuse = True if 'ssd_net' in locals() else None
            reuse = tf.AUTO_REUSE
            self.ssd_net = ssd_vgg_300.SSDNet()
            
            with slim.arg_scope(
                    self.ssd_net.arg_scope(
                        data_format = data_format
                    )
            ):
                self.predictions, self.localisations, _, _ = self.ssd_net.net(
                        self.image_4d, 
                        is_training = False,
                        reuse = reuse
                )
            
            # SSD default anchor boxes.
            self.ssd_anchors = self.ssd_net.anchors(net_shape)

            # Restore SSD model.
            self.isess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(self.isess, model_fn)
            self.ready = True
            
    def detect_with_camera(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, img = cap.read()
            rclasses, rscores, rbboxes = self.process_image(img)
            visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
            cv2.imshow('detection result', img)
            if cv2.waitKey(1) & 0xFF == ord('p'):
                break
        cap.release()
        cv2.destroyAllWindows()
        
    def detect_with_image(self, im):
        # save the file as image
        im_fn = 'tmp_detect_with_image.jpg'
        im.save(im_fn)
        img = mpimg.imread(im_fn)
        rclasses, rscores, rbboxes = self.process_image(img)
        visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
        #visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
        result = Image.open('current detected result.jpg')
        
        objs = []
        
        for c in rclasses:
            objs.append(visualization.num2class(c))
        
        return (result, objs, rbboxes)

    # use SSD detector to get bounding boxs
    def process_image(
            self, 
            img, 
            select_threshold = 0.5, 
            nms_threshold = 0.45, 
            net_shape = (300, 300)
    ):
        # Run SSD network.
        with self.graph.as_default():
            rimg, rpredictions, rlocalisations, rbbox_img = self.isess.run(
                    [
                        self.image_4d, 
                        self.predictions, 
                        self.localisations, 
                        self.bbox_img
                    ],
                    feed_dict={ self.img_input: img }
            )
            # Get classes and bboxes from the net outputs.
            rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                    rpredictions, 
                    rlocalisations, 
                    self.ssd_anchors,
                    select_threshold = select_threshold, 
                    img_shape = net_shape, 
                    num_classes = 21, 
                    decode = True
            )
            
            rbboxes = np_methods.bboxes_clip(
                    rbbox_img, 
                    rbboxes
            )
            
            rclasses, rscores, rbboxes = np_methods.bboxes_sort(
                    rclasses, 
                    rscores, 
                    rbboxes, 
                    top_k = 400
            )
            
            rclasses, rscores, rbboxes = np_methods.bboxes_nms(
                    rclasses, 
                    rscores, 
                    rbboxes, 
                    nms_threshold = nms_threshold
            )
            
            # Resize bboxes to original image shape. Note: useless for Resize.WARP!
            rbboxes = np_methods.bboxes_resize(
                    rbbox_img, 
                    rbboxes
            )
            
            return rclasses, rscores, rbboxes
    
    def free(self):
        # release net resource
        with self.graph.as_default():
            self.isess.close()

