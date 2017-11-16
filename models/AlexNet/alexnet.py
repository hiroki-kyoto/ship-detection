#-*- coding: utf8 -*-

import os
import numpy as np
import tensorflow as tf
import input_data

from PIL import Image
import matplotlib.pyplot as plt

class AlexNetLoader:
    '''
    AlexNet model loader
    '''
    def __init__(self, model_dir):
        self.N_CLASSES = 5
        self.IMG_W = 227
        self.IMG_H = 227
        self.BATCH_SIZE = 1
        self.CAPACITY = 200
        self.MAX_STEP = 10000
        self.learning_rate = 0.0001
        self.tmp_test_path = './tmp_recognize.jpg'

        self.weights = {}
        self.biases = {}
        self.im_fn_list = None
        self.label_list = None
        
        self.Labels = {
            '大型船坞登陆舰': (0, '护卫舰'),
            '导弹快艇': (1, '护卫舰'),
            '反潜舰': (2, '护卫舰'),
            '扫雷舰': (3, '护卫舰'),
            '鱼雷艇': (4, '护卫舰')
        }

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.train_logits = self.inference(
                        self.N_CLASSES
            )
            self.sess = tf.Session() 
            self.saver = tf.train.Saver()
        
            #print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(model_dir)
        
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                self.ready = True
            else:
                self.ready = False

    def num2class(self, n):
        x = self.Labels.items()
        for name, item in x:
            if n in item:
                return name

    def classify(self, im):
        im.save(self.tmp_test_path)
        with self.graph.as_default():
            prediction = self.sess.run(
                    self.fc3, 
                    feed_dict = {
                        self.im_path : self.tmp_test_path
                    }
            )
            max_index = np.argmax(prediction)
            name = self.num2class(max_index)

        return name 

    def inference(self, n_classes):
        with tf.variable_scope("weights"):
            self.weights = {
                'conv1': tf.get_variable(
                    'conv1', 
                    [11, 11, 3, 16], 
                    initializer = tf.truncated_normal_initializer(
                        stddev = 0.1, 
                        dtype = tf.float32
                    )
                ),
                'conv2': tf.get_variable(
                    'conv2', 
                    [5, 5, 16, 16], 
                    initializer = tf.truncated_normal_initializer(
                        stddev = 0.1, 
                        dtype = tf.float32
                    )
                ),
                'conv3': tf.get_variable(
                    'conv3', 
                    [3, 3, 16, 2], 
                    initializer = tf.truncated_normal_initializer(
                        stddev = 0.1, 
                        dtype = tf.float32
                    )
                ),
                'conv4': tf.get_variable(
                    'conv4', 
                    [3, 3, 2, 2], 
                    initializer = tf.truncated_normal_initializer(
                        stddev = 0.1, 
                        dtype = tf.float32
                    )
                ),
                'conv5': tf.get_variable(
                    'conv5', 
                    [3, 3, 2, 2], 
                    initializer = tf.truncated_normal_initializer(
                        stddev = 0.1, 
                        dtype = tf.float32
                    )
                ),
                'fc1': tf.get_variable(
                    'fc1', 
                    [6 * 6 * 2, 10], 
                    initializer = tf.truncated_normal_initializer(
                        stddev = 0.1, 
                        dtype = tf.float32
                    )
                ),
                'fc2': tf.get_variable(
                    'fc2', 
                    [10, 10], 
                    initializer=tf.truncated_normal_initializer(
                        stddev = 0.1, 
                        dtype = tf.float32
                    )
                ),
                'fc3': tf.get_variable(
                        'fc3', 
                        [10, n_classes], 
                        initializer = tf.truncated_normal_initializer(
                            stddev = 0.1, 
                            dtype = tf.float32
                        )
                    ),
            }
        
        with tf.variable_scope("biases"):
            self.biases = {
                'conv1': tf.get_variable(
                    'conv1', 
                    [16, ], 
                    initializer = tf.constant_initializer(
                        value=0.0, 
                        dtype=tf.float32
                    )
                ),
                'conv2': tf.get_variable(
                    'conv2', 
                    [16, ], 
                    initializer = tf.constant_initializer(
                        value = 0.0, 
                        dtype = tf.float32
                    )
                ),
                'conv3': tf.get_variable(
                    'conv3', 
                    [2, ], 
                    initializer = tf.constant_initializer(
                        value = 0.0, 
                        dtype = tf.float32
                    )
                ),
                'conv4': tf.get_variable(
                    'conv4', 
                    [2, ], 
                    initializer = tf.constant_initializer(
                        value = 0.0, 
                        dtype = tf.float32
                    )
                ),
                'conv5': tf.get_variable(
                    'conv5', 
                    [2, ], 
                    initializer = tf.constant_initializer(
                        value = 0.0, 
                        dtype = tf.float32
                    )
                ),
                'fc1': tf.get_variable(
                    'fc1', 
                    [10, ], 
                    initializer = tf.constant_initializer(
                        value = 0.0, 
                        dtype = tf.float32
                    )
                ),
                'fc2': tf.get_variable(
                    'fc2', 
                    [10, ], 
                    initializer = tf.constant_initializer(
                        value = 0.0, 
                        dtype = tf.float32
                    )
                ),
                'fc3': tf.get_variable(
                    'fc3', 
                    [n_classes, ], 
                    initializer = tf.constant_initializer(
                        value = 0.0, 
                        dtype = tf.float32
                    )
                )
            }

        # define graph (operations)
        # preprocessing graph
        self.im_path = tf.placeholder(dtype = tf.string)
        self.image_c = tf.read_file(self.im_path)
        self.image = tf.image.decode_jpeg(self.image_c, channels=3)
        self.image_resized = tf.image.resize_image_with_crop_or_pad(
                self.image, 
                self.IMG_W,
                self.IMG_H
        )
        self.image_std = tf.image.per_image_standardization(
                self.image_resized
        )
        self.image_input = tf.cast(
                self.image_std, 
                tf.float32
        )
        # network graph
        self.conv1 = tf.nn.bias_add(
                tf.nn.conv2d(
                    [self.image_input],
                    self.weights['conv1'],
                    strides = [1, 4, 4, 1], 
                    padding = 'VALID'
                ), 
                self.biases['conv1']
        )
        self.relu1 = tf.nn.relu(self.conv1)
        self.pool1 = tf.nn.max_pool(
                self.relu1, 
                ksize = [1, 3, 3, 1], 
                strides = [1, 2, 2, 1], 
                padding = 'VALID'
        )
        self.pool1 = tf.nn.lrn(
                self.pool1, 
                depth_radius = 4, 
                bias = 1.0, 
                alpha = 0.001 / 9.0, 
                beta = 0.75, 
                name = 'norm1'
        )
        self.conv2 = tf.nn.bias_add(
                tf.nn.conv2d(
                    self.pool1, 
                    self.weights['conv2'], 
                    strides = [1, 1, 1, 1], 
                    padding = 'SAME'
                ), 
                self.biases['conv2']
        )
        self.relu2 = tf.nn.relu(self.conv2)
        self.relu2 = tf.nn.lrn(
                self.relu2, 
                depth_radius = 4, 
                bias = 1.0, 
                alpha = 0.001 / 9.0, 
                beta = 0.75, 
                name = 'norm2'
        )
        self.pool2 = tf.nn.max_pool(
                self.relu2, 
                ksize = [1, 3, 3, 1], 
                strides = [1, 2, 2, 1], 
                padding = 'VALID'
        )
        self.conv3 = tf.nn.bias_add(
                tf.nn.conv2d(
                    self.pool2, 
                    self.weights['conv3'], 
                    strides = [1, 1, 1, 1], 
                    padding = 'SAME'
                ), 
                self.biases['conv3']
        )
        self.relu3 = tf.nn.relu(self.conv3)
        self.conv4 = tf.nn.bias_add(
                tf.nn.conv2d(
                    self.relu3, 
                    self.weights['conv4'], 
                    strides = [1, 1, 1, 1], 
                    padding = 'SAME'
                ), 
                self.biases['conv4']
        )
        self.conv5 = tf.nn.bias_add(
                tf.nn.conv2d(
                    self.conv4, 
                    self.weights['conv5'], 
                    strides = [1, 1, 1, 1], 
                    padding = 'SAME'
                ), 
                self.biases['conv5']
        )
        self.relu5 = tf.nn.relu(self.conv5)
        self.pool5 = tf.nn.max_pool(
                self.relu5, 
                ksize = [1, 3, 3, 1], 
                strides = [1, 2, 2, 1], 
                padding='VALID'
        )
        self.pool5 = tf.nn.lrn(
                self.pool5, 
                depth_radius = 4, 
                bias = 1.0, 
                alpha = 0.001 / 9.0, 
                beta = 0.75, 
                name = 'norm2'
        )
        self.flatten = tf.reshape(
                self.pool5, 
                [
                    -1, 
                    self.weights['fc1'].get_shape().as_list()[0]
                    ]
        )
        self.drop1 = tf.nn.dropout(self.flatten, 0.5)
        self.fc1 = tf.matmul(
                self.drop1, 
                self.weights['fc1']
                ) + self.biases['fc1']
        self.fc_relu1 = tf.nn.relu(self.fc1)
        self.fc2 = tf.matmul(
                self.fc_relu1, 
                self.weights['fc2']
                ) + self.biases['fc2']
        self.fc_relu2 = tf.nn.relu(self.fc2)
        self.fc3 = tf.matmul(
                self.fc_relu2, 
                self.weights['fc3']
                ) + self.biases['fc3']

    def free(self):
        with self.graph.as_default():
            self.sess.close()
