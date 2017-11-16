#-*- coding: utf8 -*-

import os
import numpy as np
import tensorflow as tf
import input_data
import model
from PIL import Image
import matplotlib.pyplot as plt

def recognize(image):

    N_CLASSES = 5
    IMG_W = 227
    IMG_H = 227
    BATCH_SIZE = 1
    CAPACITY = 200
    MAX_STEP = 10000
    learning_rate = 0.0001

    logs_train_dir = './trained/' 
    tmp_test_path = './tmp_recognize.jpg'

    image.save(tmp_test_path)
    
    Labels = {
        '军舰0': (0, '护卫舰'),
        '军舰1': (1, '护卫舰'),
        '军舰2': (2, '护卫舰'),
        '军舰3': (3, '护卫舰'),
        '军舰4': (4, '护卫舰'),
    }
    
    def num2class(n):
        x = Labels.items()
        for name, item in x:
            if n in item:
                return name
    
    labels_train = []
    images_train = []
    images_train.append(tmp_test_path)
    labels_train.append(0)
    temp = np.array([images_train, labels_train])
    temp = temp.transpose()
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    train_batch,train_label_batch = \
            input_data.get_batches(
                    image_list,
                    label_list,
                    IMG_W,
                    IMG_H,
                    BATCH_SIZE,
                    CAPACITY
            )

    train_logits = model.inference(
            train_batch, 
            BATCH_SIZE, 
            N_CLASSES
    )

    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(
            sess = sess, 
            coord = coord
    )
    
    saver = tf.train.Saver()
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(logs_train_dir)
    
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %s' % global_step)
    else:
        print('No checkpoint file found')
    
    prediction = sess.run(train_logits)
    max_index = np.argmax(prediction)
    name = num2class(max_index)
    
    return name

