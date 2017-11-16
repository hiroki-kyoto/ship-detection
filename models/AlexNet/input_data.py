#coding=utf-8
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

def get_files(filename):
    images_train = []
    labels_train = []
    for _class_ in os.listdir(filename):
        class_path = filename+_class_
        for image in os.listdir(class_path):
            image_path = class_path+'/'+image
            images_train.append(image_path)
            labels_train.append(_class_)

    temp = np.array([images_train, labels_train])
    temp = temp.transpose()
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in label_list]
    return image_list, label_list

def get_batches(image,label,resize_w,resize_h,batch_size,capacity):
    # convert the list of images and labels to tensor
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int64)
    queue = tf.train.slice_input_producer([image, label])
    label = queue[1]
    image_c = tf.read_file(queue[0])
    image = tf.image.decode_jpeg(image_c, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, resize_w, resize_h)
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image,label], batch_size = batch_size,
                                              num_threads = 64, capacity = capacity)
    image_batch = tf.cast(image_batch, tf.float32)
    labels_batch = tf.reshape(label_batch, [batch_size])
    return image_batch, labels_batch

# # test
# BATCH_SIZE = 2
# CAPACITY = 256
# IMG_W = 208
# IMG_H = 208
#
# train_dir = '/home/ljx/Desktop/军舰2/'
# image_list, label_list = get_files(train_dir)
# image_batch, label_batch = get_batches(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
# with tf.Session() as sess:
#     i = 0
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     try:
#         while not coord.should_stop() and i<2:
#
#             img, label = sess.run([image_batch, label_batch])
#
#             # just test one batch
#             for j in np.arange(BATCH_SIZE):
#                 print('label: %d' %label[j])
#                 plt.imshow(img[j,:,:,:])
#                 plt.show()
#             i+=1
#
#     except tf.errors.OutOfRangeError:
#         print('done!')
#     finally:
#         coord.request_stop()
#     coord.join(threads)