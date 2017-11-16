#coding=utf-8
import numpy as np
import tensorflow as tf
import os

# with tf.variable_scope("weights"):
#     weights = {
#
#         # 39*39*3->36*36*20->18*18*20
#
#         'conv1': tf.get_variable('conv1', [11, 11, 3, 96],
#                                  initializer=tf.contrib.layers.xavier_initializer_conv2d()),
#
#         # 18*18*20->16*16*40->8*8*40
#
#         'conv2': tf.get_variable('conv2', [5, 5, 96, 256],
#                                  initializer=tf.contrib.layers.xavier_initializer_conv2d()),
#
#         # 8*8*40->6*6*60->3*3*60
#
#         'conv3': tf.get_variable('conv3', [3, 3, 256, 384],
#                                  initializer=tf.contrib.layers.xavier_initializer_conv2d()),
#
#         # 3*3*60->120
#
#         'conv4': tf.get_variable('conv4', [3, 3, 384, 384],
#                                  initializer=tf.contrib.layers.xavier_initializer_conv2d()),
#
#         'conv5': tf.get_variable('conv5', [3, 3, 384, 256],
#                                  initializer=tf.contrib.layers.xavier_initializer_conv2d()),
#
#         'fc1': tf.get_variable('fc1', [6 * 6 * 256, 4096], initializer=tf.contrib.layers.xavier_initializer()),
#
#         'fc2': tf.get_variable('fc2', [4096, 4096], initializer=tf.contrib.layers.xavier_initializer()),
#
#         # 120->6
#
#         'fc3': tf.get_variable('fc3', [4096, 2], initializer=tf.contrib.layers.xavier_initializer()),
#
#     }
#
# with tf.variable_scope("biases"):
#     biases = {
#
#         'conv1': tf.get_variable('conv1', [96, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
#
#         'conv2': tf.get_variable('conv2', [256, ],
#                                  initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
#
#         'conv3': tf.get_variable('conv3', [384, ],
#                                  initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
#
#         'conv4': tf.get_variable('conv4', [384, ],
#                                  initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
#
#         'conv5': tf.get_variable('conv5', [256, ],
#                                  initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
#
#         'fc1': tf.get_variable('fc1', [4096, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
#
#         'fc2': tf.get_variable('fc2', [4096, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
#
#         'fc3': tf.get_variable('fc3', [2, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
#
#     }


def inference(images, batch_size, n_classes):
    with tf.variable_scope("weights"):
        weights = {

            # 39*39*3->36*36*20->18*18*20

            'conv1': tf.get_variable('conv1', [11, 11, 3, 96],
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d()),

            # 18*18*20->16*16*40->8*8*40

            'conv2': tf.get_variable('conv2', [5, 5, 96, 256],
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d()),

            # 8*8*40->6*6*60->3*3*60

            'conv3': tf.get_variable('conv3', [3, 3, 256, 384],
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d()),

            # 3*3*60->120

            'conv4': tf.get_variable('conv4', [3, 3, 384, 384],
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d()),

            'conv5': tf.get_variable('conv5', [3, 3, 384, 256],
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d()),

            'fc1': tf.get_variable('fc1', [6 * 6 * 256, 4096], initializer=tf.contrib.layers.xavier_initializer()),

            'fc2': tf.get_variable('fc2', [4096, 4096], initializer=tf.contrib.layers.xavier_initializer()),

            # 120->6

            'fc3': tf.get_variable('fc3', [4096, 5], initializer=tf.contrib.layers.xavier_initializer()),

        }

    with tf.variable_scope("biases"):
        biases = {

            'conv1': tf.get_variable('conv1', [96, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),

            'conv2': tf.get_variable('conv2', [256, ],
                                     initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),

            'conv3': tf.get_variable('conv3', [384, ],
                                     initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),

            'conv4': tf.get_variable('conv4', [384, ],
                                     initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),

            'conv5': tf.get_variable('conv5', [256, ],
                                     initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),

            'fc1': tf.get_variable('fc1', [4096, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),

            'fc2': tf.get_variable('fc2', [4096, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),

            'fc3': tf.get_variable('fc3', [5, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

        }
    # 第一层  定义卷积偏置和下采样
    images = tf.reshape(images, shape=[-1, 227, 227, 3])  # [batch, in_height, in_width, in_channels]
    images = (tf.cast(images, tf.float32) / 255. - 0.5) * 2  # 归一化处理
    conv1 = tf.nn.bias_add(tf.nn.conv2d(images, weights['conv1'], strides=[1, 4, 4, 1], padding='VALID'),

                           biases['conv1'])
    relu1 = tf.nn.relu(conv1)

    pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 第二层

    conv2 = tf.nn.bias_add(tf.nn.conv2d(pool1, weights['conv2'], strides=[1, 1, 1, 1], padding='SAME'),

                           biases['conv2'])

    relu2 = tf.nn.relu(conv2)

    pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 第三层

    conv3 = tf.nn.bias_add(tf.nn.conv2d(pool2, weights['conv3'], strides=[1, 1, 1, 1], padding='SAME'),

                           biases['conv3'])

    relu3 = tf.nn.relu(conv3)

    #  pool3=tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    conv4 = tf.nn.bias_add(tf.nn.conv2d(relu3, weights['conv4'], strides=[1, 1, 1, 1], padding='SAME'),

                           biases['conv4'])

    relu4 = tf.nn.relu(conv4)

    conv5 = tf.nn.bias_add(tf.nn.conv2d(relu4, weights['conv5'], strides=[1, 1, 1, 1], padding='SAME'),

                            biases['conv5'])

    relu5 = tf.nn.relu(conv5)

    pool5 = tf.nn.max_pool(relu5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 全连接层1，先把特征图转为向量

    flatten = tf.reshape(pool5, [-1, weights['fc1'].get_shape().as_list()[0]])

    drop1 = tf.nn.dropout(flatten, 0.5)

    fc1 = tf.matmul(drop1, weights['fc1']) + biases['fc1']

    fc_relu1 = tf.nn.relu(fc1)

    fc2 = tf.matmul(fc_relu1, weights['fc2']) + biases['fc2']

    fc_relu2 = tf.nn.relu(fc2)

    fc3 = tf.matmul(fc_relu2, weights['fc3']) + biases['fc3']

    return fc3

def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
            (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss

def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy
