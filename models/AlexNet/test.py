#coding=utf-8
import os
import numpy as np
import tensorflow as tf
import input_data
import model
import matplotlib.pyplot as plt
from PIL import Image

test_dir = './test_data/'
logs_train_dir = './trained/'
Labels = {
    '0': (0, 'huweijian'),
    '1': (1, 'huweijian'),
    '2': (2, 'huweijian'),
    '3': (3, 'huweijian'),
    '4': (4, 'huweijian'),
}

batch_size = 100

def num2class(n):
    x = Labels.items()
    for name, item in x:
        if n in item:
            return name

def get_one_image(img_dir):
    image = Image.open(img_dir)
    plt.imshow(image)
    image = image.resize([227, 227])
    image = np.array(image)
    return image

def evaluate_one_image(image_array):
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 5

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 227, 227, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[227, 227, 3])

        # you need to change the directories to yours.
        saver = tf.train.Saver()

        with tf.Session() as sess:
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            name = num2class(max_index)
            print('this is a %s with possibility %.6f' % (name, prediction[:, max_index]))


train, train_label = input_data.get_files(test_dir)
train_batch,train_label_batch=input_data.get_batches(train,
                                train_label,
                                227,
                                227,
                                batch_size,
                                5)
train_logits = model.inference(train_batch, batch_size, 5)
train__acc = model.evaluation(train_logits, train_label_batch)
#产生一个会话
sess = tf.Session()
#所有节点初始化
sess.run(tf.global_variables_initializer())
#队列监控
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
logs_train_dir = './trained/'
saver = tf.train.Saver()
#load model
print("Reading checkpoints...")
ckpt = tf.train.get_checkpoint_state(logs_train_dir)
if ckpt and ckpt.model_checkpoint_path:
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Loading success, global_step is %s' % global_step)
else:
    print('No checkpoint file found')
#test
try:
    #执行MAX_STEP步的训练，一步一个batch
    for step in np.arange(300):
        if coord.should_stop():
                break
        #启动以下操作节点
        tra_acc = sess.run(train__acc)
        print('Step %d, Every %d images, train accuracy = %.2f%%' %(step, batch_size, tra_acc))
except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    coord.request_stop()

# img = get_one_image('/home/ljx/Desktop/舰船识别/军舰2/0/denglujian9.jpg')
# evaluate_one_image(img)
