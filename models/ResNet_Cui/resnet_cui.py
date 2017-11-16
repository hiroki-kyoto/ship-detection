# -*- coding:utf8 -*-
# resnet_test.py
# using well trained resnet tensorflow model
# from ./model/small/ to restore a model in 
# memory and run test with ship dataset

from resnet import *
from PIL import Image

class ResNetLoader:
    def __init__(self, model_path):
        # labels:
        self.labels = [
                '登陆舰',
                '航母',
                '货船',
                '集装箱',
                '军舰1',
                '军舰2',
                '大型油轮',
                '小型油轮',
                '游艇',
                '渔船',
        ]
        self.map = [5,1,9,4,6,7,8,2,0,3]
        self.graph = tf.Graph()
        self.ready = False
        with self.graph.as_default():
            # Create a new resnet classifier.
            self.classifier = tf.estimator.Estimator(
                    model_fn = res_net_model,
                    model_dir = model_path
            )
            self.ready = True
        
        self.im_w = 64 # width
        self.im_h = 256 # height
        self.im_c = 3 # channel

    def classify(self, img):
        im = img.resize(
                [self.im_w, self.im_h], 
                Image.ANTIALIAS
        )
        im_data = np.array(im)
        im_data = np.reshape(
                im_data, 
                [1, self.im_h, self.im_w, self.im_c]
        )
        im_data = im_data.astype(np.float32)
        im_data = np.multiply(im_data, 1.0/255.0)
        
        with self.graph.as_default():
            test_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={X_FEATURE: im_data},
                    y=np.zeros([1]),
                    num_epochs=1,
                    shuffle=False
            )
            res = self.classifier.predict(input_fn = test_input_fn)
            label_id = 0
            prob = 0.0
            for i in res:
                label_id = i['class']
        return self.labels[self.map[label_id]]
    
    def free(self):
        # do nothing
        print 'freed'


