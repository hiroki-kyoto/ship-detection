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
                '中国-补给舰',
                '中国-6607型鞍山级-驱逐舰',
                '中国-051B-051C型-驱逐舰',
                '中国-054A-护卫舰',
                '中国-中型登陆舰-登陆舰',
                '中国-056-护卫舰',
                '中国-054-护卫舰',
                '中国-053K-护卫舰',
                '中国-65型-护卫舰',
                '中国-现代级-驱逐舰',
                '中国-051型-驱逐舰',
                '中国-053H-护卫舰',
                '中国-反潜舰',
                '中国-053H1G型-护卫舰',
                '中国-扫雷舰',
                '中国-导弹艇',
                '中国-053H1Q-护卫舰',
                '中国-6601型-护卫舰',
                '中国-气垫船-登陆舰',
                '中国-053H3-护卫舰',
                '中国-053H2-护卫舰',
                '中国-053H1-护卫舰',
                '中国-鱼雷艇',
                '中国-052型-驱逐舰',
                '中国-小型登陆舰-登陆舰',
                '中国-大型登陆舰-登陆舰',
                '中国-071型大型船坞登陆舰-登陆舰',
                '中国-053H2G-护卫舰'
        ]
        self.graph = tf.Graph()
        self.ready = False
        with self.graph.as_default():
            # Create a new resnet classifier.
            self.classifier = tf.estimator.Estimator(
                    model_fn = res_net_model,
                    model_dir = model_path
            )
            self.ready = True
        
        self.im_w = 48 # width
        self.im_h = 32 # height
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
        return self.labels[label_id]
    
    def free(self):
        # do nothing
        print 'freed'


