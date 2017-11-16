# -*- coding:utf8 -*-
# resnet_test.py
# using well trained resnet tensorflow model
# from ./model/small/ to restore a model in 
# memory and run test with ship dataset

from resnet import *


def main(unused_args):
  # load ship data.
  ship = tf.contrib.learn.datasets.DATASETS['ship']()

  # labels:
  labels = [
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

  # Create a new resnet classifier.
  classifier = tf.estimator.Estimator(model_fn=res_net_model, model_dir='model/')

  tf.logging.set_verbosity(tf.logging.INFO)

  # Calculate accuracy.
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={X_FEATURE: ship.test.images},
      y=ship.test.labels.astype(np.int32),
      num_epochs=1,
      shuffle=False)
  scores = classifier.evaluate(input_fn=test_input_fn)
  print('Accuracy: {0:f}'.format(scores['accuracy']))

  #res = classifier.predict(input_fn=test_input_fn)
  #label_id = 0
  #prob = 0.0
  #for i in res:
  #    label_id = i['class']
  #    prob = i['prob'][label_id]
  #print labels[label_id], prob

if __name__ == '__main__':
  tf.app.run()
