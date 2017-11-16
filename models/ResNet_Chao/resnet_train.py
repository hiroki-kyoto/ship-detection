# resnet_train.py
# training resnet with ship dataset
# training models are stored in ./model/
# training epoch number <= 1 million

from resnet import *

def main(unused_args):
    # load ship data:
    ship = tf.contrib.learn.datasets.DATASETS['ship']()

    # create a new resnet classifier
    classifier = tf.estimator.Estimator(model_fn=res_net_model, model_dir='model/')
    
    # turn on logging system
    tf.logging.set_verbosity(tf.logging.INFO)

    # training input function
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {X_FEATURE:ship.train.images},
            y = ship.train.labels.astype(np.int32),
            batch_size = 5,
            num_epochs = None, # not defined here
            shuffle = True
    )
    classifier.train(input_fn = train_input_fn, steps=1000000)
    print 'training done.'

if __name__ == '__main__':
    tf.app.run()
