# make test dataset

from make_dataset import *

if __name__ == "__main__":
    make_std_mnist_dataset('../raw_data_test/', '../test/')
