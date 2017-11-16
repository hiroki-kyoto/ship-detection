''' make standard mnist format dataset. '''

import os
from scan_file import ScanFile
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from struct import *

def make_std_mnist_dataset(in_dir, out_dir):
    ''' this function will turn images and
        labels from directory $in_dir
        into ubytes and save at $out_dir
        directory
    '''
    scaner = ScanFile(in_dir, postfix='.jpg')
    subdirs = scaner.scan_subdir()
    num_img = len(scaner.scan_files())
    num_class = len(subdirs)
    magic_code_images = 2051
    magic_code_labels = 2049
    num_row = 2*128
    num_col = 3*128
    num_chn = 3 # color channel
    num_byte = num_row * num_col * num_chn
    fn_images = 'images'
    fn_labels = 'labels'

    print 'classes found: ' + str(num_class)
    print 'images found: ' + str(num_img)

    f_images = open(os.path.join(out_dir, fn_images), 'wb')
    f_labels = open(os.path.join(out_dir, fn_labels), 'wb')

    f_images.write(pack('>iiii', magic_code_images, num_img, num_row, num_col))
    f_labels.write(pack('>ii', magic_code_labels, num_img))

    for cid in xrange(num_class): # cid : class id
        subdir = subdirs[cid]
        flag = 1
        print 'class#' + subdir.split('/')[-1]
        scaner = ScanFile(subdir, postfix='.jpg')
        images = scaner.scan_files()
        for i in images:
            im = Image.open(i)
            im = im.resize((num_col, num_row), Image.ANTIALIAS)
            data = np.array(im)
            # [height, width, channel]
            #plt.figure()
            #plt.imshow(data)
            #plt.show()
            data = data.reshape(1,1,num_byte)
            for k in xrange(num_byte):
                f_images.write(pack('>B', data[0,0,k]))
            f_labels.write(pack('>B', cid))

    f_images.close()
    f_labels.close()

