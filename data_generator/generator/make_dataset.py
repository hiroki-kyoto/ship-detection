''' make standard mnist format dataset. '''

import os
import sys
from scan_file import ScanFile
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from struct import *
import subprocess
import random

def make_std_mnist_dataset(
        in_dir='../raw_data/', 
        out_dir='/tmp/ships/', 
        im_w=256,
        im_h=256,
        ratio=0.2
    ):
    ''' this function will turn images and
        labels from directory $in_dir
        into ubytes and save at $out_dir
        directory
    '''
    exts = ['.jpg', '.png']
    scaner = ScanFile(in_dir, postfix=exts)
    subdirs = scaner.scan_subdir()
    num_img = len(scaner.scan_files())
    num_class = len(subdirs)
    magic_code_images = 2051
    magic_code_labels = 2049
    num_row = im_h
    num_col = im_w
    num_chn = 3 # color channel
    num_byte = num_row * num_col * num_chn
    dir_train = 'train'
    dir_test = 'test'
    fn_images = 'images'
    fn_labels = 'labels'
    fn_desc = 'desc.txt'

    print 'classes found: ' + str(num_class)
    print 'images found: ' + str(num_img)

    # create dir if not exist, else remove it and recreate
    if os.path.isdir(out_dir):
        subprocess.call([
            'rm', 
            '-rf', 
            os.path.join(out_dir, '*')
        ])
    else:
        subprocess.call([
            'mkdir', 
            '-p', 
            out_dir
        ])
    # check if success
    if not os.path.isdir(out_dir):
        raise NameError(
                'Unable to create output dir:'+out_dir
        )
    # create sub-output dirs
    path_train = os.path.join(out_dir, dir_train)
    subprocess.call([
        'mkdir', 
        '-p', 
        path_train
    ])
    path_test = os.path.join(out_dir, dir_test)
    subprocess.call([
        'mkdir', 
        '-p', 
        path_test
    ])
    # create bin files for saving MNIST format data
    f_images_train = open(
            os.path.join(path_train, fn_images),
            'wb'
    )
    f_labels_train = open(
            os.path.join(path_train, fn_labels),
            'wb'
    )
    f_images_test = open(
            os.path.join(path_test, fn_images),
            'wb'
    )
    f_labels_test = open(
            os.path.join(path_test, fn_labels),
            'wb'
    )
    # create label description file:
    f_label_desc = open(
            os.path.join(out_dir, fn_desc),
            'wt'
    )
    # generate choice sequence
    test_num = int(ratio * num_img)
    train_num = num_img - test_num
    test_seq = random.sample(range(num_img), test_num)
    test_flags = np.zeros(num_img)

    for i in xrange(test_num):
        test_flags[test_seq[i]] = 1

    # saving data into files
    f_images_train.write(
            pack(
                '>iiii', 
                magic_code_images, 
                train_num, 
                num_row, 
                num_col
            )
    )
    f_labels_train.write(
            pack(
                '>ii', 
                magic_code_labels, 
                train_num
            )
    )
    f_images_test.write(
            pack(
                '>iiii',
                magic_code_images,
                test_num,
                num_row,
                num_col
            )
    )
    f_labels_test.write(
            pack(
                '>ii',
                magic_code_labels,
                test_num
            )
    )

    ctr = 0 # counter
    desc = '' # description
    for cid in xrange(num_class): # cid : class id
        subdir = subdirs[cid]
        flag = 1
        info = str(cid)+':'+subdir.split('/')[-1]+'\n'
        print info
        desc += info
        scaner = ScanFile(subdir, postfix=exts)
        images = scaner.scan_files()
        for i in images:
            print i
            im = Image.open(i)
            im = im.resize(
                    (
                        num_col, 
                        num_row
                    ), 
                    Image.ANTIALIAS
            )
            data = np.array(im)
            # [height, width, channel]
            #plt.figure()
            #plt.imshow(data)
            #plt.show()
            data = data.reshape(1, 1, num_byte)
            
            if test_flags[ctr] == 0: # for training
                for k in xrange(num_byte):
                    f_images_train.write(
                            pack('>B', data[0,0,k])
                    )
                f_labels_train.write(
                        pack('>B', cid)
                )
            else:
                for k in xrange(num_byte):
                    f_images_test.write(
                            pack('>B', data[0,0,k])
                    )
                f_labels_test.write(
                        pack('>B', cid)
                )
            ctr += 1 # counter increase

    f_label_desc.writelines(desc) # write description file
    
    f_images_train.close()
    f_labels_train.close()
    f_images_test.close()
    f_labels_test.close()
    f_label_desc.close()


def main():
    if len(sys.argv)==1:
        print 'WARNNING: using default configuration!'
        print 'default settings are:'
        print '\traw data:\t../raw_data/'
        print '\timage width:\t256'
        print '\timage height:\t256'
        print '\ttest/all:\t0.2'
        print '\toutput data:\t/tmp/ships/'
        make_std_mnist_dataset()
    elif len(sys.argv)==6:
        raw_path = sys.argv[1]
        im_width = sys.argv[2]
        im_height = sys.argv[3]
        im_ratio = sys.argv[4]
        out_path = sys.argv[5]
        make_std_mnist_dataset(
                raw_path, 
                out_path, 
                int(im_width), 
                int(im_height),
                float(im_ratio)
        )
    else:
        print 'help:'
        print '\t[arg#0] : path for raw data'
        print '\t[arg#1] : the width of image to resize'
        print '\t[arg#2] : the height of image to resize'
        print '\t[arg#3] : the ratio of test/all'
        print '\t[arg#4] : path for output data'
        return



main()

