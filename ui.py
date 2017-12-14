# -*- coding:utf8 -*-
# Python 2.7+ and Tkinter 2.0+ and PIL 1.0+

import Tkinter as tk
import tkFileDialog as fd
import tkMessageBox as msg
import ttk
import io
from PIL import Image, ImageTk
import time
import sys
import os

# SSD
from models.SSD import ssd

# alexnet
from models.AlexNet import alexnet
from models.ResNet_Chao import resnet_chao
from models.ResNet_Cui import resnet_cui

# preprocessing
from preprocess.enhance import *
from preprocess.median_blur import *
from preprocess.dehaze import *
from preprocess.segmentation import *

def resize_with_ratio_held(im, w, h):
    ''' resize an image to at maxium [wxh],
    however the ratio of original one will
    be held the same.
    '''
    ow, oh = im.size
    ow = float(ow)
    oh = float(oh)

    # align with axis x:
    if oh*w/ow > h:
        return im.resize([int(ow*h/oh), h], Image.ANTIALIAS)
    else:
        return im.resize([w, int(oh*w/ow)], Image.ANTIALIAS)

def make_detect_report(objs):
    '''
    make a detection report with object list
    '''
    d = dict()
    for o in objs:
        if o not in d:
            d[o] = 1
        else:
            d[o] += 1
    
    report = '发现 '

    for i in d:
        report += ('%s 数量: %d ; ') % (i, d[i])
    return report


def _main_():
    root = tk.Tk()
    root.title('舰船检测识别系统')
    rw = 1024
    rh = 768
    root.geometry(str(rw)+'x'+str(rh))
    root.resizable(width=True,height=True)

    # using grid to place panels
    w = 800
    h = 600 

    # initialization for global variables
    _bg_im = None # background image on display as tKPhoto
    _im_fn = None # selected image file name
    _res_im = None # processed image
    _crop_ims = None # images cropped from original image
    _crop_photos = None # tkPhoto objects for display
    _bboxes = None # bounding boxes
    _detector_loader = None # detector loader
    _classifier_loader = None # classifier loader
    _objs = None # the id for each detected object

    # canvas
    canvas = tk.Canvas(
            root,
            width = w,
            height = h,
            bg = 'black'
    )
    canvas.grid(
            row = 0,
            column = 0,
            rowspan = 16,
            columnspan = 18
    )

    # status bar
    status_bar = tk.Label(
            root,
            height = 4,
            text = '状态信息显示...'
    )
    status_bar.grid(
            row = 16,
            column = 0,
            rowspan = 2,
            columnspan = 18
    )

    def reset_display():
        canvas.create_rectangle(0,0,w,h,fill='black')

    def change_display(flag='full'):
        global _res_im
        global _bg_im
        global _crop_ims
        global _crop_photos
        global _bboxes
        global _objs

        reset_display()

        if flag=='full':
            im = resize_with_ratio_held(_res_im, w, h)
            _bg_im = ImageTk.PhotoImage(im)
            canvas.create_image(w/2,h/2,image=_bg_im)
        elif flag=='half':
            im = resize_with_ratio_held(_res_im, w, h/2)
            _bg_im = ImageTk.PhotoImage(im)
            canvas.create_image(w/2, h/4, image=_bg_im)
            # plot all cropped images on the below
            _crop_photos = []
            _n = len(_crop_ims)
            _w,_h = im.size
            _im = None # temp resized cropped image

            xmargin = 8 # margin on axis x for each cropped image
            ymargin = 32 # margin on axis y

            for _i in xrange(_n):
                _im = resize_with_ratio_held(
                        _crop_ims[_i],
                        int(1.0*w/_n) - xmargin,
                        int(1.0*h/2) - ymargin
                )
                _crop_photos.append(ImageTk.PhotoImage(_im))

                ox = int(1.0*w*(2*_i+1)/_n/2)
                oy = int(1.0*h*3/4)
                cx, cy = _im.size
                
                canvas.create_image(
                        ox,
                        oy,
                        image=_crop_photos[-1]
                )
                
                _x = int((_w*_bboxes[_i,1]+_w*_bboxes[_i,3])/2)
                _y = int((_h*_bboxes[_i,0]+_h*_bboxes[_i,2])/2)
                
                canvas.create_line(
                        _x+w/2-_w/2, _y+h/4-_h/2,
                        ox, oy-cy/2,
                        fill='red',
                        arrow=tk.LAST,
                        smooth=True
                )
                text_margin = 6
                canvas.create_text(
                        ox,
                        oy + cy/2 + text_margin,
                        fill = 'white',
                        text = _objs[_i]
                )
        else:
            raise NameError('unacceptable args:' + str(flag))
        
        root.update()

    # task bar
    # select
    def select_file():
        global _im_fn
        global _res_im

        _im_fn = fd.askopenfilename(
                title='选择图像文件',
                filetypes=[
                    ('images','*.jpg *.png'),
                    ('All Files','*')
                ]
        )

        if type(_im_fn)!=type('Latin') and \
                type(_im_fn)!=type(unicode('中文','utf8')) or \
                not os.path.isfile(_im_fn):
            return

        _res_im = Image.open(_im_fn)
        change_display()
        status_bar.config(text='图片成功加载！')
        root.update()

    select_file_btn = tk.Button(
            root,
            text='选择图片',
            height = 3,
            command = select_file
    )

    select_file_btn.grid(
            row = 18,
            column = 0,
            rowspan = 2,
            columnspan = 2
    )

    # sorting the bounding boxes
    # from left to right
    # from up to down
    def bbox_ahead_of(a, b):
        if a[1]+a[3]<b[1]+b[3]: # a is left
            return True
        elif a[1]+a[3]==b[1]+b[3] and a[0]+a[2]<b[0]+b[2]:
            return True
        else:
            return False
    
    def bbox_swap(i, j): # corresponding object names too
        global _bboxes
        global _objs
        _c = range(4)
        _c[:] = _bboxes[i,:]
        _bboxes[i,:] = _bboxes[j,:]
        _bboxes[j,:] = _c[:]
        _str = _objs[i]
        _objs[i] = _objs[j]
        _objs[j] = _str

    def bbox_sort():
        global _bboxes
        _n = len(_bboxes)

        for _i in xrange(_n):
            for _j in xrange(_i+1,_n):
                if bbox_ahead_of(_bboxes[_j],_bboxes[_i]):
                    bbox_swap(_i, _j)
            # inner selection loop over
        # outer selection loop over
        #print _bboxes

    # detect
    def detect():
        global _detector_loader
        global _res_im
        global _crop_ims
        global _bboxes
        global _objs

        if '_res_im' not in globals():
            msg.showerror('操作流程错误','请先导入要检测的图片!')
            return
        if '_detector_loader' not in globals() or _detector_loader.ready == False:
            msg.showerror('操作流程错误','请先导入检测模型!')
            return
        status_bar.config(text='正在检测...')
        # before chaning res_im, back it up for other uses
        _im_org = _res_im
        (_res_im, _objs, _bboxes) = _detector_loader.detect_with_image(_res_im)
        _crop_ims = []
        _w,_h = _res_im.size

        # sort the bboxes
        bbox_sort()

        for _box in _bboxes:
            x1,x2 = int(_w*_box[1]),int(_w*_box[3])
            y1,y2 = int(_h*_box[0]),int(_h*_box[2])
            _crop_ims.append(_im_org.crop((x1,y1,x2,y2)))

        change_display('half')
        status_bar.config(text='检测完成:' + make_detect_report(_objs))
        root.update()

    ship_detect_btn = tk.Button(
            root,
            text = '检测',
            height = 3,
            command = detect
    )

    ship_detect_btn.grid(
            row = 18,
            column = 2,
            rowspan = 2,
            columnspan = 1
    )
    
    # recognize
    def recognize():
        global _res_im
        global _classifier_loader
        
        if '_res_im' not in globals():
            msg.showerror('操作流程错误','请先导入要识别的图片!')
            return
        if '_classifier_loader' not in globals() or _classifier_loader.ready == False:
            msg.showerror('操作流程错误','请先导入识别模型!')
            return

        status_bar.config(text='正在识别...')
        obj = _classifier_loader.classify(_res_im)
        status_bar.config(text='识别结果： ' + obj )
        root.update()
        
    ship_recognize_btn = tk.Button(
            root,
            text = '识别',
            height = 3,
            command = recognize
    )

    ship_recognize_btn.grid(
            row = 18,
            column = 3,
            rowspan = 2,
            columnspan = 1
    )

    # load detector model
    def load_detector_model():
        global _detector_loader
        if detector_model_chooser.get() == '':
            msg.showerror('配置错误','请先设置要导入哪种检测模型！')
        elif detector_model_chooser.get()=='SSD-Leng':
            path = fd.askdirectory()
            # load SSD model
            if (type(path)==type('Latin') or type(path)==type(unicode('中文', 'utf8'))) and os.path.isdir(path):
                info = '当前使用检测模型为:' + detector_model_chooser.get()
                status_bar.config(text = info + ', 正在读取检测模型...')
                root.update()

                if '_detector_loader' in globals() and _detector_loader.ready:
                    _detector_loader.free()
                _detector_loader = ssd.SSDLoader(path)
                if _detector_loader.ready:
                    status_bar.config(text = '检测模型导入成功！')
                else:
                    status_bar.config(text = '检测模型导入失败！')
                root.update()

    load_detect_model_btn = tk.Button(
            root,
            text = '导入检测模型',
            height = 3,
            command = load_detector_model
    )

    load_detect_model_btn.grid(
            row = 18,
            column = 4,
            rowspan = 2,
            columnspan = 3
    )

    # load recognization model
    def load_classifier_model():
        global _classifier_loader
        
        if classifier_model_chooser.get() == '':
            msg.showerror('配置错误','请先设置要导入哪种识别模型！')
        else:
            path = fd.askdirectory()
            # load AlexNet model
            if (type(path)==type('Latin') or \
                    type(path)==type(unicode('中文','utf8'))) and \
                    os.path.isdir(path):
                info = '当前使用识别模型为:' + classifier_model_chooser.get()
                status_bar.config(text = info + ', 正在读取识别模型...')
                root.update()

                if '_classifier_loader' in globals() and \
                        _classifier_loader.ready:
                    _classifier_loader.free()
                if classifier_model_chooser.get()=='AlexNet-Leng':
                    _classifier_loader = alexnet.AlexNetLoader(path)
                elif classifier_model_chooser.get()=='ResNet-Chao':
                    _classifier_loader = resnet_chao.ResNetLoader(path)
                elif classifier_model_chooser.get()=='ResNet-Cui':
                    _classifier_loader = resnet_cui.ResNetLoader(path)
                else:
                    _classifier_loader = None
                if _classifier_loader.ready:
                    status_bar.config(text = '识别模型导入成功！')
                else:
                    status_bar.config(text = '识别模型导入失败！')
                root.update()

    load_classifier_model_btn = tk.Button(
            root,
            text = '导入识别模型',
            height = 3,
            command = load_classifier_model
    )

    load_classifier_model_btn.grid(
            row = 18,
            column = 7,
            rowspan = 2,
            columnspan = 3
    )

    # save image button
    def save_image():
        global _res_im
        fn = fd.asksaveasfilename(
                initialdir='./',
                title='保存处理后的图片'
        )
        if type(fn) not in [type('Latin'), type(unicode('中文','utf8'))]:
            return
        _res_im.save(fn)
        status_bar.config(text='图像已保存在：' + fn)
        root.update()
        
    save_image_btn = tk.Button(
            root,
            text = '保存图片',
            height = 3,
            command = save_image
    )

    save_image_btn.grid(
            row = 18,
            column = 10,
            rowspan = 2,
            columnspan = 2
    )
    
    def camera_detection():
        global _detector_loader
        if '_detector_loader' not in globals():
            msg.showerror('操作流程错误','请先导入检测模型')
            return
        _detector_loader.detect_with_camera()
    
    open_camera = tk.Button(
            root,
            text = '摄像头实时检测',
            height = 3,
            command = camera_detection
    )
    
    open_camera.grid(
            row = 18,
            column = 12,
            rowspan = 2,
            columnspan = 4
    )

    # denosing preprocessing
    def denoise():
        global _res_im
        if '_res_im' not in globals():
            msg.showerror(
                    '操作流程错误',
                    '请先导入图片！'
            )
            return
        _res_im = median_blur(_res_im)
        change_display()
        status_bar.config(text='图像去噪声点完成！')
        root.update()

    denoise_btn = tk.Button(
            root,
            text = '去噪',
            height = 3,
            command = denoise
    )

    denoise_btn.grid(
            row = 18,
            column = 16,
            rowspan = 2,
            columnspan = 1
    )

    # dehazing preprocessing
    def im_dehaze():
        global _res_im
        if '_res_im' not in globals():
            msg.showerror(
                    '操作流程错误',
                    '请先导入图片！'
            )
            return
        _res_im = dehaze(_res_im)
        change_display()
        status_bar.config(text='图像去雾完成!')
        root.update()

    dehaze_btn = tk.Button(
            root,
            text = '去雾',
            height = 3,
            command = im_dehaze
    )
    
    dehaze_btn.grid(
            row = 18,
            column = 17,
            rowspan = 2,
            columnspan = 1
    )

    # enhancing image quality
    def im_enhance():
        global _res_im
        if '_res_im' not in globals():
            msg.showerror(
                    '操作流程错误',
                    '请先导入图片！'
            )
            return
        _res_im = enhance(_res_im)
        change_display()
        status_bar.config(text='图像增强完成！')
        root.update()

    enhance_btn = tk.Button(
            root,
            text = '图像增强',
            height = 3,
            command = im_enhance
    )

    enhance_btn.grid(
            row = 18,
            column = 18,
            rowspan = 2,
            columnspan = 2
    )

    # traditional segmentation
    def im_seg():
        global _res_im
        if '_res_im' not in globals():
            msg.showerror(
                    '操作流程错误',
                    '请先导入图片！'
            )
            return
        (_res_im, nobj) = seg(_res_im)
        change_display()
        status_bar.config(text='俯拍图舰船检测完成！共检测到目标： ' + str(nobj) + '个！' )
        root.update()

    seg_btn = tk.Button(
            root,
            text = '俯拍检测',
            height = 3,
            command = im_seg
    )

    seg_btn.grid(
            row = 18,
            column = 20,
            rowspan = 2,
            columnspan = 2
    )

    # right panels
    detector_model_chooser_label = tk.Label(
            text = '检测模型',
            width = 10
    )

    detector_model_chooser_label.grid(
            row = 0,
            column = 18,
            rowspan = 1,
            columnspan = 1
    )

    detector_list = ['SSD-Leng']
    
    detector_model_chooser = ttk.Combobox(
            values = detector_list,
            width = 12
    )

    detector_model_chooser.grid(
            row = 0,
            column = 19,
            rowspan = 1,
            columnspan = 2
    )
    
    classifier_model_chooser_label = tk.Label(
            text = '识别模型',
            width = 10
    )

    classifier_model_chooser_label.grid(
            row = 1,
            column = 18,
            rowspan = 1,
            columnspan = 1
    )

    classifier_list = ['AlexNet-Leng', 'ResNet-Chao', 'ResNet-Cui']
    
    classifier_model_chooser = ttk.Combobox(
            values = classifier_list,
            width = 12
    )
    
    classifier_model_chooser.grid(
            row = 1,
            column = 19,
            rowspan = 1,
            columnspan = 2
    )

    root.mainloop()

# execute the program
_main_()
