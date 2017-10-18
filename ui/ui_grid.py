# -*- coding:utf8 -*-
# Python 2.7+ and Tkinter 2.0+ and PIL 1.0+

import Tkinter as tk
import tkFileDialog as fd
import ttk
import io
from PIL import Image, ImageTk
import time

def _main_():
    root = tk.Tk()
    root.title('舰船检测识别原型系统')
    rw = 800
    rh = 500
    root.geometry(str(rw)+'x'+str(rh))
    root.resizable(width=False,height=False)

    # using grid to place panels
    w = 600
    h = 400
    # canvas
    canvas = tk.Canvas(root,width=w,height=h,bg='black')
    canvas.grid(row=0,column=0,rowspan=16,columnspan=12)

    # status bar
    status_bar = tk.Label(root,height=4,text='状态信息显示...')
    status_bar.grid(row=16,column=0,rowspan=2,columnspan=12)
    # task bar
    def select_file():
        global bg
        fn = fd.askopenfilename(title='选择图像文件',filetypes=[('images','*.jpg *.png'),('All Files','*')])
        im = Image.open(fn)
        im = im.resize([w,h],Image.ANTIALIAS)
        bg = ImageTk.PhotoImage(im)
        canvas.create_image(w/2,h/2,image=bg)
        root.update()

    select_file_btn = tk.Button(root,text='选择图片',height=3,command=select_file)
    select_file_btn.grid(row=18,column=0,rowspan=2,columnspan=1)

    ship_detect_btn = tk.Button(root,text='舰船检测',height=3)
    ship_detect_btn.grid(row=18,column=1,rowspan=2,columnspan=1)
    ship_recognize_btn = tk.Button(root,text='舰船识别',height=3)
    ship_recognize_btn.grid(row=18,column=2,rowspan=2,columnspan=1)
    load_detect_model_btn = tk.Button(root,text='导入检测模型',height=3)
    load_detect_model_btn.grid(row=18,column=3,rowspan=2,columnspan=1)
    load_recog_model_btn = tk.Button(root,text='导入识别模型',height=3)
    load_recog_model_btn.grid(row=18,column=4,rowspan=2,columnspan=1)


    # right panels
    detect_model_chooser_label = tk.Label(text='检测模型',width=10)
    detect_model_chooser_label.grid(row=0,column=12,rowspan=1,columnspan=1)
    detect_model_chooser = ttk.Combobox(values=['SSD','VGG'],width=12)
    detect_model_chooser.grid(row=0,column=13,rowspan=1,columnspan=1)
    
    recog_model_chooser_label = tk.Label(text='识别模型',width=10)
    recog_model_chooser_label.grid(row=1,column=12,rowspan=1,columnspan=1)
    recog_model_chooser = ttk.Combobox(values=['ResNet','DenseNet'],width=12)
    recog_model_chooser.grid(row=1,column=13,rowspan=1,columnspan=1)


    root.mainloop()

# execute the program
_main_()
