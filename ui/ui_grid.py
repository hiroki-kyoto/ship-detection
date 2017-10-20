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
sys.path.append('../ssd_vgg_300/')

# ssd-vgg-300
import ssd

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
    
    # global data
    global fn
    global detect_model_file
    fn = ''
    detect_model_file = ''

    # canvas
    canvas = tk.Canvas(root,width=w,height=h,bg='black')
    canvas.grid(row=0,column=0,rowspan=16,columnspan=12)

    # status bar
    status_bar = tk.Label(root,height=4,text='状态信息显示...')
    status_bar.grid(row=16,column=0,rowspan=2,columnspan=12)

    # task bar
    # select
    def select_file():
        global bg
        global fn
        fn = fd.askopenfilename(title='选择图像文件',filetypes=[('images','*.jpg *.png'),('All Files','*')])
        im = Image.open(fn)
        im = im.resize([w,h],Image.ANTIALIAS)
        bg = ImageTk.PhotoImage(im)
        canvas.create_image(w/2,h/2,image=bg)
        root.update()

    select_file_btn = tk.Button(root,text='选择图片',height=3,command=select_file)
    select_file_btn.grid(row=18,column=0,rowspan=2,columnspan=1)
    
    # detect
    def detect():
        global fn
        global detect_model_file
        if fn==[] or fn=='':
            msg.showerror('操作流程错误','请先导入要检测的图片!')
            return
        if detect_model_file==[] or detect_model_file=='':
            msg.showerror('操作流程错误','请先导入舰船检测模型!')
            return
        status_bar.config(text='正在检测船只...')
        root.update()
        ssd.ssd_run(detect_model_file, fn)
        status_bar.config(text='船只检测完成！')
        root.update()

    ship_detect_btn = tk.Button(root,text='舰船检测',height=3,command=detect)
    ship_detect_btn.grid(row=18,column=1,rowspan=2,columnspan=1)
    
    # recognize
    def recognize():
        status_bar.config(text='正在识别舰船类别...')
        root.update()
        
    ship_recognize_btn = tk.Button(root,text='舰船识别',height=3,command=recognize)
    ship_recognize_btn.grid(row=18,column=2,rowspan=2,columnspan=1)

    # load detect model
    def load_detect_model():
        global detect_model_file
        if detect_model_chooser.get() == '':
            msg.showerror('配置错误','请先设置要导入哪种检测模型！')
        elif detect_model_chooser.get()=='SSD':
            detect_model_file = fd.askdirectory()
            info = '当前使用检测模型为:' + detect_model_chooser.get()
            status_bar.config(text=info + ', 正在读取检测模型...')
            root.update()
            # load SSD model
            if detect_model_file != '':
                status_bar.config(text='舰船检测模型设置成功！')
                root.update()


    load_detect_model_btn = tk.Button(root,text='导入检测模型',height=3,command=load_detect_model)
    load_detect_model_btn.grid(row=18,column=3,rowspan=2,columnspan=1)

    # load recognization model
    load_recog_model_btn = tk.Button(root,text='导入识别模型',height=3)
    load_recog_model_btn.grid(row=18,column=4,rowspan=2,columnspan=1) 
    
    # right panels
    detect_model_chooser_label = tk.Label(text='检测模型',width=10)
    detect_model_chooser_label.grid(row=0,column=12,rowspan=1,columnspan=1)
    dm_list = ['SSD']
    detect_model_chooser = ttk.Combobox(values=dm_list,width=12)
    detect_model_chooser.grid(row=0,column=13,rowspan=1,columnspan=1)
    
    recog_model_chooser_label = tk.Label(text='识别模型',width=10)
    recog_model_chooser_label.grid(row=1,column=12,rowspan=1,columnspan=1)
    rm_list = ['VGG','RestNet']
    recog_model_chooser = ttk.Combobox(values=rm_list,width=12)
    recog_model_chooser.grid(row=1,column=13,rowspan=1,columnspan=1)


    root.mainloop()

# execute the program
_main_()
