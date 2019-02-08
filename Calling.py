# -*- coding: utf-8 -*-
"""
Created on 2019/1/6
@author: Zhang Yuanxin
"""
# -*- coding: utf-8 -*-
"""
Created on 2019/1/6
@author: Zhang Yuanxin
"""
import numpy as np
import os,sys
import cv2
import dlib
from scipy.spatial import distance
import matplotlib as mpl
from imutils import face_utils
import matplotlib.pyplot as plt


def RunCalling(data_path,filename = None):
    if filename is None:
        files = os.listdir(data_path+'/calling/')
        for f in files:
            filepath = os.path.join(data_path, 'calling')
            CallingRecognition(filepath,f)
    else:
        for f in filename:
            filepath = os.path.join(data_path, 'calling')
            CallingRecognition(filepath,f)

def PlotEAR(E):
    N = len(E)
    x = list(range(0,N))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(ylim=[0.20, 0.35], ylabel='EAR', xlabel='frame')
    t = np.zeros(N)+0.3
    ax.plot(x,E,'r-+')
    ax.plot(x,t,'b',label = 'Threshold=0.3')
    ax.legend()
    plt.show()


    
def contrast_brightness_image(src1, a, g):
    h, w = src1.shape#获取shape的数值，height和width、通道
 
    #新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
    src2 = np.zeros([h, w], src1.dtype)
    dst = cv2.addWeighted(src1, a, src2, 1-a, g)#addWeighted函数说明如下
    return dst

def CalciRatio(b1,T):
    M = b1.shape[0]
    N = b1.shape[1]
    ratio = np.sum(b1>T)/(M*N)
    return ratio

def CallingRecognition(filepath,filename):
    MOUTH_THRESH = 2.5
    RATIO_THRESH = 0.3
    face_classifier_path = os.path.join(r'D:\model\haarcascades', 'haarcascade_frontalface_alt.xml')
    face_classifier_path2 = os.path.join(r'D:\model\haarcascades', 'haarcascade_profileface.xml')
    face_haar=cv2.CascadeClassifier(face_classifier_path)
    face_haar2=cv2.CascadeClassifier(face_classifier_path2)
    video_name = os.path.join(filepath, filename)
    cap = cv2.VideoCapture(video_name)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(os.path.join(filepath,'out',filename),cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    while(True):
        ret,img = cap.read()
        if(ret):
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces=face_haar.detectMultiScale(gray,scaleFactor =1.1,minNeighbors = 2,minSize=(200,200))
            ratio = 0 #初始化ratio
            if len(faces) == 0:  #没有检测到正脸，使用侧脸检测器
                gray = cv2.flip(gray,flipCode = 1)
                img = cv2.flip(img,flipCode = 1)
                faces=face_haar2.detectMultiScale(gray,scaleFactor =1.1,minNeighbors = 1,minSize=(200,200))
                for x,y,w,h in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 2)
                    right_x1 = int(x - w*0.3)
                    right_x2 = int(x + w*0.05)
                    right_y1 = int(y + h*0.6)
                    right_y2 = int(y + h*1.2)
                    left_x1 = int(x + w*0.9)
                    left_x2 = int(x + w*1.3)
                    left_y1 = int(y + h*0.8)
                    left_y2 = int(y + h*1.4)
                    
                    block_right = gray[right_y1:right_y2,right_x1:right_x2]
                    block_left = gray[left_y1:left_y2,left_x1:left_x2]
                    cv2.rectangle(img,(right_x1,right_y1),(right_x2,right_y2),(255,0,0), 2)
                    cv2.rectangle(img,(left_x1,left_y1),(left_x2,left_y2),(255,0,0), 2)
                    ratio = max(CalciRatio(block_right,T=100),CalciRatio(block_left,T = 100))
                img = cv2.flip(img,flipCode = 1)
            else:
                for x,y,w,h in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0), 2)
                    right_x1 = int(x - w*0.3)
                    right_x2 = int(x + w*0.05)
                    right_y1 = int(y + h*0.6)
                    right_y2 = int(y + h*1.2)
                    left_x1 = int(x + w*1.1)
                    left_x2 = int(x + w*1.3)
                    left_y1 = int(y + h*0.8)
                    left_y2 = int(y + h*1.4)
                    block_right = gray[right_y1:right_y2,right_x1:right_x2]
                    block_left = gray[left_y1:left_y2,left_x1:left_x2]
                    cv2.rectangle(img,(right_x1,right_y1),(right_x2,right_y2),(255,0,0), 2)
                    cv2.rectangle(img,(left_x1,left_y1),(left_x2,left_y2),(255,0,0), 2)
                    ratio = max(CalciRatio(block_right,T=100),CalciRatio(block_left,T = 100))
            cv2.putText(img, 'Ratio:{}'.format(round(ratio,2)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            if ratio > RATIO_THRESH:
                cv2.putText(img, "Calling", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:
                cv2.putText(img, "Not Calling", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(img, filename, (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)                      
            cv2.imshow('image',img)
            out.write(img)
        else:
            break
        if cv2.waitKey(1)==27:
            break
    cap.release()
    #PlotEAR(R)
if __name__ == "__main__":
    RunCalling(data_path = 'D:\Office\研究生课程\数字图像处理\Fatigue Detection')
    #RunCalling(data_path = 'D:\Office\研究生课程\数字图像处理\Fatigue Detection',filename = ['Phoning_02.avi'])


            