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


def RunLooking(data_path,filename = None):
    if filename is None:
        files = os.listdir(data_path+'/looking/')
        for f in files:
            filepath = os.path.join(data_path, 'looking')
            LookingRecognition(filepath,f)
    else:
        for f in filename:
            filepath = os.path.join(data_path, 'looking')
            LookingRecognition(filepath,f)

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



def LookingRecognition(filepath,filename):
    MOUTH_THRESH = 2.5
    face_classifier_path = os.path.join(r'D:\model\haarcascades', 'haarcascade_frontalface_alt.xml')
    face_haar=cv2.CascadeClassifier(face_classifier_path)
    video_name = os.path.join(filepath, filename)
    cap = cv2.VideoCapture(video_name)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(os.path.join(filepath,'out',filename),cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    while(True):
        ret,img = cap.read()
        if(ret):
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces=face_haar.detectMultiScale(gray,scaleFactor =1.3,minNeighbors = 3)
            if len(faces) == 0:
                cv2.putText(img, "Looking Around", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            else:
                cv2.putText(img, "Not Looking Around", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
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
    RunLooking(data_path = 'D:\Office\研究生课程\数字图像处理\Fatigue Detection')
    #RunEye(data_path = 'D:\Office\研究生课程\数字图像处理\Fatigue Detection',filename = ['eye_closed_04.avi'])


            