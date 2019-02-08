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

def eye_aspect_ratio(p):
    EAR = (np.linalg.norm(p[1]-p[5])+np.linalg.norm(p[2]-p[4]))/(2*np.linalg.norm(p[0]-p[3]))
    return EAR

def RunEye(data_path,filename = None):
    if filename is None:
        files = os.listdir(data_path+'/eye/')
        for f in files:
            filepath = os.path.join(data_path, 'eye')
            EyeRecognition(filepath,f)
    else:
        for f in filename:
            filepath = os.path.join(data_path, 'eye')
            EyeRecognition(filepath,f)

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


def EyeRecognition(filepath,filename):
    EYE_AR_THRESH = 0.3# EAR阈值
    # 对应特征点的序号
    RIGHT_EYE_START = 37 - 1
    RIGHT_EYE_END = 42 - 1
    LEFT_EYE_START = 43 - 1
    LEFT_EYE_END = 48 - 1
     
    pwd = os.getcwd()# 获取当前路径
    model_path = os.path.join(pwd, 'model')# 模型文件夹路径
    shape_detector_path = os.path.join('D:\model', 'shape_predictor_68_face_landmarks.dat')# 人脸特征点检测模型路径
    detector = dlib.get_frontal_face_detector()# 人脸检测器
    predictor = dlib.shape_predictor(shape_detector_path)# 人脸特征点检测器
    video_name = os.path.join(filepath, filename)
    cap = cv2.VideoCapture(video_name)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(os.path.join(filepath,'out',filename),cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    E = []
    while(True):
        ret,img = cap.read()
        if(ret):
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2HLS_FULL)
            gray = Roi_Gray(img,filename)
            kernal = np.ones((5,5),np.float32)/25
            #gray = cv2.filter2D(gray,-1,kernal)
            #gray = cv2.equalizeHist(gray) #直方图均衡
            faces = detector(gray, 0)
            face_classifier_path = os.path.join(r'D:\model\haarcascades', 'haarcascade_frontalface_alt.xml')
            face_haar=cv2.CascadeClassifier(face_classifier_path)
            if len(faces) != 0:
                for face in faces:
                    shape = predictor(gray,face)
                    points = face_utils.shape_to_np(shape)
                    left_eye = points[LEFT_EYE_START:LEFT_EYE_END + 1]
                    right_eye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]
                    left_EAR = eye_aspect_ratio(left_eye)
                    right_EAR = eye_aspect_ratio(right_eye)
                    EAR = left_EAR*0.3 + right_EAR*0.7
                    E.append(EAR)
                    #left_EyeHull = cv2.convexHull(left_eye)# 寻找左眼轮廓
                    #right_EyeHull = cv2.convexHull(right_eye)# 寻找右眼轮廓
                    #cv2.drawContours(img, [left_EyeHull], -1, (0, 255, 0), 1)# 绘制左眼轮廓
                    #cv2.drawContours(img, [right_EyeHull], -1, (0, 255, 0), 1)# 绘制右眼轮廓
                    #cv2.rectangle(img,(left_eye[0][0],min(left_eye[4][1],left_eye[4][1])),((left_eye[3][0],min(left_eye[1][1],left_eye[2][1]))),(255,0,0), 1)
                    #cv2.rectangle(img,(right_eye[0][0],min(right_eye[4][1],right_eye[4][1])),((right_eye[3][0],min(right_eye[1][1],right_eye[2][1]))),(255,0,0), 1)
                    if EAR < EYE_AR_THRESH:
                        cv2.putText(img, "Close", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    else:
                        cv2.putText(img, "Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.putText(img, "EAR:{:.2f}".format(EAR), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.putText(img, filename, (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    for index, pt in enumerate(shape.parts()):
                        if index in range(RIGHT_EYE_START,LEFT_EYE_END+1):
                        #if(1):
                            pt_pos = (pt.x, pt.y)
                            #cv2.circle(gray, pt_pos, 2, (0, 255, 0), 3) 
                            cv2.circle(img, pt_pos, 2, (0, 255, 0), 3)    
                    cv2.imshow('image',img)
                    out.write(img)
                    #cv2.waitKey()
                    #cv2.imshow('image',img)
                    #cv2.waitKey()
            else:
                faces=face_haar.detectMultiScale(gray,scaleFactor =1.1,minNeighbors = 3, minSize = (80,80))
                for face_x,face_y,face_w,face_h in faces:
                    #cv2.rectangle(img, (face_x, face_y), (face_x+face_w, face_y+face_h), (0,255,0), 2)
                    face = dlib.rectangle(face_x, face_y,face_x+face_w, face_y+face_h)             
                    shape = predictor(gray,face)
                    points = face_utils.shape_to_np(shape)
                    left_eye = points[LEFT_EYE_START:LEFT_EYE_END + 1]
                    right_eye = points[RIGHT_EYE_START:RIGHT_EYE_END + 1]
                    left_EAR = eye_aspect_ratio(left_eye)
                    right_EAR = eye_aspect_ratio(right_eye)
                    EAR = left_EAR*0.3 + right_EAR*0.7
                    E.append(EAR)
                    #cv2.rectangle(img,(left_eye[0][0],min(left_eye[4][1],left_eye[4][1])),((left_eye[3][0],min(left_eye[1][1],left_eye[2][1]))),(255,0,0), 1)
                    #cv2.rectangle(img,(right_eye[0][0],min(right_eye[4][1],right_eye[4][1])),((right_eye[3][0],min(right_eye[1][1],right_eye[2][1]))),(255,0,0), 1)
                    if EAR < EYE_AR_THRESH:
                        cv2.putText(img, "Close", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    else:
                        cv2.putText(img, "Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.putText(img, "EAR:{:.2f}".format(EAR), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.putText(img, filename, (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    for index, pt in enumerate(shape.parts()):
                        if index in range(RIGHT_EYE_START,LEFT_EYE_END+1):
                        #if(1):
                            pt_pos = (pt.x, pt.y)
                            #cv2.circle(gray, pt_pos, 2, (0, 255, 0), 3)  
                            cv2.circle(img, pt_pos, 2, (0, 255, 0), 3)         
                              
                    cv2.imshow('image',img)
                    out.write(img)
                    #cv2.waitKey()
                    #cv2.imshow('image',img)
                    #cv2.waitKey()
        else:
            break
        if cv2.waitKey(1)==27:
            break
    cap.release()
    #PlotEAR(E)

def Roi_Gray(gray, f):
    if f == 'eye_closed_04.avi':
        gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
        gray = contrast_brightness_image(gray,2,15)
    return gray


def EyeRecognition2(filepath,filename):
    eye_classifier_path = os.path.join(r'D:\model\haarcascades', 'haarcascade_eye_tree_eyeglasses.xml')
    face_classifier_path = os.path.join(r'D:\model\haarcascades', 'haarcascade_frontalface_default.xml')
    face_haar=cv2.CascadeClassifier(face_classifier_path)
    eye_haar=cv2.CascadeClassifier(eye_classifier_path)
    video_name = os.path.join(filepath, filename)
    cap = cv2.VideoCapture(video_name)
    while(True):
        ret,img = cap.read()
        if(ret):
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces=face_haar.detectMultiScale(gray,scaleFactor =1.1,minNeighbors = 5)
            for face_x,face_y,face_w,face_h in faces:
                cv2.rectangle(img, (face_x, face_y), (face_x+face_w, face_y+face_h), (0,255,0), 2)
                roi_gray_img=gray[face_y:face_y+face_h,face_x:face_x+face_w]
                roi_img=img[face_y:face_y+face_h,face_x:face_x+face_w]
                eyes=eye_haar.detectMultiScale(roi_gray_img,scaleFactor =1.3,minNeighbors = 4)
                for eye_x,eye_y,eye_w,eye_h in eyes:
                    cv2.rectangle(roi_img, (eye_x,eye_y), (eye_x+eye_w, eye_y+eye_h), (255,0,0), 2)      
            cv2.imshow('img', img)
            cv2.waitKey()
        else:
            break
        if cv2.waitKey(1)==27:
            break
    cap.release()
            
if __name__ == "__main__":
    RunEye(data_path = 'D:\Office\研究生课程\数字图像处理\Fatigue Detection')
    #RunEye(data_path = 'D:\Office\研究生课程\数字图像处理\Fatigue Detection',filename = ['eye_closed_04.avi'])


            