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
import matplotlib
matplotlib.rcsetup.interactive_bk # 获取 interactive backend
matplotlib.rcsetup.non_interactive_bk # 获取 non-interactive backend
matplotlib.rcsetup.all_backends # 获取 所有 backend

def mouth_aspect_ratio(p):
    EAR = (np.linalg.norm(p[1]-p[11])+np.linalg.norm(p[2]-p[10])+np.linalg.norm(p[3]-p[9])+
           np.linalg.norm(p[4]-p[8])+np.linalg.norm(p[5]-p[7]))/(5*np.linalg.norm(p[0]-p[6]))
    return EAR

def RunMouth(data_path,filename = None):
    if filename is None:
        files = os.listdir(data_path+'/mouth/')
        for f in files:
            filepath = os.path.join(data_path, 'mouth')
            MouthRecognition2(filepath,f)
    else:
        for f in filename:
            filepath = os.path.join(data_path, 'mouth')
            MouthRecognition2(filepath,f)

def PlotEAR(E):
    N = len(E)
    x = list(range(0,N))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(ylim=[0, 8], ylabel='Ratio', xlabel='frame')
    t = np.zeros(N)+2.5
    ax.plot(x,E,'r-+')
    ax.plot(x,t,'b',label = 'Threshold=2.5')
    ax.legend()
    plt.show()


    
def contrast_brightness_image(src1, a, g):
    h, w = src1.shape#获取shape的数值，height和width、通道
 
    #新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
    src2 = np.zeros([h, w], src1.dtype)
    dst = cv2.addWeighted(src1, a, src2, 1-a, g)#addWeighted函数说明如下
    return dst


def MouthRecognition(filepath,filename):
    MOUTH_AR_THRESH = 0.5# EAR阈值
    # 对应特征点的序号
    MOUTH_START = 48
    MOUTH_END = 60
     
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
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #gray = contrast_brightness_image(gray,2,15)
            #gray = Roi_Gray(img,filename)
            #kernal = np.ones((5,5),np.float32)/25
            #gray = cv2.filter2D(gray,-1,kernal)
            #gray = cv2.equalizeHist(gray) #直方图均衡
            faces = detector(gray, 0)
            face_classifier_path = os.path.join(r'D:\model\haarcascades', 'haarcascade_profileface.xml')
            face_haar=cv2.CascadeClassifier(face_classifier_path)
            #if len(faces) != 0:
            if(0):
                for face in faces:
                    shape = predictor(gray,face)
                    points = face_utils.shape_to_np(shape)
                    mouth = points[MOUTH_START:MOUTH_END + 1]
                    EAR = mouth_aspect_ratio(mouth)
                    E.append(EAR)
                   
                    if EAR < MOUTH_AR_THRESH:
                        cv2.putText(img, "Close", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    else:
                        cv2.putText(img, "Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.putText(img, "EAR:{:.2f}".format(EAR), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.putText(img, filename, (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    for index, pt in enumerate(shape.parts()):
                        #if index in range(MOUTH_START,MOUTH_END+1):
                        if(1):
                            pt_pos = (pt.x, pt.y)
                            #cv2.circle(gray, pt_pos, 2, (0, 255, 0), 3) 
                            cv2.circle(img, pt_pos, 2, (0, 255, 0), 3)    
                    cv2.imshow('image',img)
                    out.write(img)
                    cv2.waitKey()
                    #cv2.imshow('image',img)
                    #cv2.waitKey()
            else:
                gray = cv2.flip(gray,flipCode = 1)
                faces=face_haar.detectMultiScale(gray,scaleFactor =1.1,minNeighbors = 3, minSize = (100,100))
                for face_x,face_y,face_w,face_h in faces:
                    face_x_flip = frame_width-(face_x+face_w)
                    cv2.rectangle(gray, (face_x, face_y), (face_x+face_w, face_y+face_h), (0,255,0), 2)
                    face = dlib.rectangle(face_x, face_y,face_x+face_w, face_y+face_h)             
                    shape = predictor(gray,face)
                    points = face_utils.shape_to_np(shape)
                    mouth = points[MOUTH_START:MOUTH_END + 1]
                    EAR = mouth_aspect_ratio(mouth)
                    E.append(EAR)
    
                    if EAR < MOUTH_AR_THRESH:
                        cv2.putText(img, "Close", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    else:
                        cv2.putText(img, "Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.putText(img, "EAR:{:.2f}".format(EAR), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.putText(img, filename, (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    for index, pt in enumerate(shape.parts()):
                        if index in range(MOUTH_START,MOUTH_END+1):
                        #if(1):
                            pt_pos = (frame_width - pt.x, pt.y)
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

def MouthRecognition2(filepath,filename):
    MOUTH_THRESH = 2.5
    face_classifier_path = os.path.join(r'D:\model\haarcascades', 'haarcascade_frontalface_alt.xml')
    face_classifier_path2 = os.path.join(r'D:\model\haarcascades', 'haarcascade_profileface.xml')
    face_haar=cv2.CascadeClassifier(face_classifier_path)
    face_haar2=cv2.CascadeClassifier(face_classifier_path2)
    #mouth_classifier_path = os.path.join(r'D:\model\haarcascades', 'haarcascade_mcs_mouth.xml')
    #mouth_haar=cv2.CascadeClassifier(mouth_classifier_path)
    video_name = os.path.join(filepath, filename)
    cap = cv2.VideoCapture(video_name)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(os.path.join(filepath,'out',filename),cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    R = []
    while(True):
        ret,img = cap.read()
        if(ret):
            ratio = 0
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            gray = cv2.flip(gray,flipCode = 1)
            faces=face_haar.detectMultiScale(gray,scaleFactor =1.1,minNeighbors = 3,minSize=(100,200))
            if len(faces)==0:
                gray = cv2.flip(gray,flipCode = 1)
                img = cv2.flip(img,flipCode = 1)
                faces=face_haar2.detectMultiScale(gray,scaleFactor =1.1,minNeighbors = 3,minSize=(100,200))
                for x,y,w,h in faces:
                    widthOneCorner = x + int(w / 8)
                    widthOtherCorner = x + int(3 * w/ 5)
                    heightOneCorner = y + int(7 * h / 11)
                    heightOtherCorner = y + int(9 * h / 10)
                    #cv2.rectangle(gray, (widthOneCorner, heightOneCorner), (widthOtherCorner, heightOtherCorner),(0,0,255), 2) 
                    mouthRegion = gray[heightOneCorner:heightOtherCorner, widthOneCorner:widthOtherCorner]
                    rectArea = (w*h)/2
                    if(len(mouthRegion) > 0):
                        ratio = thresholdContours(mouthRegion, rectArea)
                        R.append(ratio)
                        if ratio < MOUTH_THRESH:
                            cv2.putText(img, "Close", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                        else:
                            cv2.putText(img, "Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                img = cv2.flip(img,flipCode = 1)
                cv2.putText(img, "Ratio:{:.2f}".format(ratio), (90, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.putText(img, filename, (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            else:
                for x,y,w,h in faces:
                    widthOneCorner = x + int(w / 8)
                    widthOtherCorner = x + int(3 * w/ 5)
                    heightOneCorner = y + int(7 * h / 11)
                    heightOtherCorner = y + int(9 * h / 10)
                    #cv2.rectangle(gray, (widthOneCorner, heightOneCorner), (widthOtherCorner, heightOtherCorner),(0,0,255), 2) 
                    mouthRegion = gray[heightOneCorner:heightOtherCorner, widthOneCorner:widthOtherCorner]
                    rectArea = (w*h)/2
                    if(len(mouthRegion) > 0):
                        ratio = thresholdContours(mouthRegion, rectArea)
                        R.append(ratio)
                        if ratio < MOUTH_THRESH:
                            cv2.putText(img, "Close", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                        else:
                            cv2.putText(img, "Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                        cv2.putText(img, "Ratio:{:.2f}".format(ratio), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                        cv2.putText(img, filename, (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.imshow('image',img)
            out.write(img)
        else:
            break
        if cv2.waitKey(1)==27:
            break
    cap.release()
    #PlotEAR(R)

def calculateContours(image, contours):
    #cv2.drawContours(image, contours, -1, (0,255,0), 3)
    maxArea = 0
    secondMax = 0
    maxCount = 0
    secondmaxCount = 0
    for i in contours:
        count = i
        area = cv2.contourArea(count)
        if maxArea < area:
            secondMax = maxArea
            maxArea = area
            secondmaxCount = maxCount
            maxCount = count
        elif (secondMax < area):
            secondMax = area
            secondmaxCount = count
	
    return [secondmaxCount, secondMax]

def thresholdContours(mouthRegion, rectArea):
    ratio = 0

	# Histogram equalize the image after converting the image from one color space to another
	# Here, converted to greyscale    imgray = cv2.equalizeHist(cv2.cvtColor(mouthRegion, cv2.COLOR_BGR2GRAY))
    
    kernal = np.ones((5,5),np.float32)/25
    imgray = cv2.filter2D(mouthRegion,-1,kernal)
    imgray = contrast_brightness_image(imgray, 1.3, 20)
    imgray = cv2.equalizeHist(mouthRegion) 


	# Thresholding the image => outputs a binary image. 
	# Convert each pixel to 255 if that pixel each exceeds 64. Else convert it to 0. 
    ret,thresh = cv2.threshold(imgray, 70, 255, cv2.THRESH_BINARY)

	# Finds contours in a binary image
	# Constructs a tree like structure to hold the contours 
	# Contouring is done by having the contoured region made by of small rectangles and storing only the end points
	# of the rectangle 
    image,cnts,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    returnValue = calculateContours(mouthRegion, cnts)

	# returnValue[0] => secondMaxCount
	# returnValue[1] => Area of the contoured region.
    secondMaxCount = returnValue[0]
    contourArea = returnValue[1]
	
    ratio = (contourArea / rectArea)*1000

	# Draw contours in the image passed. The contours are stored as vectors in the array. 
	# -1 indicates the thickness of the contours. Change if needed. 
    if(isinstance(secondMaxCount, np.ndarray) and len(secondMaxCount) > 0):
        cv2.drawContours(imgray, [secondMaxCount], 0, (255,0,0), -1)
    #cv2.putText(imgray, "ratio:{:.2f}".format(round(ratio,2)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 2)
    #cv2.imshow('mouthRegion',imgray)
    #cv2.waitKey()    
    return ratio

          
if __name__ == "__main__":
    RunMouth(data_path = 'D:\Office\研究生课程\数字图像处理\Fatigue Detection')
    #RunMouth(data_path = 'D:\Office\研究生课程\数字图像处理\Fatigue Detection',filename = ['yawn_02.avi'])


            