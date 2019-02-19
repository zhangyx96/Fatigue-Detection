# Fatigue-Detection
### Detect the drivers fatigue real-time using python and OpenCV

The project is my final homework of the course Digital Image Processing. I use the OpenCV and python 3.6 to realize 4 different aspects for the problem of drivers' fatigue real-time detection.

## Dependencies
+ python 3.6
+ OpenCV 3.3.1
+ dlib
+ imutils
  
## **A**--- Eye Closed Detect
### Step 1: 视频读取
使用OpenCV库自带的VideoCapture类可以实现对视频每一帧的读取以及视频大小、长宽信息的读取。
### Step 2: 人脸检测

要检测眼睛闭合首先要检测出图像中的人脸。检测人脸目前开源的效果比较的python库有dlib库和OpenCV自带的人脸检测器。

<img src = "imgs/eye_table0.png" width = "500">

### Step 3: 人脸特征点检测
dlib数据库提供了训练好的人脸68点特征点检测器，并且都有固定的标号

### Step 4: 人眼闭合检测
参考Tereza Soukupova和Jan ´ Cech的论文[6]，其中提出了眼睛纵横比的概念——eye aspect ratio (EAR)。通过计算这个值，可以判断图像中的人眼是张开还是闭合。论文中给出的EAR的定义如下：
<img src = "imgs/f0.png">


