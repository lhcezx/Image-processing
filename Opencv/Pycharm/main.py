#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 00:54:49 2021

@author: haochengluo
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt 

zaodian = cv.imread('../image/zaodian.png',1) #0代表用灰度模式读取图片，1是以彩色行书读取图片 透明度会被忽略
img0 = cv.imread('../image/small_fat.jpg',1)
img = cv.imread('../image/test.jpeg',1)
img1=cv.imread('../image/kaola.jpeg',1)
detect_corner=cv.imread('../image/detect_corner.jpeg',1)
print(detect_corner)
row,col=img.shape[:2]

# img1 = cv.circle(img1, (10,100), 10, (255,0,0))
# img1[10:20,100:130]=(255,0,0)
# plt.imshow(img1[:,:,::-1])
# plt.show()

#cv.imshow('image',img)
#cv.waitKey(0)

# cv.line(img,(0,0),(1935,1890),(0,0,255),100)

# cv.putText(img,"XIAO PANG PI",(700,500), cv.FONT_HERSHEY_COMPLEX,5,(255,255,255),25)

# 图像色彩类型转变 BGR-HSV
# img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# 图像加和
# img = cv.addWeighted(img,0.7,img0,0.3,0)

# 修改图像大小
# img=cv.resize(img,dsize=None,fx=0.1,fy=0.1)

# 平移图像
# ligne,colonne=img.shape[:2]
# M=np.float32([[1,0,100],[0,1,100]])
# img=cv.warpAffine(img, M, (2*colonne+1,2*ligne))

# 图像旋转

# M=cv.getRotationMatrix2D((col/2,row/2), 90, 1)
# img=cv.warpAffine(img, M, (col*2,row*2))


# 仿射变换
# p_init=np.float32([[0,0],[1000,1000],[1500,1500]])
# p_fin=np.float32([[500,500],[1300,1300],[1800,1800]])
# M=cv.getAffineTransform(p_init, p_fin)
# img=cv.warpAffine(img,M,(row,col))


# 图像金字塔
# img_up=cv.pyrUp(img)
# plt.imshow(img_up[:,:,::-1])


# 开闭运算

# 开运算：先腐蚀后膨胀 消除噪点
# plt.subplot(1,3,1)
# plt.imshow(img[:,:,::-1])

# plt.subplot(1,3,2)
# kernel=np.ones([3,3],dtype='uint8')
# img=cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
# plt.imshow(img[:,:,::-1])

# 闭运算：先膨胀后腐蚀 填充闭合区域
# plt.subplot(1,3,3)
# kernel=np.ones([3,3],dtype='uint8')
# img=cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
# plt.imshow(img[:,:,::-1])

# 平滑处理
# plt.figure()
# plt.imshow(zaodian[:,:,::-1])

# plt.figure()
# zaodian_a = cv.Blur(zaodian,(3,3))  均值滤波 核在图像上加权平均（核内每个点加权相等）
# zaodian_a = cv.GaussianBlur(zaodian,(3,3),1)  高斯滤波  图像名称/核/X轴的方差 （核内每个点加权不相等 呈高斯分布
# zaodian_a = cv.medianBlur(zaodian,5)  #中值滤波 不受最大最小值影响
# plt.imshow(zaodian_a[:,:,::-1])

# 图像像素灰度直方图
# hist=cv.calcHist(img1,[0],None,[256],[0,256])  # [0]代表灰度 1 2 3 分别代表三个通道，[256]表示直方数量, [0,256]表示灰度范围
# plt.plot(hist)
# plt.grid()

# 掩膜(mask)  (这里用于灰度图像(单通道))
# mask=np.zeros(img1.shape[:2],np.uint8)
# print(img1.shape[:2])
# mask[50:160,50:180]=1
# plt.imshow(mask,cmap='gray')

# mask_img=cv.bitwise_and(img1,img1,mask=mask)
# plt.imshow(mask_img,cmap='gray')


#直方图均衡化 (增强全局对比度,会减少局部细节)
# hist=cv.calcHist(img1, [0], None, [256], [0,256])
# plt.subplot(2,2,1)
# plt.plot(hist)

# plt.subplot(2,2,2)
# plt.imshow(img1,cmap='gray')

# plt.subplot(2,2,3)
# img1=cv.equalizeHist(img1)
# hist_final=cv.calcHist(img1,[0],None, [256], [0,256])
# plt.plot(hist_final)

# plt.subplot(2,2,4)
# plt.imshow(img1,cmap='gray')

#自适应直方图均衡化 (全局切分后局部均衡化,不会减少局部细节)
# plt.figure()
# clahe=cv.createCLAHE(clipLimit=50.0, tileGridSize=(8,8)) #clipLimit=对比度限制, tileGridSize=分割块的大小
# img_final=clahe.apply(img1)
# plt.imshow(img_final,cmap='gray')

#边缘检测 (图像灰度的变化图 一阶导数最大值:基于搜索 二阶导数为0:基于零穿越)

#sobel算子 卷积核用于计算f(x,y)灰度函数的一阶导数 当该一阶导数大于某一值的时候便是到达灰度变化边界的位置
#Sobel卷积后的结果
# Sobel_x=cv.Sobel(zaodian, cv.CV_16S, dx=1, dy=0) # cv.CV_165是为了存储大于255或者小于0的数据，原图像为8为无符号，这里用16位有符号数据
# Sobel_y=cv.Sobel(zaodian, cv.CV_16S, dx=0, dy=1)
#Sobel数据转换
# Scale_absx=cv.convertScaleAbs(Sobel_x)
# Scale_absy=cv.convertScaleAbs(Sobel_y)
#结果合成
# resultat=cv.addWeighted(Scale_absx, 0.5, Scale_absy, 0.5, 0)
# plt.figure(figsize=(10,8),dpi=100)
# plt.subplot(1,2,1)
# plt.imshow(zaodian[:,:,::-1])
# plt.subplot(1,2,2)
# plt.imshow(resultat[:,:,::-1])

#laplacien算子 该算子用于求f(x,y)的二阶导数 如果二阶导数为0则达到灰度变化边界位置
# laplacian=cv.Laplacian(zaodian, cv.CV_16S)
# Scale_absx=cv.convertScaleAbs(laplacian)
# plt.figure(figsize=(10,8),dpi=100)
# plt.subplot(1,2,1)
# plt.imshow(zaodian[:,:,::-1])
# plt.subplot(1,2,2)
# plt.imshow(Scale_absx[:,:,::-1])
#Laplacian算子把对x和对y的二阶导数放在一起算不需要分开 但是sobel算子需要分开后addWeight加和

#Canny边缘检测 降噪-高斯平滑-边缘搜索
# low_threshold=200
# high_threshold=240
# canny=cv.Canny(zaodian, low_threshold, high_threshold)
# plt.figure(figsize=(10,8),dpi=100)
# plt.subplot(1,2,1)
# plt.imshow(zaodian[:,:,::-1])
# plt.subplot(1,2,2)
# plt.imshow(canny)


#模版匹配

#角点检测
#Harris角点检测 用determinant和tracer来定义一个R值然后和阈值进行对比
# gray_img = cv.cvtColor(detect_corner,cv.COLOR_BGR2GRAY) #将彩色图片以灰色图片模式读取但不改变原彩色图片 (相当于把imread的1改成0)
# gray_img = np.float32(gray_img) #讲图片的数据类型转变成float32（因为下面的操作可能会将灰度值大于255或者小于0）

#cornerHarris返回的是一个R值构成的灰度图像 (下面可以plot和print) R值通常特别大 所以当图像上的灰度值大于R_max的百分之一就可以被看作是角点)
# corner=cv.cornerHarris(gray_img, blockSize=2, ksize=3, k=0.05) #blockSize是角点检测中的邻域大小, ksize是sobel算子的卷积核大小，k是角点检测中的自由参数[0.04,0.06]
# plt.imshow(corner, cmap='gray')
# plt.show()
# print(corner)

# detect_corner[corner>0.01*corner.max()]=[0,0,255]  #标记出角点的位置为红色
# plt.imshow(detect_corner[:,:,::-1])
# plt.show()

# #shi_tomasi角点检测  两个特征值都大于阈值的话那就是角点
# gray_img=cv.cvtColor(detect_corner,cv.COLOR_BGR2GRAY)
# #cv.goodFeaturesToTrack()直接返回检测到的角点的坐标
# corner_shi=cv.goodFeaturesToTrack(gray_img, maxCorners=100, qualityLevel=0.01, minDistance=10)  #Maxcorners是最多返回的角点数量, qualityLevel是可接受的角点质量水平[0,1], minDistance是两个检测的角点如果距离小于指定距离则被看作为一个角点
# corner_shi = np.int0(corner_shi)
#
# for i in corner_shi:
#     x,y=i.ravel() #ravel的目的是把获得的坐标变成一维 ex: [[214,165]] -> [214,165]
#     cv.circle(detect_corner, (x, y), 3, 255, -1) #在找到的坐标点上画圆
# plt.imshow(detect_corner[:,:,::-1])
# plt.show()

#sift特征点检测算法
#对图像用高斯核进行卷积 卷积一定的层数后降采样 降低图像的尺寸后再次高斯卷积 如此往复
#构建高斯差分金字塔 从根据每个像素点的26个邻域点找极值 确定极值点位置后在其点上进行三维泰勒展开 插值法找准确的极值点
#找到后以点为圆心画一个圆 统计圆内的像素点的梯度强度和方向(ps: 圆中的点的距离和圆心的距离应当以高斯计算）后画梯度的各个方向的强度直方图 由此确定中心点的主方向以及副方向
#最后将点周围区域划分成16个小的长方形区域 每个区域内统计8个方向的梯度强度（每45度取一个方向) 总计8*16=128个区域方向的梯度强度 最后将这128个值放到一个向量里将其称之为特征向量也就是关键点描述符

# sift = cv.SIFT()
# gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
# kp, des = sift.detectAndCompute(gray, None) #kp 关键点信息，包括位置，尺度，方向信息，des是关键点描述符(128个梯度信息的特征向量)
# cv.drawKeypoints(img1,kp,img1,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) #img1对应的分别是输入和输出的图像
# plt.imshow(img1[:,:,::-1])

#Fast算法特征点检测
#提取到的特征点没有方向
# 提取到的特征点不满足尺度变化。
# fast = cv.FastFeatureDetector_create(threshold = 30, nonmaxSuppression = True) #threshold是设置的阈值 nonmaxsuppression是用来选择是否用非极大值抑制
# kp = fast.detect(img1, None)
# cv.drawKeypoints(img1,kp,img1,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# plt.imshow(img1[:,:,::-1])
# plt.show()

#ORB算法特征点检测
#ORB算法使用的是FAST算法提取的特征点，用harris角点检测对特征点进行排序后取前N个
#构建图像金字塔 每组仅有一层 每层对应一个方差且每层的尺寸不一样 构建尺度不变
# orb = cv.ORB_create(nfeatures = 10) #nfeatures 是特征点的数量
# kp, des = orb.detectAndCompute(img1, None)
# cv.drawKeypoints(img1,kp,img1)
# plt.imshow(img1[:,:,::-1])
# plt.show()




