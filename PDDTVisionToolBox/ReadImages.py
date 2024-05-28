import os

import cv2
import math
import numpy as np
import datetime as dt
from pypylon import pylon
import matplotlib.pyplot as plt


def ReadSingleImage(path):
    Img = cv2.imread(path)
    return Img


def ReadSerialImages(FolderName):
    """
        在文件夹内读序列BMP图,必须是纯数字编号的图像序列
    :param FolderName: 文件夹名字
    :return:
    """
    import os
    # 图像所在文件夹，需在当前文件夹下
    ImgList = os.listdir(r"./" + FolderName)
    # 规定文件夹内图像命名格式必须是 0.bmp
    ImgList.sort(key=lambda x: int(x.split('.')[0]))
    # 初始化列表
    ImageArray = []
    for count in range(0, len(ImgList)):
        FileName = ImgList[count]
        Img = cv2.imread(FolderName + "/" + FileName)
        ImageArray.append(Img)
    return ImageArray


def getSingleImageByBasler():
    """
        单目Basler相机，获取单张图像
    :return:
    """
    # 函数功能：调用Basler相机成像
    # 连接Basler相机列表的第一个相机
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    # 开始读取图像
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()
    # 转换为OpenCV的BGR彩色格式
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    # while camera.IsGrabbing():
    # 抓取图像,数值为曝光时间,但是相机本身已经前期设置曝光时间,改了没啥用
    grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
    if grabResult.GrabSucceeded():
        # 转换为OpenCV图像格式
        image = converter.Convert(grabResult)
        Img = image.GetArray()
    grabResult.Release()
    # 关闭相机
    camera.StopGrabbing()
    return Img


def getDirPathOfLogsbYTime(name, childrenName, num):
    """
        按照时间分类的文件名
    :param name:主文件名称
    :param childrenName:各时间分类下所包含图像名称前缀
    :param num:循环创建子文件夹 数目
    :return:
    """
    # 创建元胞
    pathTuple = []
    timeNow = dt.datetime.now()
    dirPATH = name + '/' + str(timeNow)[0:13] + '-' + str(timeNow)[14:16] + '-' + str(timeNow)[17:19]
    # pathTuple[0] = dirPATH
    os.makedirs(dirPATH)
    for i in range(num):
        dirPATHChildren = dirPATH + '/' + childrenName + str(i)
        pathTuple.append(dirPATHChildren)
        os.makedirs(dirPATHChildren)
    return dirPATH, pathTuple
