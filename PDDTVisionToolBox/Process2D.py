import cv2
import math
import os
import subprocess
import glob
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
from pylibdmtx import pylibdmtx
from PIL import Image, ImageDraw


def getRGBToGray(Img):
    ImgGray = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
    return ImgGray


def getAngleBy3PointsInDegree(point_1, point_2, point_3):
    """
        根据三点坐标计算夹角
    :param point_1: 点1坐标
    :param point_2: 点2坐标
    :param point_3: 点3坐标
    :return: 返回任意角的夹角值，这里只是返回点2的夹角
    """
    a = math.sqrt(
        (point_2[0] - point_3[0]) * (point_2[0] - point_3[0]) + (point_2[1] - point_3[1]) * (point_2[1] - point_3[1]))
    b = math.sqrt(
        (point_1[0] - point_3[0]) * (point_1[0] - point_3[0]) + (point_1[1] - point_3[1]) * (point_1[1] - point_3[1]))
    c = math.sqrt(
        (point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) + (point_1[1] - point_2[1]) * (point_1[1] - point_2[1]))
    A = math.degrees(math.acos((a * a - b * b - c * c) / (-2 * b * c)))
    B = math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c)))
    C = math.degrees(math.acos((c * c - a * a - b * b) / (-2 * a * b)))
    return B


def getDis2D(pos1x, pos1y, pos2x, pos2y):
    """
        计算两点距离
    :param pos1x: 位置1X
    :param pos1y: 位置1Y
    :param pos2x: 位置2X
    :param pos2y: 位置2Y
    :return:
    """
    Dis = math.sqrt((pos1x - pos2x) ** 2 + (pos1y - pos2y) ** 2)
    return Dis


def getDisPointToLine(Slope, Intercept, col, row):
    """
        根据斜率结局计算点线距离
    :param Slope: 斜率
    :param Intercept:截距
    :param col: col
    :param row: row
    :return: 距离
    """
    d1 = abs(Slope * col - row + Intercept)
    d2 = math.sqrt(1 + Slope ** 2)
    d = d1 / d2
    return d


def getMask(img):
    """
        通过手动框选，构成掩膜图像修正
    :param img:传入需校准图像
    :return: Mask 图像掩膜
    """

    def on_mouse(event, x, y, flags, param):
        img = param[0]
        Mask = param[1]
        global point1, point2
        img2 = img.copy()
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
            point1 = (x, y)
            cv2.circle(img, point1, 1, (0, 255, 0), 1)
            cv2.imshow('image', img2)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 移动鼠标，左键拖拽
            cv2.rectangle(img2, point1, (x, y), (255, 0, 0), 5)
            cv2.imshow('image', img2)
        elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
            point2 = (x, y)
            cv2.rectangle(img, point1, point2, (0, 0, 255), 5)
            cv2.imshow('image', img)
            min_x = min(point1[0], point2[0])
            min_y = min(point1[1], point2[1])
            Mwidth = abs(point1[0] - point2[0])
            Mheight = abs(point1[1] - point2[1])
            # print(min_x, min_y, width, height)
            # print('mask[', min_y, ':', min_y + Mheight, ',', min_x, ':', min_x + Mwidth, '] = 1')
            Mask[min_y:min_y + Mheight, min_x:min_x + Mwidth] = 1
            return Mask

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    width, height = np.shape(img)
    Mask = np.zeros((width, height), dtype=np.uint8)
    Tuple = (img, Mask)
    cv2.namedWindow('image', 0)
    cv2.resizeWindow('image', width, height)
    cv2.setMouseCallback('image', on_mouse, Tuple)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    return Mask


def getCirclePos(Img, Times, minDis, minR, maxR):
    """
        采用霍夫圆方法计算圆心并返回
    :param Img: 输入图像
    :param Times: 放大倍率
    :param minDis: 最小间距
    :param minR: 最小半径
    :param maxR: 最大半径
    :return:
    """
    kernelRow = np.ones((3, 1), np.uint8)
    kernelCol = np.ones((1, 3), np.uint8)
    ImgGray = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
    height, width = np.shape(ImgGray)
    img = cv2.bilateralFilter(ImgGray, d=0, sigmaColor=15, sigmaSpace=15)
    ImgBW = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
    ImgBW = cv2.medianBlur(ImgBW, 5)
    ImgBW = cv2.erode(ImgBW, kernelRow)
    ImgBW = cv2.erode(ImgBW, kernelCol)
    circles = cv2.HoughCircles(ImgBW, cv2.HOUGH_GRADIENT, Times, minDis, param1=50, param2=60, minRadius=minR,
                               maxRadius=maxR)
    # print(circles)
    # circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # print(i[2])
        CenterPosC = "(" + str(i[0]) + "," + str(i[1]) + ")"
        i = np.uint16(np.around(i))
        # 画出来圆的边界
        cv2.circle(Img, (i[0], i[1]), i[2], (0, 0, 255), 2)
        # 画出来圆心
        cv2.circle(Img, (i[0], i[1]), 2, (0, 0, 255), 3)
        cv2.putText(Img, CenterPosC, (i[0], i[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.namedWindow("Test1", 0)
    cv2.resizeWindow("Test1", 1920, 1080)
    cv2.imshow('Test1', Img)
    cv2.waitKey(0)

    return circles


def getCirclePosED(Img, Method, MinPathLength, MinLineLength, GradientThresholdValue):
    """
        通过EdgeDrawing进行圆检测，相比霍夫算法更稳定
    :param Img: 输入图像
    :param Method: 边缘检测算法，0 Prewitt，1 Sobel，2 Scharr, 3 LSD
    :param MinPathLength:最小连接像素 5 ~ 1000 default 50
    :param MinLineLength:最小线段长度 5 ~ 100  default 10
    :param GradientThresholdValue:灰度阈值
    :return:
    """
    ImgGray = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
    ImgDraw = Img.copy()
    ed = cv2.ximgproc.createEdgeDrawing()
    EDParams = cv2.ximgproc_EdgeDrawing_Params()
    EDParams.EdgeDetectionOperator = Method
    EDParams.MinPathLength = MinPathLength  # try changing this value between 5 to 1000
    EDParams.PFmode = False  # defaut value try to switch it to True
    EDParams.MinLineLength = MinLineLength  # try changing this value between 5 to 100
    EDParams.NFAValidation = True  # defaut value try to switch it to False
    EDParams.GradientThresholdValue = GradientThresholdValue
    ed.setParams(EDParams)
    ed.detectEdges(ImgGray)
    segments = ed.getSegments()
    # lines = ed.detectLines()
    ellipses = ed.detectEllipses()
    if ellipses is not None:  # Check if circles and ellipses have been found and only then iterate over these and add them to the image
        # print(ellipses)
        num = len(ellipses)
        # print(num)
        count = 0
        circlePos = np.zeros((num, 3))
        for i in range(len(ellipses)):
            centerPos = " (" + str(ellipses[i][0][0]) + "," + str(ellipses[i][0][1]) + ")"
            center = (int(ellipses[i][0][0]), int(ellipses[i][0][1]))
            axes = (int(ellipses[i][0][2]) + int(ellipses[i][0][3]), int(ellipses[i][0][2]) + int(ellipses[i][0][4]))
            angle = ellipses[i][0][5]
            color = (0, 0, 255)
            if ellipses[i][0][2] == 0:
                color = (0, 255, 0)
            else:
                circlePos[count, 0] = ellipses[i][0][0]
                circlePos[count, 1] = ellipses[i][0][1]
                circlePos[count, 2] = ellipses[i][0][2]
                count = count + 1
                cv2.ellipse(ImgDraw, center, axes, angle, 0, 360, color, 2, cv2.LINE_AA)
                cv2.circle(ImgDraw, center, 2, (0, 0, 255), 3)
                cv2.putText(ImgDraw, centerPos, center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(ImgDraw, str(count), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        circlePos = circlePos[~np.all(circlePos == 0, axis=1)]
        cv2.namedWindow("Test", 0)
        cv2.resizeWindow("Test", 1920, 1080)
        cv2.imshow('Test', ImgDraw)
        cv2.waitKey(0)
    return circlePos


def getCirclePosEDWithRadiusEstimate(Img, Method, MinPathLength, MinLineLength, GradientThresholdValue, ratio, minRadis,
                                     maxRadius):
    """
        通过EdgeDrawing进行圆检测，相比霍夫算法更稳定
    :param Img: 输入图像
    :param Method: 边缘检测算法，0 Prewitt，1 Sobel，2 Scharr, 3 LSD
    :param MinPathLength:最小连接像素 5 ~ 1000 default 50
    :param MinLineLength:最小线段长度 5 ~ 100  default 10
    :param GradientThresholdValue:灰度阈值
    :param ratio: 比例尺
    :param minRadis: 最小半径 mm
    :param maxRadius: 最大半径 mm
    :return:
    """
    ImgGray = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
    height, width = np.shape(ImgGray)
    ImgDraw = Img.copy()
    # 创建EdgeDrawing对象
    ed = cv2.ximgproc.createEdgeDrawing()
    EDParams = cv2.ximgproc_EdgeDrawing_Params()
    EDParams.EdgeDetectionOperator = Method
    EDParams.MinPathLength = MinPathLength  # try changing this value between 5 to 1000
    EDParams.PFmode = False  # defaut value try to switch it to True
    EDParams.MinLineLength = MinLineLength  # try changing this value between 5 to 100
    EDParams.NFAValidation = True  # defaut value try to switch it to False
    EDParams.GradientThresholdValue = GradientThresholdValue
    ed.setParams(EDParams)
    ed.detectEdges(ImgGray)
    segments = ed.getSegments()
    # lines = ed.detectLines()
    ellipses = ed.detectEllipses()
    # ellipses 返回的是 列 行
    if ellipses is not None:  # Check if circles and ellipses have been found and only then iterate over these and add them to the image
        # print(ellipses)
        num = len(ellipses)
        # print(num)
        count = 0
        circlePos = np.zeros((num, 3))
        for i in range(len(ellipses)):
            radius = ellipses[i][0][2] * ratio
            if ellipses[i][0][2] == 0:
                color = (0, 255, 0)
            elif minRadis < radius < maxRadius:
                circlePos[count, 0] = ellipses[i][0][0]
                circlePos[count, 1] = ellipses[i][0][1]
                circlePos[count, 2] = ellipses[i][0][2]
                count = count + 1
        # 删除相邻点
        index = np.lexsort((circlePos[:, 2], circlePos[:, 1], circlePos[:, 0]))
        circlePos = circlePos[index]
        for i in range(num - 1):
            Dis = math.sqrt((circlePos[i + 1, 0] - circlePos[i, 0]) ** 2 + (circlePos[i + 1, 1] - circlePos[i, 1]) ** 2)
            if Dis < 50:
                if circlePos[i + 1, 2] > circlePos[i, 2]:
                    circlePos[i + 1, 0] = 0
                    circlePos[i + 1, 1] = 0
                    circlePos[i + 1, 2] = 0
                else:
                    circlePos[i, 0] = 0
                    circlePos[i, 1] = 0
                    circlePos[i, 2] = 0
        # 删除零行
        circlePos = circlePos[~np.all(circlePos == 0, axis=1)]
        index = np.lexsort((circlePos[:, 1], circlePos[:, 0], circlePos[:, 2]))
        circlePos = circlePos[index]
        # 绘图
        count = 0
        for i in range(len(circlePos[:, 0])):
            r = circlePos[i][2] * ratio
            marks = 1 / (1 + (abs(r - (minRadis + maxRadius) / 2) * 3) ** 2)
            gainMarks = '   Marks:' + str(round(marks * 100, 4))
            # centerPos = " (" + str(round(circlePos[i, 0], 3)) + "," + str(round(circlePos[i, 1], 3)) + ")"
            radius = " r = " + str(round(circlePos[i, 2] * ratio, 4)) + 'mm'
            radiusPos = (int(circlePos[i, 0] - 10), int(circlePos[i, 1] + 30))
            center = (int(circlePos[i, 0]), int(circlePos[i, 1]))
            axes = (int(circlePos[i, 2]), int(circlePos[i, 2]))
            # angle = ellipses[i][0][5]
            color = (0, 0, 255)
            count = count + 1
            cv2.ellipse(ImgDraw, center, axes, 0, 0, 360, color, 2, cv2.LINE_AA)
            cv2.circle(ImgDraw, center, 2, (0, 0, 255), 3)
            # cv2.putText(ImgDraw, centerPos, center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(ImgDraw, gainMarks, center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(ImgDraw, str(count), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(ImgDraw, radius, radiusPos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # cv2.namedWindow("Test", 0)
        # cv2.resizeWindow("Test", 1920, 1080)
        # cv2.imshow('Test', ImgDraw)
        # cv2.waitKey(0)
    return circlePos, ImgDraw


def getCirclePosAdapted(Img, Method, MinPathLength, MinLineLength, GradientThresholdValue, ratio, minRadis, maxRadius):
    """
        通过EdgeDrawing进行圆检测，相比霍夫算法更稳定
    :param Img: 输入图像
    :param Method: 边缘检测算法，0 Prewitt，1 Sobel，2 Scharr, 3 LSD
    :param MinPathLength:最小连接像素 5 ~ 1000 default 50
    :param MinLineLength:最小线段长度 5 ~ 100  default 10
    :param GradientThresholdValue:灰度阈值
    :param ratio: 比例尺
    :param minRadis: 最小半径 mm
    :param maxRadius: 最大半径 mm
    :return:
    """
    ImgGray = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
    height, width = np.shape(ImgGray)
    ImgDraw = Img.copy()
    # 创建EdgeDrawing对象
    ed = cv2.ximgproc.createEdgeDrawing()
    EDParams = cv2.ximgproc_EdgeDrawing_Params()
    EDParams.EdgeDetectionOperator = Method
    EDParams.MinPathLength = MinPathLength  # try changing this value between 5 to 1000
    EDParams.PFmode = False  # defaut value try to switch it to True
    EDParams.MinLineLength = MinLineLength  # try changing this value between 5 to 100
    EDParams.NFAValidation = False  # defaut value try to switch it to False
    EDParams.GradientThresholdValue = GradientThresholdValue
    ed.setParams(EDParams)
    ed.detectEdges(ImgGray)
    segments = ed.getSegments()
    # lines = ed.detectLines()
    ellipses = ed.detectEllipses()
    # ellipses 返回的是 列 行
    if ellipses is not None:  # Check if circles and ellipses have been found and only then iterate over these and add them to the image
        # print(ellipses)
        num = len(ellipses)
        # print(num)
        count = 0
        circlePos = np.zeros((num, 3))
        for i in range(len(ellipses)):
            radius = ellipses[i][0][2] * ratio
            if ellipses[i][0][2] == 0:
                color = (0, 255, 0)
            elif minRadis < radius < maxRadius:
                circlePos[count, 0] = ellipses[i][0][0]
                circlePos[count, 1] = ellipses[i][0][1]
                circlePos[count, 2] = ellipses[i][0][2]
                count = count + 1
        # 删除相邻点
        index = np.lexsort((circlePos[:, 2], circlePos[:, 1], circlePos[:, 0]))
        circlePos = circlePos[index]
        for i in range(num - 1):
            Dis = math.sqrt((circlePos[i + 1, 0] - circlePos[i, 0]) ** 2 + (circlePos[i + 1, 1] - circlePos[i, 1]) ** 2)
            if Dis < 10:
                if circlePos[i + 1, 2] > circlePos[i, 2]:
                    circlePos[i + 1, 0] = 0
                    circlePos[i + 1, 1] = 0
                    circlePos[i + 1, 2] = 0
                else:
                    circlePos[i, 0] = 0
                    circlePos[i, 1] = 0
                    circlePos[i, 2] = 0
        # 删除零行
        circlePos = circlePos[~np.all(circlePos == 0, axis=1)]
        index = np.lexsort((circlePos[:, 1], circlePos[:, 0], circlePos[:, 2]))
        circlePos = circlePos[index]
        # 绘图
        count = 0
        for i in range(len(circlePos[:, 0])):
            centerPos = " (" + str(round(circlePos[i, 0], 3)) + "," + str(round(circlePos[i, 1], 3)) + ")"
            radius = " r = " + str(round(circlePos[i, 2] * ratio, 4)) + 'mm'
            radiusPos = (int(circlePos[i, 0] - 10), int(circlePos[i, 1] + 30))
            center = (int(circlePos[i, 0]), int(circlePos[i, 1]))
            axes = (int(circlePos[i, 2]), int(circlePos[i, 2]))
            # angle = ellipses[i][0][5]
            color = (0, 0, 255)
            count = count + 1
            cv2.ellipse(ImgDraw, center, axes, 0, 0, 360, color, 2, cv2.LINE_AA)
            cv2.circle(ImgDraw, center, 2, (0, 0, 255), 3)
            cv2.putText(ImgDraw, centerPos, center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(ImgDraw, str(count), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(ImgDraw, radius, radiusPos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # cv2.namedWindow("Test", 0)
        # cv2.resizeWindow("Test", 1920, 1080)
        # cv2.imshow('Test', ImgDraw)
        # cv2.waitKey(0)
    return circlePos, ImgDraw


def getCirclePosEDWithLinearFitting(Img, Method, MinPathLength, MinLineLength, GradientThresholdValue, ratio, minRadis,
                                    maxRadius):
    """
        通过EdgeDrawing进行圆检测，相比霍夫算法更稳定
    :param Img: 输入图像
    :param Method: 边缘检测算法，0 Prewitt，1 Sobel，2 Scharr, 3 LSD
    :param MinPathLength:最小连接像素 5 ~ 1000 default 50
    :param MinLineLength:最小线段长度 5 ~ 100  default 10
    :param GradientThresholdValue:灰度阈值
    :param ratio: 比例尺
    :param minRadis: 最小半径 mm
    :param maxRadius: 最大半径 mm
    :return:
    """
    ImgGray = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
    height, width = np.shape(ImgGray)
    ImgDraw = Img.copy()
    # 创建EdgeDrawing对象
    ed = cv2.ximgproc.createEdgeDrawing()
    EDParams = cv2.ximgproc_EdgeDrawing_Params()
    EDParams.EdgeDetectionOperator = Method
    EDParams.MinPathLength = MinPathLength  # try changing this value between 5 to 1000
    EDParams.PFmode = False  # defaut value try to switch it to True
    EDParams.MinLineLength = MinLineLength  # try changing this value between 5 to 100
    EDParams.NFAValidation = True  # defaut value try to switch it to False
    EDParams.GradientThresholdValue = GradientThresholdValue
    ed.setParams(EDParams)
    ed.detectEdges(ImgGray)
    segments = ed.getSegments()
    # lines = ed.detectLines()
    ellipses = ed.detectEllipses()
    # ellipses 返回的是 列 行
    if ellipses is not None:  # Check if circles and ellipses have been found and only then iterate over these and add them to the image
        # print(ellipses)
        num = len(ellipses)
        # print(num)
        count = 0
        circlePos = np.zeros((num, 3))
        for i in range(len(ellipses)):
            # centerPos = " (" + str(round(ellipses[i][0][0], 3)) + "," + str(round(ellipses[i][0][1], 3)) + ")"
            # radius = " r = " + str(round(ellipses[i][0][2] * ratio, 3)) + 'mm'
            # radiusPos = (int(ellipses[i][0][0] - 10), int(ellipses[i][0][1] + 30))
            # center = (int(ellipses[i][0][0]), int(ellipses[i][0][1]))
            # axes = (int(ellipses[i][0][2]) + int(ellipses[i][0][3]), int(ellipses[i][0][2]) + int(ellipses[i][0][4]))
            # angle = ellipses[i][0][5]
            # color = (0, 0, 255)
            dis = math.sqrt((ellipses[i][0][0] - width / 2) ** 2 + (ellipses[i][0][1] - height / 2) ** 2)
            ratioR = ratio[0] * dis + ratio[1]
            radius = ellipses[i][0][2] * ratioR
            if ellipses[i][0][2] == 0:
                color = (0, 255, 0)
            elif minRadis < radius < maxRadius:
                circlePos[count, 0] = ellipses[i][0][0]
                circlePos[count, 1] = ellipses[i][0][1]
                circlePos[count, 2] = ellipses[i][0][2]
                count = count + 1
                # cv2.ellipse(ImgDraw, center, axes, angle, 0, 360, color, 2, cv2.LINE_AA)
                # cv2.circle(ImgDraw, center, 2, (0, 0, 255), 3)
                # cv2.putText(ImgDraw, centerPos, center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # cv2.putText(ImgDraw, str(count), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.putText(ImgDraw, radius, radiusPos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # 删除相邻点
        index = np.lexsort((circlePos[:, 2], circlePos[:, 1], circlePos[:, 0]))
        circlePos = circlePos[index]
        # print(circlePos)
        for i in range(num - 1):
            Dis = math.sqrt((circlePos[i + 1, 0] - circlePos[i, 0]) ** 2 + (circlePos[i + 1, 1] - circlePos[i, 1]) ** 2)
            if Dis < 10:
                if circlePos[i + 1, 2] > circlePos[i, 2]:
                    circlePos[i + 1, 0] = 0
                    circlePos[i + 1, 1] = 0
                    circlePos[i + 1, 2] = 0
                else:
                    circlePos[i, 0] = 0
                    circlePos[i, 1] = 0
                    circlePos[i, 2] = 0
        # 删除零行
        circlePos = circlePos[~np.all(circlePos == 0, axis=1)]
        index = np.lexsort((circlePos[:, 1], circlePos[:, 2], circlePos[:, 2]))
        circlePos = circlePos[index]
        # 绘图
        count = 0
        for i in range(len(circlePos[:, 0])):
            centerPos = " (" + str(round(circlePos[i, 0], 3)) + "," + str(round(circlePos[i, 1], 3)) + ")"
            dis = math.sqrt((ellipses[i][0][0] - width / 2) ** 2 + (ellipses[i][0][1] - height / 2) ** 2)
            ratioR = ratio[0] * dis + ratio[1]
            radius = " r = " + str(round(circlePos[i, 2] * ratioR, 4)) + 'mm'
            radiusPos = (int(circlePos[i, 0] - 10), int(circlePos[i, 1] + 30))
            center = (int(circlePos[i, 0]), int(circlePos[i, 1]))
            axes = (int(circlePos[i, 2]), int(circlePos[i, 2]))
            # angle = ellipses[i][0][5]
            color = (0, 0, 255)
            count = count + 1
            cv2.ellipse(ImgDraw, center, axes, 0, 0, 360, color, 2, cv2.LINE_AA)
            cv2.circle(ImgDraw, center, 2, (0, 0, 255), 3)
            cv2.putText(ImgDraw, centerPos, center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(ImgDraw, str(count), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(ImgDraw, radius, radiusPos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # cv2.namedWindow("Test", 0)
        # cv2.resizeWindow("Test", 1920, 1080)
        # cv2.imshow('Test', ImgDraw)
        # cv2.waitKey(0)
    return circlePos, ImgDraw


def getCalibrationMatrix(fx, fy, u, v, k1, k2, p1, p2, k3):
    """
        将标定数据重组为矩阵，一般用于处理来自Matlab的数据
    :param fx: 焦距x
    :param fy: 焦距y
    :param u: 光心偏差
    :param v: 光心偏差
    :param k1: 畸变参数k1
    :param k2: 畸变参数k2
    :param p1: 畸变参数p1
    :param p2: 畸变参数p2
    :param k3: 畸变参数k3
    :return:
    """
    Intrinsic = np.array([[fx, 0, u],
                          [0, fy, v],
                          [0, 0, 1]])
    Distortion = np.array([[k1, k2, p1, p2, k3]])
    return Intrinsic, Distortion


def getLocationByYOLOX(Img, Model_Type):
    """
        利用YOLOX识别工件类型并返回其形心位置
    :param Img: 输入图像
    :param Model_Type:调用模型
    :return:
    """
    if Model_Type == 0:
        pathImgWrite = './img/1.bmp'
        pathPosRead = './pose_data/data.csv'
    else:
        print('所选模型未训练,请重新选择')
        os._exit(0)
    cv2.imwrite(pathImgWrite, Img)
    myPose = subprocess.Popen("predict.exe")
    try:
        myPose.wait(timeout=100)
    except Exception as e:
        print("Time Out")
        myPose.kill()
    ImgResult = cv2.imread('./img_out/1.bmp')
    # cv2.namedWindow("Test", 0)
    # cv2.resizeWindow("Test", 1920, 1080)
    # cv2.imshow('Test', ImgResult)
    # cv2.waitKey(0)
    pos = np.genfromtxt(pathPosRead, delimiter=',', dtype=int)
    return pos


def getMarkersPos(Img):
    """
        通过ArUco贴码对对面所在平面进行检测
    :param Img: 输入图像
    :return: 贴码位置,如果检测靶标数目相同，则返回靶标位置将是按照顺序对应的
    """
    # 灰度化
    ImgD = Img.copy()
    Img = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
    # 确定亚像素提升精度
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    # 确定使用的ArUco码字典
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters_create()
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(Img, dictionary, parameters=parameters)
    markerIdsSorted = sorted(markerIds)
    # print(markerIdsSorted)
    if len(markerCorners) > 0:
        # 展平 ArUCo ID 列表
        ids = markerIds.flatten()
        markerPos = np.zeros((100, 3))
        # 循环检测到的 ArUCo 标记
        for (markerCorner, markerID) in zip(markerCorners, ids):
            # print(markerID)
            # 提取始终按以下顺序返回的标记：
            # TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT
            corners = markerCorner.reshape((4, 2))
            # 采用亚像素角点优化提高精度
            corners = cv2.cornerSubPix(Img, corners, (5, 5), (-1, -1), criteria)
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # 先计算ArUco位置中心
            cX = (topLeft[0] + bottomRight[0]) / 2.0
            cY = (topLeft[1] + bottomRight[1]) / 2.0
            # 根据MarkerID给MarkerPos赋值
            markerPos[markerID, 0] = cX
            markerPos[markerID, 1] = cY
            markerPos[markerID, 2] = markerID
            # 将每个 (x, y) 坐标对转换为整数
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            # 绘制ArUCo检测的边界框
            cv2.line(ImgD, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(ImgD, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(ImgD, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(ImgD, bottomLeft, topLeft, (0, 255, 0), 2)
            # 计算并绘制 ArUCo 标记的中心 (x, y) 坐标
            icX = int(cX)
            icY = int(cY)
            cv2.circle(ImgD, (icX, icY), 4, (0, 255, 0), -1)
            # # 在图像上绘制 ArUco 标记 ID
            cv2.putText(ImgD, str(markerID), (topLeft[0], topLeft[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2)
    # 显示识别ID
    cv2.namedWindow("ProcessingTest", 0)
    cv2.resizeWindow("ProcessingTest", 1920, 1080)
    cv2.imshow('ProcessingTest', ImgD)
    cv2.waitKey(0)
    markerPos = markerPos[~np.all(markerPos == 0, axis=1)]
    # print(markerPos)
    return markerPos


def getDMMarkersMatrix(Level):
    """
        生成多阶DM二维码
    :param Level: 阶数
    :return:
    """
    # 创建文件夹，若无
    img_dir = "./DMCodes"
    timeNow = dt.datetime.now()
    dirPATH = img_dir + '/' + str(timeNow)[0:13] + '-' + str(timeNow)[14:16] + '-' + str(timeNow)[17:19]
    os.makedirs(dirPATH)
    Num = Level ** 2
    for i in range(Num):
        if i < 10:
            info = '''Number: 000''' + str(i) + '''
            '''
            img = pylibdmtx.encode(info.encode('utf8'))
            img = Image.frombytes('RGB', (img.width, img.height), img.pixels)
            img.save(dirPATH + '/000' + str(i) + '.jpg')
        elif i < 100:
            info = '''Number: 00''' + str(i) + '''
            '''
            img = pylibdmtx.encode(info.encode('utf8'))
            img = Image.frombytes('RGB', (img.width, img.height), img.pixels)
            img.save(dirPATH + '/00' + str(i) + '.jpg')
        else:
            info = '''Number: 0''' + str(i) + '''
            '''
            img = pylibdmtx.encode(info.encode('utf8'))
            img = Image.frombytes('RGB', (img.width, img.height), img.pixels)
            img.save(dirPATH + '/0' + str(i) + '.jpg')
        # img = pylibdmtx.encode(info.encode('utf8'))
        # img = Image.frombytes('RGB', (img.width, img.height), img.pixels)
        # img.save(dirPATH + '/' + str(i) + '.jpg')

    def image_concat(image_names):
        """ image_names: list, 存放的是图片的绝对路径 """
        # 1.创建一块背景布
        image = Image.open(image_names[0])
        width, height = image.size
        target_shape = ((Level + 1) * width, (Level + 1) * height)
        background = Image.new('RGBA', target_shape, (0, 0, 0, 0,))

        # 2.依次将图片放入背景中(注意图片尺寸规整、mode规整、放置位置)
        for ind, image_name in enumerate(image_names):
            img = Image.open(image_name)
            img = img.resize((width, height))  # 尺寸规整
            if img.mode != "RGBA":  # mode规整
                img = img.convert("RGBA")
            row, col = ind // Level, ind % Level
            location = (round(col * width * ((Level + 1) / Level)), round(row * height * ((Level + 1) / Level)))  # 放置位置
            background.paste(img, location)

        # 3.中心画十字
        x = round(((Level ** 2 + Level - 1) / Level * width) / 2)
        y = round(((Level ** 2 + Level - 1) / Level * height) / 2)  # 找到图形中心
        line1 = [(x - round(width / 2), y), (x + round(width / 2), y)]
        line2 = [(x, y - round(height / 2)), (x, y + round(height / 2))]
        backgroundWithLine = ImageDraw.Draw(background)
        backgroundWithLine.line(line1, fill='black', width=0)
        backgroundWithLine.line(line2, fill='black', width=0)
        # cv2.line(background, (x - round(width / 2), y), (x + round(width / 2), y), (255, 255, 255), 2)
        # cv2.line(background, (x, y - round(height / 2)), (x, y + round(height / 2)), (255, 255, 255), 2)
        background.save("./DMCodesMixed.png")
        # backgroundWithLine.

    img_dir = dirPATH + '/'
    image_names = sorted(glob.glob(img_dir + "*"))
    image_concat(image_names)


def getDMMarkersPos(Img, TimeOut, MarkerLevel, MaxMarkersNum):
    """
        识别DM 二维码并定位起始点
        二维码编码格式为getDMMarkers编码格式
    :param Img: 输入图像
    :param TimeOut:时间
    :param MarkerLevel:待检测二维码数量
    :param MaxMarkersNum: 最大待检测编码点总数
    :return: 
    """
    # 获得图像高
    ImgGray = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
    height, width = np.shape(ImgGray)
    # 解码
    # codeInfo = pylibdmtx.decode(Img, timeout=round(height*width/500), max_count=MarkerTotal)
    codeInfo = pylibdmtx.decode(Img, timeout=TimeOut, max_count=MaxMarkersNum)
    # print(len(codeInfo))
    # 作图定位
    ImgD = Img.copy()
    codePos = np.zeros((MarkerLevel ** 2, 2))
    for i in range(len(codeInfo)):
        # print(codeInfo[i].data.decode('utf8'))
        # 确定编码编号
        MarkerInfo = codeInfo[i].data.decode('utf8')[9:12]
        # print(MarkerInfo)
        MarkerID = int(MarkerInfo)
        codePos[MarkerID, 0] = codeInfo[i].rect[0]
        codePos[MarkerID, 1] = height - codeInfo[i].rect[1]
        # print(int(MarkerInfo))
        cv2.circle(ImgD, (codeInfo[i].rect[0], height - codeInfo[i].rect[1]), 4, (20, 0, 255), 2)
        cv2.putText(ImgD, MarkerInfo, (codeInfo[i].rect[0], height - codeInfo[i].rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 4)
    # cv2.namedWindow("Test", 0)
    # cv2.resizeWindow("Test", round(height / 2), round(width / 2))
    # cv2.imshow('Test', ImgD)
    # cv2.waitKey(0)
    return codeInfo, codePos, ImgD


def getCirclesGridsPose(Img, GridRow, GridCol, CircleMinArea, CircleMaxArea, CircleDistance, SymmetricEnable):
    """
        获得圆形标定板中，各标定圆位置
    :param Img: 输入图像
    :param GridRow: 圆行数
    :param GridCol: 圆列数
    :param CircleMinArea: 圆区域最小像素数 推荐1000
    :param CircleMaxArea: 圆区域最大像素数 推荐10000
    :param CircleDistance:  圆间距像素数
    :param SymmetricEnable: 是否进行角点细化 推荐不做
    :return:
    """
    # 灰度化
    gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    # 初始化斑点检测器
    params = cv2.SimpleBlobDetector_Params()
    params.maxArea = CircleMaxArea
    params.minArea = CircleMinArea
    params.minDistBetweenBlobs = CircleDistance
    blobDetector = cv2.SimpleBlobDetector_create(params)
    # 对称分化检测参数
    if SymmetricEnable == 1:
        ret, corners = cv2.findCirclesGrid(gray, (GridRow, GridCol), cv2.CALIB_CB_SYMMETRIC_GRID, blobDetector, None)
    else:
        ret, corners = cv2.findCirclesGrid(gray, (GridRow, GridCol), cv2.CALIB_CB_ASYMMETRIC_GRID, blobDetector, None)
    cv2.drawChessboardCorners(Img, (GridRow, GridCol), corners, corners is not None)
    # cv2.namedWindow("Test", 0)
    # cv2.resizeWindow("Test", 1920, 1080)
    # cv2.imshow('Test', Img)
    # cv2.waitKey(0)
    return Img, corners
