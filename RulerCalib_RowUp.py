import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from time import sleep

import PDDTVisionToolBox as pd

# 设定IP地址及端口号
IPRob = "192.168.0.101"
PortNumber = 30003
# 回到初始位置
pd.getUR10EMove(IPRob, PortNumber, 0.315, 0.115, 0.680, 1.760, 0.729, 1.760)
# 相机标定参数
Intrinsic = np.array([[2988.40434044024, 0, 1934.71209993405],
                      [0, 2987.99338701425, 1036.58326580770],
                      [0, 0, 1]])
Distortion = np.array([-0.0501552424699594, 0.0238122396566109, 0, 0])
# 初始尺度
dRuler0 = 0.05927844978595915
# 检测数目
detectNum = 4
# 目标半径尺寸
# M4
# minRadius = 1.5
# maxRadius = 2
# M6
minRadius = 2.5
maxRadius = 4
# 机械臂0位
posX = 0.315
posY = 0.115
posZ = 0.680
# 初始位置检测
count = 0
while 1:
    count = count + 1
    print('初始位置,第', count, '次检测')
    # 读图
    Img = pd.getSingleImageByBasler()
    # 去畸变
    Img = cv2.undistort(Img, Intrinsic, Distortion)
    # 灰度化
    ImgGray = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
    # 获得图像高宽
    Height, Width = np.shape(ImgGray)
    # 检测圆
    circlePos, ImgDraw = pd.getCirclePosEDWithRadiusEstimate(Img, 1, 50, 10, 30, dRuler0, minRadius, maxRadius)
    # 图示显示
    cv2.namedWindow("Test", 0)
    cv2.resizeWindow("Test", 1920, 1080)
    cv2.imshow('Test', ImgDraw)
    cv2.waitKey(0)
    if len(circlePos[:, 0]) == detectNum:
        print('检测成功')
        break
# # 图示显示
# cv2.namedWindow("Test", 0)
# cv2.resizeWindow("Test", 1920, 1080)
# cv2.imshow('Test', ImgDraw)
# cv2.waitKey(0)
# 从坐到右排序
index = np.lexsort((circlePos[:, 2], circlePos[:, 1], circlePos[:, 0]))
circlePos = circlePos[index]
# 循环位置检测
# 初始化尺度
IterDis = 0
for i in range(3):
    for j in range(3):
        if i == 1 and j == 1:
            continue
        else:
            # 计算增量
            dy = i * 0.005 - 0.005
            dz = j * 0.005 - 0.005
            # 移动机械臂到位
            pd.getUR10EMove(IPRob, PortNumber, posX, posY + dy, posZ + dz, 1.760, 0.729, 1.760)
            # 检测圆
            count = 0
            while 1:
                count = count + 1
                print('(', i + 1, ',', j + 1, ')位置, 第', count, '次检测')
                # 读图
                Img = pd.getSingleImageByBasler()
                # 去畸变
                Img = cv2.undistort(Img, Intrinsic, Distortion)
                # 灰度化
                ImgGray = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
                # 获得图像高宽
                Height, Width = np.shape(ImgGray)
                # 检测圆
                circlePosNew, ImgDraw = pd.getCirclePosEDWithRadiusEstimate(Img, 1, 50, 10, 30, dRuler0, minRadius,
                                                                            maxRadius)
                # 图示显示
                cv2.namedWindow("Test", 0)
                cv2.resizeWindow("Test", 1920, 1080)
                cv2.imshow('Test', ImgDraw)
                cv2.waitKey(0)
                if len(circlePosNew[:, 0]) == detectNum:
                    print('检测成功')
                    break
            # # 图示显示
            # cv2.namedWindow("Test", 0)
            # cv2.resizeWindow("Test", 1920, 1080)
            # cv2.imshow('Test', ImgDraw)
            # cv2.waitKey(0)
            # 从坐到右排序
            index = np.lexsort((circlePosNew[:, 2], circlePosNew[:, 1], circlePosNew[:, 0]))
            circlePosNew = circlePosNew[index]
            # 计算像素距离
            dis0 = pd.getDis2D(circlePos[0, 0], circlePos[0, 1], circlePosNew[0, 0], circlePosNew[0, 1])
            dis1 = pd.getDis2D(circlePos[1, 0], circlePos[1, 1], circlePosNew[1, 0], circlePosNew[1, 1])
            dis2 = pd.getDis2D(circlePos[2, 0], circlePos[2, 1], circlePosNew[2, 0], circlePosNew[2, 1])
            dis3 = pd.getDis2D(circlePos[3, 0], circlePos[3, 1], circlePosNew[3, 0], circlePosNew[3, 1])
            meanDis = (dis0 + dis1 + dis2 + dis3) / 4
            # 计算实际距离
            realDis = math.sqrt(dy ** 2 + dz ** 2)
            meanDis = realDis * 1000 / meanDis
            IterDis = IterDis + meanDis
            print('(', i + 1, ',', j + 1, ')位置, 平均比例', meanDis, ' mm/pixel')

print('平均比例：', IterDis / 8, ' mm/pixel')
# 机械臂回零
pd.getUR10EMove(IPRob, PortNumber, 0.315, 0.115, 0.680, 1.760, 0.729, 1.760)
