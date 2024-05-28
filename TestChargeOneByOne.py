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
# 前期标定尺度比例
dReal = 25
dCircle = 432.1439
dRuler = 0.05696312634314870000
# 前期标定工具中心偏移
# dx = 529.48 - 397.36
# dy = 98.03 - 94.84
# dx = 507.28-374.46
# dy = 136.67-134.36
dx = 493.19 - 360.92
dy = 79.84 - 76.32 - 0.5
# 回到初始位置
pd.getUR10EMove(IPRob, PortNumber, 0.385, 0.09, 0.7, 2.902, 1.202, 0)
# 相机标定参数
Intrinsic = np.array([[1102.88782596463, 0, 1941.87744425955],
                      [0, 1102.71207996845, 1048.10843897160],
                      [0, 0, 1]])
Distortion = np.array([-0.00681736444841834, 0.000426087049750034, 0, 0])
# 读图
Img = pd.getSingleImageByBasler()
Img = cv2.undistort(Img, Intrinsic, Distortion)
ImgGray = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
height, width = np.shape(ImgGray)
# Mask制作
Mask = np.zeros((height, width))
Mask[height / 2 - 100:height / 2 + 100, width / 2 - 100:width / 2 + 100] = 1
# 检测M4螺纹孔
count = 0
while 1:
    count = count + 1
    print('第', count, '次检测')
    Img = pd.getSingleImageByBasler()
    circlePos, ImgDraw = pd.getCirclePosEDWithRadiusEstimate(Img, 1, 50, 10, 20, 25 / 431.83822607171555, 1.5, 2)
    if len(circlePos[:, 0]) == 4:
        PosNow = pd.getUR10EPose(IPRob, PortNumber)
        print('检测成功')
        break

print(circlePos)
cv2.namedWindow("Test", 0)
cv2.resizeWindow("Test", 1920, 1080)
cv2.imshow('Test', ImgDraw)
cv2.waitKey(0)
cv2.imwrite("TestImage.bmp", Img)
# 控制机械臂到位
num = len(circlePos[:, 0])
Move = np.zeros((num, 2))
MoveC = np.zeros((num, 2))
index = np.lexsort((circlePos[:, 2], circlePos[:, 1], circlePos[:, 0]))
circlePos = circlePos[index]
for i in range(num):
    # print(i)
    # print(-(i[1] - height / 2) * dRuler + dx)
    Move[i, 0] = -(circlePos[i][1] - height / 2) * dRuler + dx
    Move[i, 1] = -(circlePos[i][0] - width / 2) * dRuler + dy
    print("第", i + 1, "个孔位综合位移x：", -(circlePos[i][1] - height / 2) * dRuler + dx)
    print("第", i + 1, "个孔位综合位移y：", -(circlePos[i][0] - width / 2) * dRuler + dy)
    MoveC[i, 0] = -(circlePos[i][1] - height / 2) * dRuler
    MoveC[i, 1] = -(circlePos[i][0] - width / 2) * dRuler

# PosNow = pd.getUR10EPose(IPRob, PortNumber)
# 示教移动位置
# pd.getUR10EMove(IPRob, 30003, PosNow[0] + Move[i, 0] / 1000, PosNow[1] + Move[i, 1] / 1000, 0.6, 2.902, 1.202, 0)
for i in range(num):
    print('移动至目标孔', i + 1)
    # 移动目标孔至光心
    pd.getUR10EMove(IPRob, 30003, PosNow[0] + MoveC[i, 0] / 1000, PosNow[1] + MoveC[i, 1] / 1000, 0.700, 2.902, 1.202,
                    0)
    Img = pd.getSingleImageByBasler()
    Img = cv2.undistort(Img, Intrinsic, Distortion)
    ImgGray = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
    circlePos, ImgDraw = pd.getCirclePosEDWithRadiusEstimate(Img, 1, 50, 10, 20, 25 / 431.83822607171555, 1.5, 2)
    if len(circlePos[:, 0]) == 1:
        PosNow = pd.getUR10EPose(IPRob, PortNumber)
        print('检测成功')
        break

    # M4 高度
    pd.getUR10EMove(IPRob, 30003, PosNow[0] + Move[i, 0] / 1000, PosNow[1] + Move[i, 1] / 1000, 0.600, 2.902, 1.202, 0)
    pd.getUR10EMove(IPRob, 30003, PosNow[0] + Move[i, 0] / 1000, PosNow[1] + Move[i, 1] / 1000, 0.516, 2.902, 1.202, 0)
    sleep(10)
    pd.getUR10EMove(IPRob, 30003, PosNow[0] + Move[i, 0] / 1000, PosNow[1] + Move[i, 1] / 1000, 0.600, 2.902, 1.202, 0)

print('完成自动拧钉,进行复位：')
pd.getUR10EMove(IPRob, PortNumber, 0.385, 0.09, 0.7, 2.902, 1.202, 0)
