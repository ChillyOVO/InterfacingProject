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
# M6
# dRuler = 0.06041976007364834
# M4
dRuler = 0.0594430595424421
# 前期标定工具中心偏移
dx = 0
dy = 0
# dx = 493.19 - 360.92
# dy = 79.84 - 76.32 - 0.6
ratio = np.array([4.054e-07, 0.05825])
# 回到初始位置
pd.getUR10EMove(IPRob, PortNumber, 0.385, 0.09, 0.7, 2.902, 1.202, 0)
# 相机标定参数
Intrinsic = np.array([[2988.40434044024, 0, 1934.71209993405],
                      [0, 2987.99338701425, 1036.58326580770],
                      [0, 0, 1]])
Distortion = np.array([-0.0501552424699594, 0.0238122396566109, 0, 0])
# 读图
# Img = cv2.imread("./circlrtest6.bmp")
Img = pd.getSingleImageByBasler()
Img = cv2.undistort(Img, Intrinsic, Distortion)
ImgGray = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
height, width = np.shape(ImgGray)
# 检测M4螺纹孔
count = 0
while 1:
    count = count + 1
    print('第', count, '次检测')
    # Img = pd.getSingleImageByBasler()
    circlePos, ImgDraw = pd.getCirclePosEDWithRadiusEstimate(Img, 1, 50, 10, 30, dRuler, 1.5, 2)
    if len(circlePos[:, 0]) == 1:
        PosNow = pd.getUR10EPose(IPRob, PortNumber)
        print('检测成功')
        break
    else:
        Img = pd.getSingleImageByBasler()

# 检测M6孔位
# while 1:
#     count = count + 1
#     print('第', count, '次检测')
#     # Img = pd.getSingleImageByBasler()
#     circlePos, ImgDraw = pd.getCirclePosEDWithRadiusEstimate(Img, 1, 50, 10, 40, ratio, 2.5, 4)
#     if len(circlePos[:, 0]) == 4:
#         PosNow = pd.getUR10EPose(IPRob, PortNumber)
#         print('检测成功')
#         break
#     else:
#         Img = pd.getSingleImageByBasler()

cv2.namedWindow("Test", 0)
cv2.resizeWindow("Test", 1920, 1080)
cv2.imshow('Test', ImgDraw)
cv2.waitKey(0)
cv2.imwrite("TestImage.bmp", ImgDraw)

# 控制机械臂到位
num = len(circlePos[:, 0])
Move = np.zeros((num, 2))
Move1 = Move
index = np.lexsort((circlePos[:, 2], circlePos[:, 1], circlePos[:, 0]))
circlePos = circlePos[index]
for i in range(num):
    # 原始算法
    # Move[i, 0] = -(circlePos[i][1] - height / 2) * dRuler + dx
    # Move[i, 1] = -(circlePos[i][0] - width / 2) * dRuler + dy
    # print("第", i + 1, "个孔位综合位移x：", -(circlePos[i][1] - height / 2) * dRuler + dx)
    # print("第", i + 1, "个孔位综合位移y：", -(circlePos[i][0] - width / 2) * dRuler + dy)
    # 积分校正算法
    Move[i, 0] = - (
            (circlePos[i][1] - height / 2) ** 2 * ratio[0] / 2 + (circlePos[i][1] - height / 2) * ratio[1]) + dx
    Move[i, 1] = - ((circlePos[i][0] - width / 2) ** 2 * ratio[0] / 2 + (circlePos[i][0] - width / 2) * ratio[1]) + dy
    print("第", i + 1, "个孔位修正位移x：", Move[i, 0])
    print("第", i + 1, "个孔位修正位移y：", Move[i, 1])

pd.getUR10EMove(IPRob, 30003, PosNow[0] + Move[0, 0] / 1000, PosNow[1] + Move[0, 1] / 1000, 0.600, 2.902, 1.202, 0)
# pd.getUR10EMove(IPRob, 30003, PosNow[0] + Move[0, 0] / 1000, PosNow[1] + Move[0, 1] / 1000, 0.516, 2.902, 1.202, 0)
cv2.namedWindow("Test", 0)
cv2.resizeWindow("Test", 1920, 1080)
cv2.imshow('Test', ImgDraw)
cv2.waitKey(0)
cv2.imwrite("TestImage.bmp", ImgDraw)

    # sleep(10)
# PosNow = pd.getUR10EPose(IPRob, PortNumber)
# 示教移动位置
# pd.getUR10EMove(IPRob, 30003, PosNow[0] + Move[i, 0] / 1000, PosNow[1] + Move[i, 1] / 1000, 0.6, 2.902, 1.202, 0)
# for i in range(num):
#     print('移动至目标孔', i + 1)
#     # M6 高度
#     pd.getUR10EMove(IPRob, 30003, PosNow[0] + Move[i, 0] / 1000, PosNow[1] + Move[i, 1] / 1000, 0.55, 2.902, 1.202, 0)
#     pd.getUR10EMove(IPRob, 30003, PosNow[0] + Move[i, 0] / 1000, PosNow[1] + Move[i, 1] / 1000, 0.525, 2.902, 1.202, 0)
#     sleep(3)
#     pd.getUR10EMove(IPRob, 30003, PosNow[0] + Move[i, 0] / 1000, PosNow[1] + Move[i, 1] / 1000, 0.55, 2.902, 1.202, 0)
#     # M4 高度
#     # pd.getUR10EMove(IPRob, 30003, PosNow[0] + Move[i, 0] / 1000, PosNow[1] + Move[i, 1] / 1000, 0.600, 2.902, 1.202, 0)
#     # pd.getUR10EMove(IPRob, 30003, PosNow[0] + Move[i, 0] / 1000, PosNow[1] + Move[i, 1] / 1000, 0.516, 2.902, 1.202, 0)
#     # sleep(10)
#     # pd.getUR10EMove(IPRob, 30003, PosNow[0] + Move[i, 0] / 1000, PosNow[1] + Move[i, 1] / 1000, 0.600, 2.902, 1.202, 0)
#
# print('完成自动拧钉,进行复位：')
# pd.getUR10EMove(IPRob, PortNumber, 0.385, 0.09, 0.7, 2.902, 1.202, 0)
