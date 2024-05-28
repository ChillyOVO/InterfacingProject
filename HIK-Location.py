import cv2
import math
import random
import socket
import numpy as np
import PDDTVisionToolBox as pd
from time import sleep

# 设定ur机械臂IP地址及端口号
IPRob = "192.168.0.101"
PortNumberRob = 30003
# 回到初始位置
# PosNow0 = pd.getUR10EPose(IPRob, PortNumberRob)
pd.getUR10EMove(IPRob, PortNumberRob, 0.299, 0.139, 0.881, 2.902, 1.202, 0)
# sleep(10)
# 设定海康相机IP地址及端口号
IPCam = "192.168.0.69"
PortNumberCam = 8192
# 创建客户端读取SC MVS数据
Data = pd.getDataFromHikSmartCameraByTCP(IPCam, PortNumberCam, 9)
radiusR = Data[0, 0]
numTotal = 4
circlePos = np.zeros((numTotal, 2))
for i in range(numTotal):
    circlePos[i, 0] = Data[2 * i + 1, 0]
    circlePos[i, 1] = Data[2 * i + 2, 0]
# 末端到相机旋转矩阵
RotEndToCamera = np.array([[-0.707, -0.707, 0],
                           [0.707, -0.707, 0],
                           [0, 0, 1]])
# 海康相机尺度变换
# dReal = 85.23
# dPixel = 864.68
dReal = 85.52
dPixel = radiusR
Ruler = dReal / dPixel
# 海康图像尺寸
Width = 2368
Height = 1760
# 读取当前位姿
PosNow0 = pd.getUR10EPose(IPRob, PortNumberRob)

# dX = 660
# dY = 1000
# 前期示教偏差
dx = -(194.05 - 161.94)
dy = -(811.08 - 703.07)
# dx = -(210.57 - 179.81)
# dy = -(811.15 - 720.07)
# 计算末端旋转矩阵
RotVec = np.array([PosNow0[3], PosNow0[4], PosNow0[5]])
RotBaseToEnd = pd.getRotVec2RotMAT(RotVec)

# 手动输入孔位置
# circlePos = np.array([[posX, posY],
#                       [0, 0],
#                       [0, 0],
#                       [0, 0]])
# 手动标定前进距离
DisForwardScrew = 0


# 定义移动函数
def MoveScrew(CirclePos, DisForward, PosNow):
    MoveX = (CirclePos[0] - Width / 2) * Ruler + dx
    # print(MoveX)
    MoveY = (CirclePos[1] - Height / 2) * Ruler + dy
    # print(MoveY)
    Move1 = np.array([[MoveX], [MoveY], [0]])
    Move1 = Move1 / 1000
    MoveCam = RotBaseToEnd @ RotEndToCamera @ Move1
    pd.getUR10EMove(IPRob, PortNumberRob, PosNow[0] + MoveCam[0], PosNow[1] + MoveCam[1], PosNow[2] + MoveCam[2],
                    PosNow[3], PosNow[4], PosNow[5])
    Move2 = np.array([[0], [0], [DisForward]])
    Move2 = Move2 / 1000
    MoveGri = RotBaseToEnd @ Move2
    # PosNow = pd.getUR10EPose(IPRob, PortNumber)
    pd.getUR10EMove(IPRob, PortNumberRob, PosNow[0] + MoveGri[0] + MoveCam[0], PosNow[1] + MoveGri[1] + MoveCam[1],
                    PosNow[2] + MoveGri[2] + MoveCam[2], PosNow[3], PosNow[4], PosNow[5])
    sleep(10)
    pd.getUR10EMove(IPRob, PortNumberRob, PosNow[0] + MoveCam[0], PosNow[1] + MoveCam[1], PosNow[2] + MoveCam[2],
                    PosNow[3], PosNow[4], PosNow[5])


# 螺杆示教移动
# print(circlePos[0, :])
# MoveScrew(circlePos[0, :], 210, PosNow0)

# 进行移动
num = len(circlePos[:, 0])
for i in range(num):
    MoveScrew(circlePos[i, :], DisForward=340, PosNow=PosNow0)

# 测试移动
# MoveScrew(circlePos[0, :], DisForward=340, PosNow=PosNow0)

# 位置复位
print('完成自动拧钉,进行复位：')
pd.getUR10EMove(IPRob, PortNumberRob, 0.299, 0.139, 0.881, 2.902, 1.202, 0)
