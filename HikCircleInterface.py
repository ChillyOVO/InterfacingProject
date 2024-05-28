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
# 设定海康相机IP地址及端口号
IPCam = "192.168.0.46"
PortNumberCam = 8192
# 前期标定特征尺寸
dReal = 50.00
# 海康相机图像尺寸
Width = 2368
Height = 1760
# 示教参数-电批相对相机偏移，相机坐标系下
dx = -(194.50 - 161.75)
dy = -(811.32 - 703.52)
# 通讯触发信号
msgStart = '123'
# 插入深度
disForward = 210
# 前期标定初始位姿
Pos0 = np.array([[0.321, -0.227, 0.731, -2.200, -0.909, -0.378],
                 [0.306, 0.097, 0.889, -2.902, -1.202, 0],
                 [0.310, 0.418, 0.792, 2.208, 0.914, -0.379]])
# 末端到相机旋转矩阵
RotEndToCamera = np.array([[-0.707, -0.707, 0],
                           [0.707, -0.707, 0],
                           [0, 0, 1]])
# 回到初始位置
# PosNow0 = pd.getUR10EPose(IPRob, PortNumberRob)
for i in range(3):
    # 移动机械臂到检测零位
    pd.getUR10EMove(IPRob, PortNumberRob, Pos0[i, 0], Pos0[i, 1], Pos0[i, 2], Pos0[i, 3], Pos0[i, 4], Pos0[i, 5])
    # 检测孔位
    print('正在从海康智能相机读取检测结果...')
    # 调用Hik智能相机检测，若未检测到会一直循环检测，此处应添加调整或报警
    Data = pd.getDataFromHikSmartCameraByTCP(IPCam, PortNumberCam, msgStart, 3)
    dPixel = Data[0, 0]
    numTotal = 1
    circlePos = np.zeros((numTotal, 2))
    circlePos[0, 0] = Data[1, 0]
    circlePos[0, 1] = Data[2, 0]
    Ruler = dReal / dPixel
    # 读取当前位姿
    PosNow = pd.getUR10EPose(IPRob, PortNumberRob)
    # 计算末端旋转矩阵
    RotVec = np.array([PosNow[3], PosNow[4], PosNow[5]])
    RotBaseToEnd = pd.getRotVec2RotMAT(RotVec)
    # 平面内移动
    MoveX = (circlePos[0, 0] - Width / 2) * Ruler + dx
    MoveY = (circlePos[0, 1] - Height / 2) * Ruler + dy
    Move1 = np.array([[MoveX], [MoveY], [0]])
    Move1 = Move1 / 1000
    MoveCam = RotBaseToEnd @ RotEndToCamera @ Move1
    pd.getUR10EMove(IPRob, PortNumberRob, PosNow[0] + MoveCam[0], PosNow[1] + MoveCam[1], PosNow[2] + MoveCam[2],
                    PosNow[3], PosNow[4], PosNow[5])
    # sleep(20)  # 示教使用
    # 垂直方向移动
    Move2 = np.array([[0], [0], [disForward]])
    Move2 = Move2 / 1000
    MoveGri = RotBaseToEnd @ Move2
    pd.getUR10EMove(IPRob, PortNumberRob, PosNow[0] + MoveGri[0] + MoveCam[0], PosNow[1] + MoveGri[1] + MoveCam[1],
                    PosNow[2] + MoveGri[2] + MoveCam[2], PosNow[3], PosNow[4], PosNow[5])
    sleep(5)
    # 回示教平面
    pd.getUR10EMove(IPRob, PortNumberRob, PosNow[0] + MoveCam[0], PosNow[1] + MoveCam[1], PosNow[2] + MoveCam[2],
                    PosNow[3], PosNow[4], PosNow[5])

# 位置复位
print('完成自动拧钉,进行复位：')
# 周圈零位1
pd.getUR10EMove(IPRob, PortNumberRob, 3.059e-01, 9.744e-02, 8.894e-01, -2.902, -1.202, 0)
