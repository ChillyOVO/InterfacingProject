import cv2
import math
import random
import socket
import numpy as np
import PDDTVisionToolBox as pd
from time import sleep

# 通讯设置
# 设定ur机械臂IP地址及端口号
IPRob = "192.168.0.101"
PortNumberRob = 30003
# 设定海康相机IP地址及端口号
IPCam = "192.168.0.66"
PortNumberCam = 8192
# 通讯触发信号
msgStart = '123'

# 相机参数
# 海康相机图像尺寸
Width = 2368
Height = 1760

# 标定参数
# 前期标定特征尺寸
# 控制盒
# dReal = 74.13
# 控制盒1
# dReal = 62.03
# 天线
dReal = 55.00
# 示教参数-电批相对相机偏移，相机坐标系下
dx = -(194.50 - 161.75)
dy = -(811.32 - 703.52)
# 插入深度
disForward = 210
# 前期标定初始位姿
# Pos0 = np.array([[0.321, -0.227, 0.731, -2.200, -0.909, -0.378]])
# 末端到相机旋转矩阵
RotEndToCamera = np.array([[-0.707, -0.707, 0],
                           [0.707, -0.707, 0],
                           [0, 0, 1]])

# 移动机械臂到检测位置并获得孔位信息
# 移动机械臂到检测零位
pd.getUR10EMove(IPRob, PortNumberRob, 0.38908, 0.09743, 0.88939, -2.90201, -1.20198, 0)
# 读取当前位姿
PosNow = pd.getUR10EPose(IPRob, PortNumberRob)
# 切换Hik相机方案至控制盒检测
# pd.getHikSwitchPlanByTCP(IPCam, PortNumberCam, 'switch', 'Kongzhihe')
# pd.getHikSwitchPlanByTCP(IPCam, PortNumberCam, 'switch', 'Kongzhihe1')
pd.getHikSwitchPlanByTCP(IPCam, PortNumberCam, 'switch', 'Tianxian')
# 检测孔位
print('正在从海康智能相机读取检测结果...')
# 调用Hik智能相机检测，若未检测到会一直循环检测，此处应添加调整或报警
# 检测结果数量: 比例尺（1）+孔（2）×孔数量（4）
Data = pd.getDataFromHikSmartCameraByTCP(IPCam, PortNumberCam, msgStart, 9)
dPixel = Data[0, 0]
numTotal = 4
circlePos = np.zeros((numTotal, 2))
for i in range(numTotal):
    circlePos[i, 0] = Data[2 * i + 1, 0]
    circlePos[i, 1] = Data[2 * i + 2, 0]
Ruler = dReal / dPixel

# 控制移动
# 控制移动
pd.getRobertInsert(IPRob, PortNumberRob, Width, Height, dx, dy, RotEndToCamera, Ruler, PosNow, circlePos, 210)
# 位置复位
print('完成自动拧钉,进行复位：')
# 回零位
pd.getUR10EMove(IPRob, PortNumberRob, 0.38908, 0.09743, 0.88939, -2.90201, -1.20198, 0)
