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
PosNow0 = pd.getUR10EPose(IPRob, PortNumberRob)
# 平放零位
# pd.getUR10EMove(IPRob, PortNumberRob, 0.299, 0.139, 0.881, 2.902, 1.202, 0)
# 竖置零位
# pd.getUR10EMove(IPRob, PortNumberRob, 0.27213127, 0.08579944, 0.69573254, 1.75975888, 0.72904033, 1.7599637)
# 周圈零位1
# pd.getUR10EMove(IPRob, PortNumberRob, 3.059e-01, 9.744e-02, 8.894e-01, -2.902, -1.202, 0)
# 周圈零位2
# pd.getUR10EMove(IPRob, PortNumberRob, 0.321, -0.227, 0.731, -2.200, -0.909, -0.378)
# 周圈零位3
pd.getUR10EMove(IPRob, PortNumberRob, 0.310, 0.418, 0.792, 2.208, 0.914, -0.379)
# sleep(10)
# 设定海康相机IP地址及端口号
IPCam = "192.168.0.18"
PortNumberCam = 8192
# 创建客户端读取SC MVS数据
print('正在从海康智能相机读取检测结果...')
msgStart = '123'
# 数据数量 = 比例尺 + 中心点坐标X数量 + 中心点坐标Y数量
Data = pd.getDataFromHikSmartCameraByTCP(IPCam, PortNumberCam, msgStart, 3)
# print(Data)
radiusR = Data[0, 0]
numTotal = 1
circlePos = np.zeros((numTotal, 2))
for i in range(numTotal):
    circlePos[i, 0] = Data[2 * i + 1, 0]
    circlePos[i, 1] = Data[2 * i + 2, 0]
# 末端到相机旋转矩阵
RotEndToCamera = np.array([[-0.707, -0.707, 0],
                           [0.707, -0.707, 0],
                           [0, 0, 1]])
# 海康相机尺度变换
# 平放件
# dReal = 85.52
# 竖置件
# dReal = 85.02
# 周圈件
dReal = 50.00
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
# 平放示教
# dx = -(194.05 - 161.94)
# dy = -(811.08 - 703.07)
# 竖置示教
dx = -(194.50 - 161.75)
dy = -(811.32 - 703.52)
# 计算末端旋转矩阵
RotVec = np.array([PosNow0[3], PosNow0[4], PosNow0[5]])
RotBaseToEnd = pd.getRotVec2RotMAT(RotVec)


# 手动输入孔位置
# circlePos = np.array([[posX, posY],
#                       [0, 0],
#                       [0, 0],
#                       [0, 0]])
# 手动标定前进距离
# DisForwardScrew = 0


# 定义移动函数
def MoveScrew(CirclePos, DisForward, PosNow):
    # 平面内移动
    MoveX = (CirclePos[0] - Width / 2) * Ruler + dx
    MoveY = (CirclePos[1] - Height / 2) * Ruler + dy
    Move1 = np.array([[MoveX], [MoveY], [0]])
    Move1 = Move1 / 1000
    MoveCam = RotBaseToEnd @ RotEndToCamera @ Move1
    pd.getUR10EMove(IPRob, PortNumberRob, PosNow[0] + MoveCam[0], PosNow[1] + MoveCam[1], PosNow[2] + MoveCam[2],
                    PosNow[3], PosNow[4], PosNow[5])
    # sleep(20)  # 示教使用
    # 垂直方向移动
    Move2 = np.array([[0], [0], [DisForward]])
    Move2 = Move2 / 1000
    MoveGri = RotBaseToEnd @ Move2
    pd.getUR10EMove(IPRob, PortNumberRob, PosNow[0] + MoveGri[0] + MoveCam[0], PosNow[1] + MoveGri[1] + MoveCam[1],
                    PosNow[2] + MoveGri[2] + MoveCam[2], PosNow[3], PosNow[4], PosNow[5])
    sleep(10)
    # 回示教平面
    pd.getUR10EMove(IPRob, PortNumberRob, PosNow[0] + MoveCam[0], PosNow[1] + MoveCam[1], PosNow[2] + MoveCam[2],
                    PosNow[3], PosNow[4], PosNow[5])


# 螺杆示教移动
# print(circlePos[0, :])
# MoveScrew(circlePos[0, :], 210, PosNow0)

# 进行移动
# 竖置 D = 196
# 周圈 D = 200
num = len(circlePos[:, 0])
for i in range(num):
    MoveScrew(circlePos[i, :], DisForward=220, PosNow=PosNow0)

# 螺杆标定移动
# MoveScrew(circlePos[0, :], DisForward=0, PosNow=PosNow0)

# 位置复位
print('完成自动拧钉,进行复位：')
# 平放零位
# pd.getUR10EMove(IPRob, PortNumberRob, 0.299, 0.139, 0.881, 2.902, 1.202, 0)
# 竖置零位
# pd.getUR10EMove(IPRob, PortNumberRob, 0.27213127, 0.08579944, 0.69573254, 1.75975888, 0.72904033, 1.7599637)
# 周圈零位1
# pd.getUR10EMove(IPRob, PortNumberRob, 3.059e-01, 9.744e-02, 8.894e-01, -2.902, -1.202, 0)
# 周圈零位2
# pd.getUR10EMove(IPRob, PortNumberRob, 0.321, -0.227, 0.731, -2.200, -0.909, -0.378)
# 周圈零位3
pd.getUR10EMove(IPRob, PortNumberRob, 0.310, 0.418, 0.792, 2.208, 0.914, -0.379)
