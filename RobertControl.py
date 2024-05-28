import cv2
import math
import numpy as np
import PDDTVisionToolBox as pd

# 设定ur机械臂IP地址及端口号
IPRob = "192.168.0.101"
PortNumberRob = 30003
# 前期标定相机到末端变换
RotEndToCamera = np.array([[-0.707, -0.707, 0],
                           [0.707, -0.707, 0],
                           [0, 0, 1]])
# 获取当前位姿
PosNow = pd.getUR10EPose(IPRob, PortNumberRob)
# 0.301581   -0.19903351  0.76024267 -2.20010885 -0.90976033 -0.37808463
# [ 0.32098934  0.39025267  0.82111195  2.20801674  0.91506302 -0.3790322 ]
# 计算变换矩阵
RotVec = np.array([PosNow[3], PosNow[4], PosNow[5]])
RotBaseToEnd = pd.getRotVec2RotMAT(RotVec)
# 设定移动距离-基于相机坐标系
Move = np.array([[-10],
                 [0],
                 [0]])
# 计算机械臂下变换
Move = Move / 1000
MoveCam = RotBaseToEnd @ RotEndToCamera @ Move
# 移动机械臂
# pd.getUR10EMove(IPRob, PortNumberRob, PosNow[0] + MoveCam[0], PosNow[1] + MoveCam[1], PosNow[2] + MoveCam[2], PosNow[3],
#                 PosNow[4], PosNow[5])
# 移动完成，结束
