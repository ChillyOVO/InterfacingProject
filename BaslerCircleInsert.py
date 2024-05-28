import cv2
import math
import random
import socket
import numpy as np
import PDDTVisionToolBox as pd
from time import sleep

# 机械臂参数
# 机械臂地址
IPRob = "192.168.0.101"
# UR机械臂实时端口
PortNumberRob = 30003

# 相机参数
# Basler相机图像尺寸
Width = 3840
Height = 2160
# 相机参数
Intrinsic = np.array([[4328.86012909069, 0, 1949.42606614499],
                      [0, 4328.79448970381, 1051.75084844019],
                      [0, 0, 1]])
Distortion = np.array([-0.101628122319289, 0.0888390492812854, 0, 0])
# 末端到相机旋转矩阵
RotEndToCamera = np.array([[-0.707, -0.707, 0],
                           [0.707, -0.707, 0],
                           [0, 0, 1]])

# 示教参数
# 前期标定初始位姿
Pos0 = np.array([[0.341, -0.227, 0.731, -2.200, -0.909, -0.378],
                 [0.326, 0.097, 0.889, -2.902, -1.202, 0],
                 [0.340, 0.432, 0.777, 2.208, 0.913, -0.378]])
# Pos0 = np.array([[0.326, 0.097, 0.889, -2.902, -1.202, 0],
#                  [0.326, 0.097, 0.889, -2.902, -1.202, 0],
#                  [0.326, 0.097, 0.889, -2.902, -1.202, 0]])
# 示教相机-螺杆位置
dx = -(131.09 - 96.78)
dy = -(429.68 - 340.23)
# 插入深度
disForward = 200
# 测量目标尺寸
dReal = 6.7
# 比例尺初始设定
dRuler = 0.067

# 建立图像存储文件夹
fileName = 'Logs_of_Images'
childFileName = 'Point'
path0, pathList = pd.getDirPathOfLogsbYTime(fileName, childFileName, 3)

# 开始检测
# 回到初始位置
# PosNow0 = pd.getUR10EPose(IPRob, PortNumberRob)
for i in range(3):
    # 移动机械臂到检测零位
    pd.getUR10EMove(IPRob, PortNumberRob, Pos0[i, 0], Pos0[i, 1], Pos0[i, 2], Pos0[i, 3], Pos0[i, 4], Pos0[i, 5])
    # 检测孔位
    # 重置计数变量
    count = 0
    countN = 0
    # 获取图像并去畸变
    Img = pd.getSingleImageByBasler()
    Img = cv2.undistort(Img, Intrinsic, Distortion)
    # 检测孔尺寸设定
    minRadius = 2.8
    maxRadius = 3.9
    # 循环检测直到检测成功
    while 1:
        # 单次计数+1
        count = count + 1
        # 检测孔位并记录图像
        circlePos, ImgDraw = pd.getCirclePosEDWithRadiusEstimate(Img, 1, 50, 10, 20, dRuler, minRadius, maxRadius)
        # cv2.namedWindow("Test", 0)
        # cv2.resizeWindow("Test", 1920, 1080)
        # cv2.imshow('Test', ImgDraw)
        # cv2.waitKey(0)
        # print(circlePos)
        # 检测孔数量
        num = len(circlePos[:, 0])
        # 检测是否满足条件，不满足条件则置空
        for j in range(num):
            # 根据孔尺寸第二次判断
            if 0.9 * maxRadius > circlePos[j][2] * dRuler > 1.1 * minRadius:
                # 根据孔所在位置第三次判断
                Dis = pd.getDis2D(circlePos[j][1], circlePos[j][0], Height / 2, Width / 2)
                # print(Dis)
                if Dis < 500:
                    print('第', countN * 5 + count, '次检测,检测成功')
                    cv2.imwrite(pathList[i] + '/' + str(countN * 5 + count) + '.bmp', ImgDraw)
                else:
                    # 非目标孔位位置置零
                    circlePos[j].fill(0)
            else:
                # 非目标孔位位置置零
                circlePos[j].fill(0)
        # 删除零行
        circlePos = circlePos[~np.all(circlePos == 0, axis=1)]
        # 再次判断检测得到位置是否仅为1个
        if len(circlePos[:, 0]) == 1:
            # 检测成功,获取机械臂当前位姿
            PosNow = pd.getUR10EPose(IPRob, PortNumberRob)
            break
        else:
            # 短暂停顿缓冲
            sleep(0.01)
            if count < 5:
                # 5次为1轮小循环,小循环内不变换位姿
                Img = pd.getSingleImageByBasler()
                Img = cv2.undistort(Img, Intrinsic, Distortion)
            else:
                # 超过5次后,随机变换位姿再进行检测
                # 重置统计计数
                countN = countN + 1
                count = 0
                # 平面内随机移动参数计算
                p = np.random.random(2)
                px = p[0] * 0.01 - 0.005
                py = p[1] * 0.01 - 0.005
                pz = 0
                DisCamRand = np.array([[px], [py], [pz]])
                # 获取当前位姿
                PosNow = pd.getUR10EPose(IPRob, PortNumberRob)
                # 计算平面内移动量
                RotVec = np.array([PosNow[3], PosNow[4], PosNow[5]])
                RotBaseToEnd = pd.getRotVec2RotMAT(RotVec)
                MoveCamRand = RotBaseToEnd @ RotEndToCamera @ DisCamRand
                # 移动机械臂
                pd.getUR10EMove(IPRob, PortNumberRob, PosNow[0] + MoveCamRand[0], PosNow[1] + MoveCamRand[1],
                                PosNow[2] + MoveCamRand[2], PosNow[3], PosNow[4], PosNow[5])
                # 获取图像并去畸变
                Img = pd.getSingleImageByBasler()
                Img = cv2.undistort(Img, Intrinsic, Distortion)

    # 重新修正比例尺
    Ruler = dReal / (2 * circlePos[0, 2])
    # print(Ruler)
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
