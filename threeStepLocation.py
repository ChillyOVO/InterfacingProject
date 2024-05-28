import cv2
import math
import random
import numpy as np
import PDDTVisionToolBox as pd
from time import sleep

# 设定IP地址及端口号
IPRob = "192.168.0.101"
PortNumber = 30003
# 回到初始位置
pd.getUR10EMove(IPRob, PortNumber, 0.35242922, 0.27856543, 0.72727759, 1.80068357, 0.2989296, 1.49943926)
# 相机标定参数
Intrinsic = np.array([[2988.40434044024, 0, 1934.71209993405],
                      [0, 2987.99338701425, 1036.58326580770],
                      [0, 0, 1]])
Distortion = np.array([-0.0501552424699594, 0.0238122396566109, 0, 0])
# 手眼参数
# 末端到相机旋转矩阵
RotEndToCamera = np.array([[-0.707, -0.707, 0],
                           [0.707, -0.707, 0],
                           [0, 0, 1]])
# 读图
Img = pd.getSingleImageByBasler()
# 去畸变
Img = cv2.undistort(Img, Intrinsic, Distortion)
# 灰度化
ImgGray = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
# 获得图像高宽
Height, Width = np.shape(ImgGray)
# 初始尺度
dRuler = 0.05979791349972892
# 尺寸范围
minRadiusM6 = 2.5
maxRadiusM6 = 4
# 检测所有圆孔
# circlePos, ImgDraw = pd.getCirclePosEDWithRadiusEstimate(Img, 1, 50, 10, 20, dRuler,1, 5)
# cv2.namedWindow("Test", 0)
# cv2.resizeWindow("Test", 1920, 1080)
# cv2.imshow('Test', ImgDraw)
# cv2.waitKey(0)
# PosNow = pd.getUR10EPose(IPRob, PortNumber)
# YOLOX检测形心
# Img = pd.getSingleImageByBasler()
PosNow0 = pd.getUR10EPose(IPRob, PortNumber)
Pos = pd.getLocationByYOLOX(Img, Model_Type=0)

# 图示显示
# cv2.namedWindow("Test", 0)
# cv2.resizeWindow("Test", 1920, 1080)
# cv2.imshow('Test', ImgDraw)
# cv2.waitKey(0)
# 计算工件中心
# print(circlePos)
centerX = Pos[1]
centerY = Pos[0]
print('工件中心位置：', centerX, centerY)
# 预设工件大致M3孔相对位姿
dX = 660
dY = 1000
# 前期示教偏差
dx = -(210.57 - 179.81)
dy = -(811.15 - 720.07)
# 计算移动距离
Move = np.zeros((4, 2))
for i in range(2):
    for j in range(2):
        fi = math.pow(-1, i + 1)
        fj = math.pow(-1, j + 1)
        # print(fi, fj)
        Move[2 * i + j, 0] = (centerX + fi * dX - Height / 2) * dRuler
        Move[2 * i + j, 1] = (centerY + fj * dY - Width / 2) * dRuler
        # print("第", i + 1, "个孔位检测位移x：", -(centerX + fi * dX - Height / 2) * dRuler, 'mm')
        # print("第", i + 1, "个孔位检测位移y：", -(centerY + fj * dY - Width / 2) * dRuler, 'mm')
# 移动阶段
# 检测参数设定
minRadiusM3 = 0.6
maxRadiusM3 = 2.7
# 图像存储文件夹建立
fileName = 'Logs_of_Images'
childFileName = 'Point'
path0, pathList = pd.getDirPathOfLogsbYTime(fileName, childFileName, 4)
# 计算末端旋转矩阵
RotVec = np.array([PosNow0[3], PosNow0[4], PosNow0[5]])
RotBaseToEnd = pd.getRotVec2RotMAT(RotVec)
# 计算综合旋转矩阵
RotBaseToCamera = RotEndToCamera @ RotBaseToEnd
for i in range(4):
    print('移动至目标孔', i + 1)
    DisCam = np.array([[Move[i, 1] / 1000], [Move[i, 0] / 1000], [0]])
    MoveCam = RotBaseToEnd @ RotEndToCamera @ DisCam
    # M3 检测高度
    pd.getUR10EMove(IPRob, PortNumber, PosNow0[0] + MoveCam[0], PosNow0[1] + MoveCam[1], PosNow0[2] + MoveCam[2],
                    PosNow0[3], PosNow0[4], PosNow0[5])
    count = 0
    countN = 0
    Img = pd.getSingleImageByBasler()
    Img = cv2.undistort(Img, Intrinsic, Distortion)
    while 1:
        count = count + 1
        # print('第', countN * 10 + count, '次检测')
        # Img = pd.getSingleImageByBasler()
        circlePos, ImgDraw = pd.getCirclePosEDWithRadiusEstimate(Img, 1, 50, 10, 20, dRuler, minRadiusM3, maxRadiusM3)
        # cv2.namedWindow("Test", 0)
        # cv2.resizeWindow("Test", 1920, 1080)
        # cv2.imshow('Test', ImgDraw)
        # cv2.waitKey(0)
        # print(circlePos)
        num = len(circlePos[:, 0])
        # 检测是否满足条件，不满足条件则置空
        for j in range(num):
            if 1.8 > circlePos[j][2] * dRuler > 1.40 or 2.2 > circlePos[j][2] * dRuler > 2.05:
                Dis = pd.getDis2D(circlePos[j][1], circlePos[j][0], Height / 2, Width / 2)
                if Dis < 600:
                    # print('检测成功')
                    print('第', countN * 5 + count, '次检测,检测成功')
                    cv2.imwrite(pathList[i] + '/' + str(countN * 5 + count) + '.bmp', ImgDraw)
                    # PosNow = pd.getUR10EPose(IPRob, PortNumber)

                else:
                    circlePos[j].fill(0)
            else:
                circlePos[j].fill(0)
        # print(circlePos)
        # 删除零行
        circlePos = circlePos[~np.all(circlePos == 0, axis=1)]
        # print(circlePos)
        if len(circlePos[:, 0]) == 1:
            PosNow = pd.getUR10EPose(IPRob, PortNumber)
            # 图示显示
            # cv2.namedWindow("Test", 0)
            # cv2.resizeWindow("Test", 1920, 1080)
            # cv2.imshow('Test', ImgDraw)
            # cv2.waitKey(0)
            # print('检测成功')
            break
        else:
            sleep(0.01)
            if count < 5:
                Img = pd.getSingleImageByBasler()
                Img = cv2.undistort(Img, Intrinsic, Distortion)
            else:
                # count大于5，随机移动后进行检测
                countN = countN + 1
                count = 0
                p = np.random.random(2)
                px = p[0] * 0.01 - 0.005
                py = p[1] * 0.01 - 0.005
                pz = 0
                DisCamRand = np.array([[px], [py], [pz]])
                MoveCamRand = RotBaseToEnd @ RotEndToCamera @ DisCamRand
                PosNow = pd.getUR10EPose(IPRob, PortNumber)
                pd.getUR10EMove(IPRob, PortNumber, PosNow[0] + MoveCamRand[0], PosNow[1] + MoveCamRand[1],
                                PosNow[2] + MoveCamRand[2], PosNow[3], PosNow[4], PosNow[5])
                Img = pd.getSingleImageByBasler()
                Img = cv2.undistort(Img, Intrinsic, Distortion)
    # 图示显示
    # cv2.namedWindow("Test", 0)
    # cv2.resizeWindow("Test", 1920, 1080)
    # cv2.imshow('Test', ImgDraw)
    # cv2.waitKey(0)
    # 获取当前位移
    MoveX = (circlePos[0][0] - Width / 2) * dRuler + dx
    MoveY = (circlePos[0][1] - Height / 2) * dRuler + dy
    Move1 = np.array([[MoveX], [MoveY], [0]])
    Move1 = Move1 / 1000
    MoveCam = RotBaseToEnd @ RotEndToCamera @ Move1
    pd.getUR10EMove(IPRob, PortNumber, PosNow[0] + MoveCam[0], PosNow[1] + MoveCam[1], PosNow[2] + MoveCam[2],
                    PosNow[3], PosNow[4], PosNow[5])
    Move2 = np.array([[0], [0], [163]])
    Move2 = Move2 / 1000
    MoveGri = RotBaseToEnd @ Move2
    # PosNow = pd.getUR10EPose(IPRob, PortNumber)
    pd.getUR10EMove(IPRob, PortNumber, PosNow[0] + MoveGri[0] + MoveCam[0], PosNow[1] + MoveGri[1] + MoveCam[1],
                    PosNow[2] + MoveGri[2] + MoveCam[2], PosNow[3], PosNow[4], PosNow[5])
    sleep(10)
    pd.getUR10EMove(IPRob, PortNumber, PosNow[0] + MoveCam[0], PosNow[1] + MoveCam[1], PosNow[2] + MoveCam[2],
                    PosNow[3], PosNow[4], PosNow[5])
    # 移动
    # pd.getUR10EMove(IPRob, 30003, 0.315, PosNow[1] + MoveM3Y / 1000, PosNow[2] + MoveM3Z / 1000, 1.760, 0.729, 1.760)
    # pd.getUR10EMove(IPRob, 30003, 0.464, PosNow[1] + MoveM3Y / 1000, PosNow[2] + MoveM3Z / 1000, 1.760, 0.729, 1.760)

    # pd.getUR10EMove(IPRob, 30003, 0.315, PosNow[1] + MoveM3Y / 1000, PosNow[2] + MoveM3Z / 1000, 1.760, 0.729, 1.760)

print('完成自动拧钉,进行复位：')
pd.getUR10EMove(IPRob, PortNumber, 0.35242922, 0.27856543, 0.72727759, 1.80068357, 0.2989296, 1.49943926)
