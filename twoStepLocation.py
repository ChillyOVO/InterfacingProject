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
# pd.getUR10EMove(IPRob, PortNumber, 0.385, 0.09, 0.70, 2.902, 1.202, 0)
# 相机标定参数
Intrinsic = np.array([[2988.40434044024, 0, 1934.71209993405],
                      [0, 2987.99338701425, 1036.58326580770],
                      [0, 0, 1]])
Distortion = np.array([-0.0501552424699594, 0.0238122396566109, 0, 0])
# 读图
Img = pd.getSingleImageByBasler()
# 去畸变
Img = cv2.undistort(Img, Intrinsic, Distortion)
# 灰度化
ImgGray = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
# 获得图像高宽
Height, Width = np.shape(ImgGray)
# 初始尺度
dRuler = 0.060358418042918
# 尺寸范围
minRadiusM6 = 2.5
maxRadiusM6 = 4
# 检测所有圆孔
circlePos, ImgDraw = pd.getCirclePosEDWithRadiusEstimate(Img, 1, 50, 10, 50, dRuler,1, 5)
cv2.namedWindow("Test", 0)
cv2.resizeWindow("Test", 1920, 1080)
cv2.imshow('Test', ImgDraw)
cv2.waitKey(0)
PosNow = pd.getUR10EPose(IPRob, PortNumber)
# 检测M6圆孔定位
count = 0
while 1:
    count = count + 1
    print('第', count, '次检测')
    # Img = pd.getSingleImageByBasler()
    circlePos, ImgDraw = pd.getCirclePosEDWithRadiusEstimate(Img, 1, 50, 10, 50, dRuler, minRadiusM6, maxRadiusM6)
    if len(circlePos[:, 0]) == 4:
        PosNow0 = pd.getUR10EPose(IPRob, PortNumber)
        print('检测成功')
        break
    else:
        Img = pd.getSingleImageByBasler()
        Img = cv2.undistort(Img, Intrinsic, Distortion)

# 图示显示
cv2.namedWindow("Test", 0)
cv2.resizeWindow("Test", 1920, 1080)
cv2.imshow('Test', ImgDraw)
cv2.waitKey(0)
# 计算工件中心
print(circlePos)
centerX = np.average(circlePos[:, 1])
centerY = np.average(circlePos[:, 0])
print('工件中心位置：', centerX, centerY)
# 预设工件大致M3孔相对位姿
dX = 660
dY = 1000
# 前期示教偏差
dx = 474.27 - 383.64
dy = 121.87 - 91.37 +0.2
# 计算移动距离
Move = np.zeros((4, 2))
for i in range(2):
    for j in range(2):
        fi = math.pow(-1, i + 1)
        fj = math.pow(-1, j + 1)
        # print(fi, fj)
        Move[2 * i + j, 0] = -(centerX + fi * dX - Height / 2) * dRuler
        Move[2 * i + j, 1] = -(centerY + fj * dY - Width / 2) * dRuler
        # print("第", i + 1, "个孔位检测位移x：", -(centerX + fi * dX - Height / 2) * dRuler, 'mm')
        # print("第", i + 1, "个孔位检测位移y：", -(centerY + fj * dY - Width / 2) * dRuler, 'mm')
# 移动阶段
minRadiusM3 = 1.1
maxRadiusM3 = 1.9
for i in range(4):
    print('移动至目标孔', i + 1)
    # M3 检测高度
    pd.getUR10EMove(IPRob, 30003, PosNow0[0] + Move[i, 0] / 1000, PosNow0[1] + Move[i, 1] / 1000, 0.700, 2.902, 1.202,
                    0)
    count = 0
    countN = 0
    Img = pd.getSingleImageByBasler()
    Img = cv2.undistort(Img, Intrinsic, Distortion)
    while 1:
        count = count + 1
        print('第', countN * 10 + count, '次检测')
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
            if 1.7 > circlePos[j][2] * dRuler > 1.40:
                Dis = pd.getDis2D(circlePos[j][1], circlePos[j][0], Height / 2, Width / 2)
                if Dis < 600:
                    print('检测成功')
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
            sleep(0.02)
            if count < 10:
                Img = pd.getSingleImageByBasler()
                Img = cv2.undistort(Img, Intrinsic, Distortion)
            else:
                # count大于10，随机移动后进行检测
                countN = countN + 1
                count = 0
                p = np.random.random(2)
                px = p[0] * 0.01 - 0.005
                py = p[1] * 0.01 - 0.005
                pd.getUR10EMove(IPRob, 30003, PosNow0[0] + Move[i, 0] / 1000 + px, PosNow0[1] + Move[i, 1] / 1000 + py,
                                0.700, 2.902, 1.202, 0)
                Img = pd.getSingleImageByBasler()
                Img = cv2.undistort(Img, Intrinsic, Distortion)
    # 图示显示
    cv2.namedWindow("Test", 0)
    cv2.resizeWindow("Test", 1920, 1080)
    cv2.imshow('Test', ImgDraw)
    cv2.waitKey(0)
    # 获取当前位移
    MoveM3X = -(circlePos[0][1] - Height / 2) * dRuler + dx
    MoveM3Y = -(circlePos[0][0] - Width / 2) * dRuler + dy
    # 移动
    pd.getUR10EMove(IPRob, 30003, PosNow[0] + MoveM3X / 1000, PosNow[1] + MoveM3Y / 1000, 0.600, 2.902, 1.202, 0)
    pd.getUR10EMove(IPRob, 30003, PosNow[0] + MoveM3X / 1000, PosNow[1] + MoveM3Y / 1000, 0.545, 2.902, 1.202, 0)
    sleep(3)
    pd.getUR10EMove(IPRob, 30003, PosNow[0] + MoveM3X / 1000, PosNow[1] + MoveM3Y / 1000, 0.600, 2.902, 1.202, 0)

print('完成自动拧钉,进行复位：')
pd.getUR10EMove(IPRob, PortNumber, 0.385, 0.09, 0.7, 2.902, 1.202, 0)
