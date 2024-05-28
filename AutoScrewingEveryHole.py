import cv2
import numpy as np
import PDDTVisionToolBox as PD
import sys
from time import sleep

# 设定IP地址及端口号
IPRob = "192.168.0.101"
PortNumber = 30003
# 回到初始位置
PD.getUR10EMove(IPRob, PortNumber, 0.385, 0.09, 0.7, 2.902, 1.202, 0)
# 读图
Img = PD.getSingleImageByBasler()
# 前期标定尺度比例
dReal = 25
dCircle = 432.1439
dRuler = dReal / dCircle
# 前期标定工具中心偏移
dx = 529.48 - 397.36
dy = 98.03 - 94.84
# 检测圆孔位置，由于需要定制化排序等功能，故未调用PDDTToolBox
kernelRow = np.ones((3, 1), np.uint8)
kernelCol = np.ones((1, 3), np.uint8)
ImgGray = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
height, width = np.shape(ImgGray)
img = cv2.bilateralFilter(ImgGray, d=0, sigmaColor=15, sigmaSpace=15)
ImgBW = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
ImgBW = cv2.medianBlur(ImgBW, 5)
ImgBW = cv2.erode(ImgBW, kernelRow)
ImgBW = cv2.erode(ImgBW, kernelCol)
# cv2.namedWindow("Test", 0)
# cv2.resizeWindow("Test", 1920, 1080)
# cv2.imshow('Test', ImgBW)
# cv2.waitKey(0)
# M4
circles = cv2.HoughCircles(ImgBW, cv2.HOUGH_GRADIENT, 3.001, 300, param1=50, param2=50, minRadius=32, maxRadius=38)
# M6
# circles = cv2.HoughCircles(ImgBW, cv2.HOUGH_GRADIENT, 3.001, 300, param1=50, param2=50, minRadius=53, maxRadius=58)
num = len(circles[0, :])
# num = len(circles)
Move = np.zeros((num, 2))
count = 0
for i in circles[0, :]:
    # print(i)
    # print(-(i[1] - height / 2) * dRuler + dx)
    Move[count, 0] = -(i[1] - height / 2) * dRuler + dx
    Move[count, 1] = -(i[0] - width / 2) * dRuler + dy
    count = count + 1
    print("第", count, "个孔位综合位移x：", -(i[1] - height / 2) * dRuler + dx)
    print("第", count, "个孔位综合位移y：", -(i[0] - width / 2) * dRuler + dy)
    CenterName = str(count)
    CenterPos = "(" + str(i[0]) + "," + str(i[1]) + ")"
    i = np.uint16(np.around(i))
    # 画出来圆的边界
    cv2.circle(Img, (i[0], i[1]), i[2], (0, 0, 255), 2)
    # 画出来圆心
    cv2.circle(Img, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv2.putText(Img, CenterName, (i[0], i[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(Img, CenterPos, (i[0], i[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.namedWindow("Test", 0)
cv2.resizeWindow("Test", 1920, 1080)
cv2.imshow('Test', Img)
cv2.waitKey(0)
cv2.imwrite("TestImage2.bmp", Img)

PosNow = PD.getUR10EPose(IPRob, PortNumber)
for i in range(num):
    print('移动至目标孔', i+1)
    # M6 高度
    # PD.getUR10EMove(IPRob, 30003, PosNow[0] + Move[i, 0] / 1000, PosNow[1] + Move[i, 1] / 1000, 0.55, 2.902, 1.202, 0)
    # PD.getUR10EMove(IPRob, 30003, PosNow[0] + Move[i, 0] / 1000, PosNow[1] + Move[i, 1] / 1000, 0.525, 2.902, 1.202, 0)
    # sleep(3)
    # PD.getUR10EMove(IPRob, 30003, PosNow[0] + Move[i, 0] / 1000, PosNow[1] + Move[i, 1] / 1000, 0.55, 2.902, 1.202, 0)
    # M4 高度
    PD.getUR10EMove(IPRob, 30003, PosNow[0] + Move[i, 0] / 1000, PosNow[1] + Move[i, 1] / 1000, 0.600, 2.902, 1.202, 0)
    PD.getUR10EMove(IPRob, 30003, PosNow[0] + Move[i, 0] / 1000, PosNow[1] + Move[i, 1] / 1000, 0.535, 2.902, 1.202, 0)
    sleep(3)
    PD.getUR10EMove(IPRob, 30003, PosNow[0] + Move[i, 0] / 1000, PosNow[1] + Move[i, 1] / 1000, 0.600, 2.902, 1.202, 0)

print('完成自动拧钉,进行复位：')
PD.getUR10EMove(IPRob, PortNumber, 0.385, 0.09, 0.7, 2.902, 1.202, 0)
