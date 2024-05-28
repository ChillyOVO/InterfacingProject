import math
import cv2
import cProfile
import numpy as np
import PDDTVisionToolBox as pd
from numba import jit

# 通讯设置
# 设定ur机械臂IP地址及端口号
IPRob = "192.168.0.101"
PortNumberRob = 30003

# 确定相机内参
Intrinsic = np.array([[4295.82123518699, 0, 1947.32510469332],
                      [0, 4297.26536461942, 1050.71392654110],
                      [0, 0, 1]])
Distortion = np.array([-0.109780715530354, 0.133569098996419, 0, 0])

# 棋盘格位姿估计
# 建立标准模板点
ChessPoints = np.zeros((8 * 11, 3), np.float32)
ChessPoints[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
ChessPoints = 5 * ChessPoints
# print(ChessPoints)
# Img = cv2.imread('./Logs_of_Images/2024-04-22 10-34-39/Pic0/0.bmp')
Img = pd.getSingleImageByBasler()
gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
size = gray.shape[::-1]
ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
cv2.drawChessboardCorners(Img, (11, 8), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
cv2.namedWindow("Test", 0)
cv2.resizeWindow("Test", 1920, 1080)
cv2.imshow('Test', Img)
cv2.waitKey(0)
if ret:
    # obj_points.append(objp)
    corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001))
    # if [corners2]:
    #     img_points.append(corners2)
    # else:
    #     img_points.append(corners)
    _, rvec, tvec = cv2.solvePnP(ChessPoints, corners2, Intrinsic, Distortion)
    print(rvec)
    print(tvec)
    Rot = np.append(rvec.T, tvec.T, axis=0)
    print(Rot)
    RO = np.zeros((3, 3))
    RO[0, :] = tvec.T
    print(RO)

else:
    print('无法检测')

# # aruco码位姿估计
# import cv2.aruco as aruco
#
#
# # @jit(nopython=True)
# def main():
#     # 建立字典
#     aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
#     parameters = aruco.DetectorParameters_create()
#     # 读图预处理
#     Img = pd.getSingleImageByBasler()
#     gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
#     # gray = cv2.undistort(gray, Intrinsic, Distortion)
#     # Img = cv2.undistort(Img, Intrinsic, Distortion)
#     # 识别
#     corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
#     # 绘图
#     aruco.drawDetectedMarkers(Img, corners)
#     # cv2.namedWindow("Test", 0)
#     # cv2.resizeWindow("Test", 1920, 1080)
#     # cv2.imshow('Test', Img)
#     # cv2.waitKey(0)
#     # 位姿估计
#     rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, Intrinsic, Distortion)
#     # print(rvec)
#     # print(tvec)
#     for i in range(rvec.shape[0]):
#         cv2.drawFrameAxes(Img, Intrinsic, Distortion, rvec[i, :, :], tvec[i, :, :], 0.03)
#     cv2.namedWindow("Test", 0)
#     cv2.resizeWindow("Test", 1920, 1080)
#     cv2.imshow('Test', Img)
#     cv2.waitKey(0)
#
#
# cProfile.run("main()")
