import math

import cv2
import PDDTVisionToolBox as pd
import numpy as np

Intrinsic = np.array([[4295.82123518699, 0, 1947.32510469332],
                      [0, 4297.26536461942, 1050.71392654110],
                      [0, 0, 1]])
Distortion = np.array([-0.109780715530354, 0.133569098996419, 0, 0])
# row = 6
# col = 7
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# # 读图
# Img = pd.getSingleImageByBasler()
# gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
# # 初始化
# params = cv2.SimpleBlobDetector_Params()
# params.maxArea = 10000
# params.minArea = 1000
# params.minDistBetweenBlobs = 10
# blobDetector = cv2.SimpleBlobDetector_create(params)
# ret, corners = cv2.findCirclesGrid(gray, (row, col), cv2.CALIB_CB_SYMMETRIC_GRID, blobDetector, None)
# # print(corners)
# # cv2.cornerSubPix(gray, corners, (row, col), (-1, -1), criteria)
# cv2.drawChessboardCorners(Img, (row, col), corners, corners is not None)
Img = pd.getSingleImageByBasler()
Img = cv2.undistort(Img, Intrinsic, Distortion)
# Img, corners = pd.getCirclesGridsPose(Img, GridRow=6, GridCol=7, CircleMinArea=1000, CircleMaxArea=10000,
#                                       CircleDistance=10, SymmetricEnable=1)
# cv2.namedWindow("Test", 0)
# cv2.resizeWindow("Test", 1920, 1080)
# cv2.imshow('Test', Img)
# cv2.waitKey(0)
Gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
Edge = cv2.Canny(Gray, 100, 200, apertureSize=3)
cv2.namedWindow("Test", 0)
cv2.resizeWindow("Test", 1920, 1080)
cv2.imshow('Test', Edge)
cv2.waitKey(0)
contours, hierarchy = cv2.findContours(Edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
num = len(contours)
# print(num)
# print(np.shape(Img))
width, height, channel = np.shape(Img)
for i in range(num):
    if 150  < len(contours[i]) < 100000:
        # ellipse = cv2.fitEllipse(contours)
        # cv2.drawContours(Img, contours[i], -1, (0, 255, 0), 3)
        ellipse = cv2.fitEllipse(contours[i])
        if 0.95 < ellipse[1][0] / ellipse[1][1] < 1.05:
            cv2.ellipse(Img, ellipse, (255, 0, 255), 2)
            if ellipse[1][0] > ellipse[1][1]:
                theta = math.acos(ellipse[1][1] / ellipse[1][0]) * 180 / np.pi
                # print(theta)
            else:
                theta = math.acos(ellipse[1][0] / ellipse[1][1]) * 180 / np.pi
                # print(theta)
            # if theta > 10 :
            #     print( ellipse[1][0], ellipse[1][1])
            cv2.circle(Img, (round(ellipse[0][0]), round(ellipse[0][1])), 4, (20, 0, 255), 2)
            Angle = '%.2f' % theta
            cv2.putText(Img, Angle, (round(ellipse[0][0]), round(ellipse[0][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
            Pos = '(' + str(ellipse[0][0] - height / 2) + ',' + str(ellipse[0][1] - width / 2) + ')'
            DisX = 'X Dis = ' + str(round(ellipse[0][1] - width / 2, 3)) + ' pixels'
            DisY = 'Y Dis = ' + str(round(ellipse[0][0] - height / 2, 3)) + ' pixels'
            cv2.putText(Img, DisX, (round(ellipse[0][0]), round(ellipse[0][1] + 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
            cv2.putText(Img, DisY, (round(ellipse[0][0]), round(ellipse[0][1] + 60)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
    # else:
    #     print(len((contours[i])))
cv2.namedWindow("Test", 0)
cv2.resizeWindow("Test", 1920, 1080)
cv2.imshow('Test', Img)
cv2.waitKey(0)
cv2.imwrite('./CirclesGrids.bmp', Img)
