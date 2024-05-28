import cv2
import numpy as np

L1 = [2415.81201171875, 1061.386474609375]
L2 = [2194.63623046875, 1039.75146484375]
L3 = [2500.107177734375, 995.1126098632812]
L4 = [2294.95068359375, 988.745361328125]

R1 = [1608.6446533203125, 1114.4833984375]
R2 = [1731.3057861328125, 1105.4022216796875]
R3 = [1736.4453125, 1061.0040283203125]
R4 = [1809.9296875, 1063.6864013671875]

ImLDong = cv2.imread('ImageL7.bmp')
ImRDong = cv2.imread('ImageR7.bmp')
ImLDing = cv2.imread('ImageL8.bmp')
ImRDing = cv2.imread('ImageR8.bmp')

cv2.arrowedLine(ImLDong, (int(L2[0]), int(L2[1])), (int(L1[0]), int(L1[1])), (0, 0, 255), 5, 0, 0, 0.1)
# cv2.line(ImLDong, (int(L1[0]), int(L1[1])), (int(L2[0]), int(L2[1])), (0, 0, 255), 10)
cv2.namedWindow('testWindow', cv2.WINDOW_NORMAL)  # 窗口大小可设置
cv2.resizeWindow('testWindow', 1920, 960)  # 重设大小
cv2.imshow('testWindow', ImLDong)
cv2.waitKey(0)
cv2.imwrite('ImLDong.png', ImLDong)

cv2.arrowedLine(ImRDong, (int(R2[0]), int(R2[1])), (int(R1[0]), int(R1[1])), (0, 0, 255), 5, 0, 0, 0.1)
# cv2.line(ImLDong, (int(L1[0]), int(L1[1])), (int(L2[0]), int(L2[1])), (0, 0, 255), 10)
cv2.namedWindow('testWindow', cv2.WINDOW_NORMAL)  # 窗口大小可设置
cv2.resizeWindow('testWindow', 1920, 960)  # 重设大小
cv2.imshow('testWindow', ImRDong)
cv2.waitKey(0)
cv2.imwrite('ImRDong.png', ImRDong)

cv2.arrowedLine(ImLDing, (int(L4[0]), int(L4[1])), (int(L3[0]), int(L3[1])), (0, 0, 255), 5, 0, 0, 0.1)
# cv2.line(ImLDong, (int(L1[0]), int(L1[1])), (int(L2[0]), int(L2[1])), (0, 0, 255), 10)
cv2.namedWindow('testWindow', cv2.WINDOW_NORMAL)  # 窗口大小可设置
cv2.resizeWindow('testWindow', 1920, 960)  # 重设大小
cv2.imshow('testWindow', ImLDing)
cv2.waitKey(0)
cv2.imwrite('ImLDing.png', ImLDing)

cv2.arrowedLine(ImRDing, (int(R4[0]), int(R4[1])), (int(R3[0]), int(R3[1])), (0, 0, 255), 5, 0, 0, 0.1)
# cv2.line(ImLDong, (int(L1[0]), int(L1[1])), (int(L2[0]), int(L2[1])), (0, 0, 255), 10)
cv2.namedWindow('testWindow', cv2.WINDOW_NORMAL)  # 窗口大小可设置
cv2.resizeWindow('testWindow', 1920, 960)  # 重设大小
cv2.imshow('testWindow', ImRDing)
cv2.waitKey(0)
cv2.imwrite('ImRDing.png', ImRDing)
