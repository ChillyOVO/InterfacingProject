import cv2
import numpy as np


# 使用霍夫直线变换做直线检测，前提条件：边缘检测已经完成

# 标准霍夫线变换
def line_detection_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)  # 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    for line in lines:
        rho, theta = line[0]  # line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的
        a = np.cos(theta)  # theta是弧度
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))  # 直线起点横坐标
        y1 = int(y0 + 1000 * (a))  # 直线起点纵坐标
        x2 = int(x0 - 1000 * (-b))  # 直线终点横坐标
        y2 = int(y0 - 1000 * (a))  # 直线终点纵坐标
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.namedWindow('testWindow', cv2.WINDOW_NORMAL)  # 窗口大小可设置
    cv2.resizeWindow('testWindow', 960, 480)  # 重设大小
    cv2.imshow('testWindow', image)
    cv2.waitKey(0)
    cv2.imwrite('HoughLineFilter.jpg', image)


# 统计概率霍夫线变换
def line_detect_possible_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    # ret,imBW = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    # cv2.namedWindow('testWindow', cv2.WINDOW_NORMAL)  # 窗口大小可设置
    # cv2.resizeWindow('testWindow', 960, 480)  # 重设大小
    # cv2.imshow('testWindow', imBW)
    # cv2.waitKey(0)
    # edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 15)
    edges = cv2.bitwise_not(edges)
    cv2.namedWindow('testWindow', cv2.WINDOW_NORMAL)  # 窗口大小可设置
    cv2.resizeWindow('testWindow', 960, 480)  # 重设大小
    cv2.imshow('testWindow', edges)
    cv2.waitKey(0)
    # 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=300, maxLineGap=10)
    ImgMasked = image.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        print((y2 - y1) / (x2 - x1))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 5)
    cv2.namedWindow('testWindow', cv2.WINDOW_NORMAL)  # 窗口大小可设置
    cv2.resizeWindow('testWindow', 960, 480)  # 重设大小
    cv2.imshow('testWindow', image)
    cv2.waitKey(0)
    cv2.imwrite('HoughLineFilter.jpg', image)


if __name__ == "__main__":
    img = cv2.imread("./20230815/RefImage/Left8000.bmp")
    # line_detection_demo(img)
    line_detect_possible_demo(img)
