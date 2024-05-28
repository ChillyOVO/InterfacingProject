import cv2
import numpy as np

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def change_contrast(Image):
    '''改变对比度和亮度'''
    blank = np.zeros(Image.shape, Image.dtype)
    # dst = alpha * img + beta * blank
    # 假设对比度为1.2，亮度为0
    dst = cv2.addWeighted(Image, 1.2, blank, 1 - 1.2, -100)

    return dst


img = cv2.imread('ImageL7.bmp')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = change_contrast(gray)
# gray = cv2.pyrMeanShiftFiltering(gray, 10, 50)    #仅适用于8bit 3Channel图像

cv2.namedWindow('testWindow', cv2.WINDOW_NORMAL)  # 窗口大小可设置
cv2.resizeWindow('testWindow', 960, 480)  # 重设大小
cv2.imshow('testWindow', gray)
cv2.waitKey(0)
# gray = cv2.medianBlur(gray, 3)
# gray = cv2.bilateralFilter(src=gray, d=25, sigmaColor=100, sigmaSpace=5)
# dst = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 6)
# ret, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ImgBW = cv2.Canny(gray, 100, 150)
# ImgBW = cv2.medianBlur(ImgBW, 3)
# ImgBW = cv2.Sobel(gray, cv2.CV_8UC1, 1, 1, ksize=3)

cv2.namedWindow('testWindow', cv2.WINDOW_NORMAL)  # 窗口大小可设置
cv2.resizeWindow('testWindow', 960, 480)  # 重设大小
cv2.imshow('testWindow', ImgBW)
cv2.waitKey(0)
cnt, hierarchy = cv2.findContours(ImgBW, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
area = []

img1 = img.copy()
cv2.drawContours(img1, cnt, -1, (0, 0, 255), 2)
cv2.namedWindow('testWindow', cv2.WINDOW_NORMAL)  # 窗口大小可设置
cv2.resizeWindow('testWindow', 960, 480)  # 重设大小
cv2.imshow('testWindow', img1)
cv2.waitKey(0)

# print(len(cnt))
for i in range(len(cnt)):
    if cnt[i].size > 1000 and cnt[i].size < 10000:
        Area = abs(cv2.contourArea(cnt[i], True))
        Length = cv2.arcLength(cnt[i], True)
        # line = cnt[i].size
        t = (Length * Length) / 4 * 3.14 * Area
        # print(t)
        if t > 10000000:
            # 椭圆拟合
            ellipse = cv2.fitEllipse(cnt[i])
            t1 = ellipse[1][1] / ellipse[1][0]
            # print(ellipse)
            print(t1)
            if t1 < 1.005:
                cv2.ellipse(img, ellipse, (0, 0, 255), 2)
                # print(cnt[i].size)
                print(ellipse[0])

cv2.namedWindow('testWindow', cv2.WINDOW_NORMAL)  # 窗口大小可设置
cv2.resizeWindow('testWindow', 960, 480)  # 重设大小
cv2.imshow('testWindow', img)
cv2.waitKey(0)

cv2.imwrite('processedImg.jpg', img)
