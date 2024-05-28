from pypylon import pylon
import numpy as np
import cv2
import math


class TransferWorkpieceToEnd:
    # 实现从工件到末端的坐标变换
    # 包含函数：
    # __init__: 初始化函数,并提供已知参数
    # getImage：获取图像
    # workpieceLocation：获取工件位置（变换）
    # endLocation:获取末端位置（变换）
    # workpieceToEnd:工件到末端变换

    def __init__(self):
        # 内参标定数据
        # 内参矩阵
        self.k = np.array([[7782.03952836787, 0, 1296.88243028351],
                           [0, 7818.69137378419, 934.200830909172],
                           [0, 0, 1]])
        # 预定平面深度 Zc 单位mm
        # cv2.sovlePnP算法不需要规定平面
        # self.Zc = []
        # 畸变参数
        self.distoration = np.array([[-0.186719169098603, 0.233713201373500, 0, 0, 0]])
        # 形态操作核
        self.kernelCircle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # 圆孔真实半径
        self.circleRadius = 3.3
        # 编码边长
        self.markerLength = 5
        # 小工件世界坐标系
        self.modelPoints3 = np.float32([[-12.5, 12.5, 0],
                                        [-12.5, -12.5, 0],
                                        [12.5, 12.5, 0],
                                        [0, 0, 0]])
        # 大工件世界坐标系 尺度与末端编码一致
        self.modelPoints4 = np.float32([[-25, 25, 0],
                                        [-25, -25, 0],
                                        [25, 25, 0],
                                        [25, -25, 0]])
        # 末端工件世界坐标系
        self.modelPointsEnd = np.float32([[0, 10, 0],
                                          [10, 0, 0],
                                          [0, -10, 0],
                                          [-10, 0, 0]])
        # 角点细化判据
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    def getImage(self):
        # 函数功能：调用Basler相机成像
        # 连接Basler相机列表的第一个相机
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        # 开始读取图像
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        converter = pylon.ImageFormatConverter()
        # 转换为OpenCV的BGR彩色格式
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        # while camera.IsGrabbing():
        # 抓取图像,数值为曝光时间,但是相机本身已经前期设置曝光时间,改了没啥用
        grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            # 转换为OpenCV图像格式
            image = converter.Convert(grabResult)
            Img = image.GetArray()
        grabResult.Release()
        # 关闭相机
        # camera.StopGrabbing()
        return Img

    def workpieceLocation(self, Image, TypeNumber):
        # 函数功能:检测工件并返回工件位姿
        # 输入部分：
        # Image：相机获取图像法
        # TypeNumber: 抓取工件类型,当前仅两种: 0 小型3孔模型;1 大型4孔模型
        # 输出部分：
        # rotationMatrix, translationVector：旋转矩阵及平移矩阵

        # 首先进行灰度化
        Image = cv2.cvtColor(Image, cv2.COLOR_RGB2GRAY)
        # 中值滤波滤除噪声
        Image = cv2.medianBlur(Image, 5)
        # # 自适应阈值滤波获得二值图
        # ImgBW = cv2.adaptiveThreshold(Image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 205, 5)
        # # 腐蚀图像
        # ImgBW = cv2.erode(ImgBW, self.kernelCircle)
        # 霍夫圆检测
        # 可调节参数（放大倍率,最小间隔像素数,canny阈值,累加器阈值,最小半径,最大半径）
        # 调节方式 需要稳定光照环境，测试环境:晴,早上10点光照环境
        circles = cv2.HoughCircles(Image, cv2.HOUGH_GRADIENT, 5.0001, 300, param1=50, param2=100, minRadius=48,
                                   maxRadius=51)
        centersCircle = circles[0, :, 0:3]
        centersCircle = np.array(centersCircle, dtype=float)
        # # 得到平均换算尺度因子, 表示每mm多少各像素
        # k = np.mean(centersCircle[:][2]) / self.circleRadius
        # 初始化工件中心值
        PositionX = 0
        PositionY = 0
        if TypeNumber == 0:
            # 小型3孔工件 前期标定根据工艺参数得到三圆间距参数CircleDistance 由实际距离换算
            CircleDistance = 250
            # 遍历获得中心累加值
            PositionX = np.sum(centersCircle, axis=0)[0]
            PositionY = np.sum(centersCircle, axis=0)[1]
            # for i in centersCircle:
            #     PositionX = PositionX + i[0]
            #     PositionY = PositionY + i[1]
            # 获得中心位置
            PositionX = (PositionX - CircleDistance) / 3 + CircleDistance / 2
            PositionY = (PositionY - CircleDistance) / 3 + CircleDistance / 2
            centerPos = [PositionX, PositionY]
            # 排序各点
            for circle in centersCircle:
                if circle[0] < PositionX and circle[1] < PositionY:
                    PosLeftTop = np.array([circle[0], circle[1]])
                elif circle[0] < PositionX and circle[1] > PositionY:
                    PosLeftBottom = np.array([circle[0], circle[1]])
                else:
                    PosRightTop = np.array([circle[0], circle[1]])
            # 方向为左上,左下,右上,中心
            CircelPoints = np.array([PosLeftTop, PosLeftBottom, PosRightTop, centerPos])
            # 求工件坐标系到相机坐标系旋转向量和平移向量
            success, rotationVector, translationVector = cv2.solvePnP(self.modelPoints3, CircelPoints, self.k,
                                                                      self.distoration, flags=cv2.SOLVEPNP_ITERATIVE)
        elif TypeNumber == 1:
            # 大型4孔工件,直接求取平均值
            # 遍历获得中心累加值
            PositionX = np.sum(centersCircle, axis=0)[0]
            PositionY = np.sum(centersCircle, axis=0)[1]
            PositionX = PositionX / 4
            PositionY = PositionY / 4
            for circle in centersCircle:
                if circle[0] < PositionX and circle[1] < PositionY:
                    PosLeftTop = np.array([circle[0], circle[1]])
                elif circle[0] < PositionX and circle[1] > PositionY:
                    PosLeftBottom = np.array([circle[0], circle[1]])
                elif circle[0] > PositionX and circle[1] < PositionY:
                    PosRightTop = np.array([circle[0], circle[1]])
                else:
                    PosRightBottm = np.array([circle[0], circles[1]])
            # 获取图像坐标系下对象定位,顺序为 左上,左下,右上,右下
            CircelPoints = np.float32([PosLeftTop, PosLeftBottom, PosRightTop, PosRightBottm])
            success, rotationVector, translationVector = cv2.solvePnP(self.modlePoints4, CircelPoints, self.k,
                                                                      self.distoration, flags=cv2.SOLVEPNP_ITERATIVE)
        # 由旋转向量求解旋转矩阵
        rotationMatrix = cv2.Rodrigues(rotationVector)[0]
        return rotationMatrix, translationVector

    def endLocation(self, Image):
        # 函数功能:检测末端并返回工件位姿
        # 输入部分：
        # Image：相机获取图像法
        # 输出部分：
        # LocationMarker: [(x,y),,,,(x,y)] 图像坐标系坐标
        # LocationCenter: (x,y) 码中心坐标

        # 首先进行灰度化
        Image = cv2.cvtColor(Image, cv2.COLOR_RGB2GRAY)
        # ArUco码字典规定
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        # 创建ArUco识别类
        parameters = cv2.aruco.DetectorParameters_create()
        # 识别ArUco Marker
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(Image.copy(), dictionary,
                                                                               parameters=parameters)
        # 预计检测到8个不同的ArUco Marker 预定义8个不同Marker位置 分别为左右夹爪各4编码 序号为42-29
        # LocLeftTop = []
        # LocLeftBottom = []
        # LocLeftLeft = []
        # LocLeftRight = []
        # LocRightTop = []
        # LocRightBottom = []
        # LocRightLeft = []
        # LocRightRight = []
        if len(markerCorners) > 0:
            # 展平 ArUCo ID 列表
            ids = markerIds.flatten()
            # 循环检测到的 ArUCo 标记
            for (markerCorner, markerID) in zip(markerCorners, ids):
                # 提取始终按以下顺序返回的标记：
                # TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT
                corners = markerCorner.reshape((4, 2))
                corners = cv2.cornerSubPix(Image, corners, (5, 5), (-1, -1), self.criteria)
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                # 先计算ArUco位置中心
                cX = (topLeft[0] + bottomRight[0]) / 2.0
                cY = (topLeft[1] + bottomRight[1]) / 2.0
                MarkerCenter = [cX, cY]
                # 识别并排序各编码 顺序应有错误 需修改
                if markerID == 42:
                    LocRightLeft = MarkerCenter
                elif markerID == 43:
                    LocRightRight = MarkerCenter
                elif markerID == 44:
                    LocRightBottom = MarkerCenter
                elif markerID == 45:
                    LocRightTop = MarkerCenter
                elif markerID == 46:
                    LocLeftBottom = MarkerCenter
                elif markerID == 47:
                    LocLeftTop = MarkerCenter
                elif markerID == 48:
                    LocLeftRight = MarkerCenter
                elif markerID == 49:
                    LocLeftLeft = MarkerCenter
            LocLeftMarker = np.float32([LocLeftTop, LocLeftRight, LocLeftBottom, LocLeftLeft])
            # LocLeftCenter = [(LocLeftTop[0] + LocLeftBottom[0] + LocLeftRight[0] + LocLeftLeft[0]) / 4,
            #                  (LocLeftTop[1] + LocLeftBottom[1] + LocLeftRight[1] + LocLeftLeft[1]) / 4]
            # LocRightMarker = np.float32([LocRightTop, LocRightRight, LocRightBottom, LocRightLeft])
            # LocRightCenter = [(LocRightTop[0] + LocRightBottom[0] + LocRightRight[0] + LocRightLeft[0]) / 4,
            #                   (LocRightTop[1] + LocRightBottom[1] + LocRightRight[1] + LocRightLeft[1]) / 4]
            success, rotationVector, translationVector = cv2.solvePnP(self.modelPointsEnd, LocLeftMarker, self.k,
                                                                      self.distoration, flags=cv2.SOLVEPNP_ITERATIVE)
            # 由旋转向量求解旋转矩阵
            rotationMatrix = cv2.Rodrigues(rotationVector)[0]
        return rotationMatrix, translationVector

    def workpieceToEnd(self, rotP, transP, rotE, transE):
        # 函数功能：获得工件到末端的旋转平移矩阵
        # 输入 rotP,transP 工件旋转平移矩阵 rotE,transE 末端旋转平移矩阵
        # 输出 rot,trans 工件到末端的旋转平移矩阵
        rot = np.linalg.inv(rotE) @ rotP
        # print(transP, "---", transE)
        # print(np.linalg.inv(rotE))
        trans = np.linalg.inv(rotE) @ (transP - transE)
        return rot, trans


# 初始化类 TargetLocation
transfer = TransferWorkpieceToEnd()
# 读取当前帧
Img = transfer.getImage()
# 获得相机坐标系下位置
rotationWorkpiece, translationWorkpiece = transfer.workpieceLocation(Img, 0)
# print(rotationWorkpiece, translationWorkpiece)
rotationEnd, translationEnd = transfer.endLocation(Img)
# print(rotationEnd, translationEnd)
rotation, translation = transfer.workpieceToEnd(rotationWorkpiece, translationWorkpiece, rotationEnd, translationEnd)
# 以平移矩阵Z平移参数设置循环，直到达到可用范围为准
count = 0
while abs(translation[2]) > 100:
    # 重新读图
    Img = transfer.getImage()
    # 获得相机坐标系下位置
    rotationWorkpiece, translationWorkpiece = transfer.workpieceLocation(Img, 0)
    # 末端定位
    rotationEnd, translationEnd = transfer.endLocation(Img)
    # 坐标变换
    rotation, translation = transfer.workpieceToEnd(rotationWorkpiece, translationWorkpiece, rotationEnd,
                                                    translationEnd)
    count = count + 1

print("旋转矩阵：", rotation)
print("平移矩阵：", translation)
print("规避奇异解次数：", count)
