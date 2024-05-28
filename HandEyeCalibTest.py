import cv2
import math

import numpy as np


class EyeToHandCalibration:
    """
        该函数用于进行眼在手外的手眼标定
        getTransEndToBase:          已知参数，通过正逆运动学求解示教器显示内容
        getTransObjectToCamera:     已知参数，通过机器视觉识别ArUco码进行位姿估计
        getTransBaseToCamera:       待求解参数，通过CV2.HandEyeCalibration求解
        getTransEndToObject:        待求解参数，通过变换方程带入求解
    """

    def __init__(self):
        """
            初始化函数，部分参数设置
        """
        # 机械臂示教器读数
        # # 第一次位置
        # self.X1 = 763.98
        # self.Y1 = -61.74
        # self.Z1 = 191.38
        # self.Rx1 = 178.94
        # self.Ry1 = 1.66
        # self.Rz1 = -44.58
        # # 第二次位置
        # self.X2 = 763.98
        # self.Y2 = -61.80
        # self.Z2 = 207.89
        # self.Rx2 = 178.94
        # self.Ry2 = 1.67
        # self.Rz2 = -36.02
        # # 第三次位置
        # self.X3 = 763.97
        # self.Y3 = -61.75
        # self.Z3 = 207.94
        # self.Rx3 = 178.94
        # self.Ry3 = 1.66
        # self.Rz3 = -71.35
        # # 第四次位置
        # self.X4 = 746.01
        # self.Y4 = -61.73
        # self.Z4 = 195.03
        # self.Rx4 = -172.35
        # self.Ry4 = -1.28
        # self.Rz4 = -71.31

        # 相机标定内参
        self.Intrinsic = np.array([[3878.67076046738, 0, 1271.86938448416],
                                   [0, 3877.43716227348, 931.199922162904],
                                   [0, 0, 1]])
        self.Distortion = np.array(
            [[-0.105685880286843, 0.129859778333246, -0.000373356703561176, -0.000601775856891997, -0.104339305476131]])

        # ArUco编码参数
        self.Criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        self.Dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.MarkerSide = 0.01

    def Skew(self, M):
        """
            偏度计算
        :param M:
        :return:
        """
        EM = np.mean(M)
        EM2 = np.mean(M ** 2)
        EM3 = np.mean(M ** 3)
        Sigma = math.sqrt(EM2 - EM ** 2)
        Skew = (EM3 - 3 * EM * (Sigma ** 2) - EM ** 3) / (Sigma ** 3)
        return Skew

    def SkewMatrix(self, Vector):
        """
            获得反对称矩阵
        :param Vector: 向量
        :return:
        """
        SkewMatrix = np.array([[0, -Vector[2], Vector[1]],
                               [Vector[2], 0, -Vector[0]],
                               [-Vector[1], Vector[0], 0]])
        return SkewMatrix

    def getTransBaseToEnd(self, X, Y, Z, Rx, Ry, Rz):
        """
            通过示教器读数确定末端位置，旋转角读数为角度值
        :param X:   示教器读数 X
        :param Y:   示教器读数 Y
        :param Z:   示教器读数 Z
        :param Rx:  示教器读数 Rx
        :param Ry:  示教器读数 Ry
        :param Rz:  示教器读数 Rz
        :return: TBE, R, t
        """
        # 构造旋转矩阵
        RotX = np.array([[1, 0, 0], [0, math.cos(Rx), -math.sin(Rx)], [0, math.sin(Rx), math.cos(Rx)]])
        RotY = np.array([[math.cos(Ry), 0, math.sin(Ry)], [0, 1, 0], [-math.sin(Ry), 0, math.cos(Ry)]])
        RotZ = np.array([[math.cos(Rz), -math.sin(Rz), 0], [math.sin(Rz), math.cos(Rz), 0], [0, 0, 1]])
        R = RotZ @ RotY @ RotX
        t = np.array([[X], [Y], [Z]])
        # 列合并
        TBE = np.column_stack([R, t])
        # 行合并
        TBE = np.row_stack((TBE, np.array([0, 0, 0, 1])))
        return TBE, R, t

    def getTransEndToBase(self, X, Y, Z, Rx, Ry, Rz):
        """
            通过示教器读数确定末端位置，旋转角读数为角度值
        :param X:   示教器读数 X
        :param Y:   示教器读数 Y
        :param Z:   示教器读数 Z
        :param Rx:  示教器读数 Rx
        :param Ry:  示教器读数 Ry
        :param Rz:  示教器读数 Rz
        :return: TEB, R, t
        """
        # 构造旋转矩阵
        RotX = np.array([[1, 0, 0], [0, math.cos(Rx), -math.sin(Rx)], [0, math.sin(Rx), math.cos(Rx)]])
        RotY = np.array([[math.cos(Ry), 0, math.sin(Ry)], [0, 1, 0], [-math.sin(Ry), 0, math.cos(Ry)]])
        RotZ = np.array([[math.cos(Rz), -math.sin(Rz), 0], [math.sin(Rz), math.cos(Rz), 0], [0, 0, 1]])
        # print(RotX)
        R = RotZ @ RotY @ RotX
        # print(np.linalg.det(R))
        t = np.array([[X], [Y], [Z]])
        R = R.T
        t = -R @ t
        # 列合并
        TEB = np.column_stack([R, t])
        # 行合并
        TEB = np.row_stack((TEB, np.array([0, 0, 0, 1])))
        return TEB, R, t

    def getTransEndToBase2(self, X, Y, Z, Rx, Ry, Rz):
        """
            通过示教器读数确定末端位置，旋转角读数为角度值
        :param X:   示教器读数 X
        :param Y:   示教器读数 Y
        :param Z:   示教器读数 Z
        :param Rx:  示教器读数 Rx
        :param Ry:  示教器读数 Ry
        :param Rz:  示教器读数 Rz
        :return: TEB, R, t
        """
        Rx = -Rx
        Ry = -Ry
        Rz = -Rz
        # 构造旋转矩阵
        RotX = np.array([[1, 0, 0], [0, math.cos(Rx), -math.sin(Rx)], [0, math.sin(Rx), math.cos(Rx)]])
        RotY = np.array([[math.cos(Ry), 0, math.sin(Ry)], [0, 1, 0], [-math.sin(Ry), 0, math.cos(Ry)]])
        RotZ = np.array([[math.cos(Rz), -math.sin(Rz), 0], [math.sin(Rz), math.cos(Rz), 0], [0, 0, 1]])
        # print(RotX)
        R = RotX @ RotY @ RotZ
        # print(np.linalg.det(R))
        t = np.array([[X], [Y], [Z]])
        t = -R @ t
        # 列合并
        TEB = np.column_stack([R, t])
        # 行合并
        TEB = np.row_stack((TEB, np.array([0, 0, 0, 1])))
        return TEB, R, t


    def getTransObjectToCamera(self, Image):
        """
            通过输入图像获得夹爪末端靶标到相机变换，采用ArUco编码
        :param Image:
        :return:TOC, R, t
        """
        # 灰度化
        Img = cv2.cvtColor(Image, cv2.COLOR_RGB2GRAY)
        # 初始化ArUco类
        parameters = cv2.aruco.DetectorParameters_create()
        Dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(Img, Dictionary,
                                                                               parameters=parameters)
        if len(markerCorners) > 0:
            # 展平 ArUCo ID 列表
            ids = markerIds.flatten()
            # 循环检测到的 ArUCo 标记
            for (markerCorner, markerID) in zip(markerCorners, ids):
                if markerID == 28:
                    Rotation, Translation, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorner, 10,
                                                                                   self.Intrinsic, self.Distortion)

        Rotation = cv2.Rodrigues(Rotation)[0]
        # print(np.linalg.det( Rotation))
        R = Rotation.T
        # print(R)
        # print(Translation[0].T)
        t = -R @ (Translation[0].T)
        # 列合并
        TOC = np.column_stack([R, t])
        # 行合并
        TOC = np.row_stack((TOC, np.array([0, 0, 0, 1])))
        # print(TOC)
        return TOC, R, t

    def getTransCameraToObject(self, Image):
        """
            通过输入图像获得夹爪末端靶标到相机变换，采用ArUco编码
        :param Image:
        :return:TOC, R, t
        """
        # 灰度化
        ImgCopy = Image.copy()
        Img = cv2.cvtColor(Image, cv2.COLOR_RGB2GRAY)
        # 初始化ArUco类
        parameters = cv2.aruco.DetectorParameters_create()
        Dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(Img, Dictionary,
                                                                               parameters=parameters)
        if len(markerCorners) > 0:
            # 展平 ArUCo ID 列表
            ids = markerIds.flatten()
            # 循环检测到的 ArUCo 标记
            for (markerCorner, markerID) in zip(markerCorners, ids):
                if markerID == 28:
                    Rotation, Translation, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorner, 10,
                                                                                   self.Intrinsic, self.Distortion)
        # img = cv2.aruco.drawDetectedMarkers(ImgCopy, markerCorners, markerIds)
        # img = cv2.drawFrameAxes(img, self.Intrinsic, self.Distortion, Rotation, Translation, 0.1)
        # cv2.namedWindow('img', 0)
        # cv2.resizeWindow('img', 1333, 1000)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        R = cv2.Rodrigues(Rotation)[0]
        # print(Translation)
        t = Translation[0].T
        # 列合并
        TCO = np.column_stack([R, t])
        # 行合并
        TCO = np.row_stack((TCO, np.array([0, 0, 0, 1])))
        # print(TCO)
        return TCO, R, t

    def getTransBaseToCamera(self, REB, TEB, ROC, TOC):
        """
            通过CalibHandEye进行手眼标定
        :param REB:Rotation of End to Base
        :param TEB:Translation of End to Base
        :param ROC:Rotation of Object to Camera
        :param TOC:Translation of Object to Camera
        :return:
        """
        R, t = cv2.calibrateHandEye(REB, TEB, ROC, TOC)
        return R, t

    def getHandEyeCalibrationTsai(self, R1, t1, R2, t2):
        """
            求解AX = XB
        :param R1: 分解后的R1
        :param t1: 分解后的t1
        :param R2: 分解后的R2
        :param t2: 分解后的t2
        :return:
        """
        num = len(R1)
        S = np.zeros(num * 3, 3)
        V = np.zeros(num * 3, 1)
        for i in range(num):
            R1m = np.log(R1[i])
            R2m = np.log(R2[i])
            R1mSkew = np.array([[R1m[3, 2]], [R1m[1, 3]], [R1m[2, 1]]])
            R2mSkew = np.array([[R2m[3, 2]], [R2m[1, 3]], [R2m[2, 1]]])
            R1mSkew = R1mSkew / np.linalg.norm(R1mSkew)
            R2mSkew = R2mSkew / np.linalg.norm(R2mSkew)
            S[3 * i:3 * i + 3, :] = np.linalg.inv(self.SkewMatrix(R1mSkew + R2mSkew))
            V[3 * i:3 * i + 3, :] = R1mSkew - R2mSkew
            # S.append(self.SkewMatrix(R1mSkew + R2mSkew))
            # V.append(R1mSkew - R2mSkew)

        # theta = 2 * math.tan(np.linalg.norm(X))
        # Rotation = (np.eye(3) @ math.cos(theta) + math.sin(theta) @ self.SkewMatrix(X) + (
        #         1 - math.cos(theta)) @ X @ X.T).T
        #
        # Translation = np.linalg.inv(np.eye(3) - R1) @ (t1 - Rotation @ t2)
        # R.append(Rotation)
        # t.append(Translation)
