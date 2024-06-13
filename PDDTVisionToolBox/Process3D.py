import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


def getPlaneFitting(Points):
    """
        拟合平面
    :param Points:三维点
    :return: 拟合平面 Z = AX + BY + C
    """
    # print(Points)
    x = Points[0, :]
    # print(x)
    y = Points[1, :]
    z = Points[2, :]
    num = np.size(x)
    # print(num)
    A = np.zeros((3, 3))
    A[2, 2] = num
    for i in range(num):
        A[0, 0] = A[0, 0] + x[i] ** 2
        A[0, 1] = A[0, 1] + x[i] * y[i]
        A[0, 2] = A[0, 2] + x[i]
        A[1, 0] = A[0, 1]
        A[1, 1] = A[1, 1] + y[i] ** 2
        A[1, 2] = A[1, 2] + y[i]
        A[2, 0] = A[0, 2]
        A[2, 1] = A[1, 2]
    b = np.zeros((3, 1))
    for i in range(num):
        b[0, 0] = b[0, 0] + x[i] * z[i]
        b[1, 0] = b[1, 0] + y[i] * z[i]
        b[2, 0] = b[2, 0] + z[i]
    # 求解X
    # print(b/A)
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        print("出现奇异，采用广义逆矩阵求解")
        A_inv = np.linalg.pinv(A)
    Plane = np.dot(A_inv, b)
    # print('\rPlane Fitting Completed', end='')
    print('平面拟合结果为：z = %.6f * x + %.6f * y + %.6f' % (Plane[0, 0], Plane[1, 0], Plane[2, 0]))
    return Plane


def getCircleFitting(Points):
    """
        三维圆拟合
    :param Points: 末端点三维坐标集
    :return: 三维圆心位置
    """
    x = Points[0, :]
    y = Points[1, :]
    z = Points[2, :]
    num = np.shape(x)[0]
    # print(num)
    # 构造A
    A = np.zeros((3, 3))
    A[0, 0] = num
    A[1, 1] = num
    A[2, 2] = num
    # 构造B
    b = np.zeros((3, 1))
    for i in range(num):
        b[0, 0] = b[0, 0] + x[i]
        b[1, 0] = b[1, 0] + y[i]
        b[2, 0] = b[2, 0] + z[i]
    A_inv = np.linalg.inv(A)
    center = np.dot(A_inv, b)
    r = 0
    for i in range(num):
        # print(abs(math.sqrt((center[0, 0] - x[i]) ** 2 + (center[1, 0] - y[i]) ** 2 + (center[2, 0] - z[i]) ** 2)))
        r = r + abs(math.sqrt((center[0, 0] - x[i]) ** 2 + (center[1, 0] - y[i]) ** 2 + (center[2, 0] - z[i]) ** 2))
    r = r / num
    print(' 三维圆拟合圆心位置：', center[0, 0], center[1, 0], center[2, 0], ' 拟合半径为： ', r, 'mm')
    return center, r


def getCameraMatrix(Intrinsic, Rotation, Translation):
    """
        获取相机矩阵
    :param Intrinsic:       内参矩阵
    :param Rotation:        旋转矩阵
    :param Translation:     平移矩阵
    :return:
    """
    mCamera = np.hstack([Rotation, Translation])
    mCamera = np.dot(Intrinsic, mCamera)
    return mCamera


def get3dPosition(mLeftM, mRightM, lx, ly, rx, ry):
    """
        根据相机矩阵完成二维到三维变换
    :param mLeftM:       左相机矩阵
    :param mRightM:      右相机矩阵
    :param lx:           左x坐标
    :param ly:           右y坐标
    :param rx:           右x坐标
    :param ry:           右y坐标
    :return:
    """
    # 构造齐次方程组
    A = np.zeros(shape=(4, 3))
    for i in range(0, 3):
        A[0][i] = lx * mLeftM[2, i] - mLeftM[0][i]
    for i in range(0, 3):
        A[1][i] = ly * mLeftM[2][i] - mLeftM[1][i]
    for i in range(0, 3):
        A[2][i] = rx * mRightM[2][i] - mRightM[0][i]
    for i in range(0, 3):
        A[3][i] = ry * mRightM[2][i] - mRightM[1][i]
        B = np.zeros(shape=(4, 1))
    for i in range(0, 2):
        B[i][0] = mLeftM[i][3] - lx * mLeftM[2][3]
    for i in range(2, 4):
        B[i][0] = mRightM[i - 2][3] - rx * mRightM[2][3]
    XYZ = np.zeros(shape=(3, 1))
    # 采用最小二乘法求其空间坐标
    cv2.solve(A, B, XYZ, cv2.DECOMP_SVD)
    return XYZ


def getIncludedAngleByVectors(Vec1, Vec2):
    """
        求解两向量夹角
    :param Vec1: 向量1
    :param Vec2: 向量2
    :return:
    """
    # 取模
    dot1 = np.sqrt(Vec1.dot(Vec1))
    dot2 = np.sqrt(Vec2.dot(Vec2))
    # 取点积
    dot3 = Vec1.dot(Vec2)
    IncludedCos = dot3 / (dot1 * dot2)
    IncludedAngle = np.arccos(IncludedCos)
    IncludedAngle = IncludedAngle * 180 / np.pi
    # print('拟合平面偏角为%f°' % IncludedAngle)
    return IncludedAngle


def getHandEyeCalibChessBoardModel(row, col, dis):
    """
        创建以棋盘格为基础的手眼标定标准模型
    :param row: 棋盘格行数
    :param col: 棋盘格列数
    :param dis: 棋盘格间距
    :return:
    """
    # 内部角点的行列数
    Row = row - 1
    Col = col - 1
    # 创建空棋盘
    ChessPoints = np.zeros((Row * Col, 3), np.float32)
    # 按棋盘格形状赋值
    ChessPoints[:, :2] = np.mgrid[0:Col, 0:Row].T.reshape(-1, 2)
    # 棋盘间隔,单位为mm
    ChessPoints = dis * ChessPoints
    return ChessPoints


def getHandEyeCalibChessBoardPoseEstimate(Img, Model, Row, Col, Intrinsic, Distortion):
    """
        手眼标定中，获得棋盘格位姿估计
    :param Img: 输入RGB图像
    :param Model: 棋盘格模型
    :param Row: 棋盘格行数
    :param Col: 棋盘格列数
    :param Intrinsic: 内参矩阵
    :param Distortion: 畸变矩阵
    :return:
    """
    # 内部棋盘格行列数
    Row = Row - 1
    Col = Col - 1
    # 灰度化
    gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    # 获取图像尺寸
    size = gray.shape[::-1]
    # 畸变校正
    # gray = cv2.undistort(gray, Intrinsic, Distortion)
    # 获取角点
    ret, corners = cv2.findChessboardCorners(gray, (Col, Row), None)
    # 角点细化
    corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001))
    # pnp求解
    _, rvec, tvec = cv2.solvePnP(Model, corners2, Intrinsic, Distortion)
    # 列向量到行向量,便于后续拼接计算
    return rvec.T, tvec.T
