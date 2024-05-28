import cv2
import math
import time
import numpy as np
import PDDTVisionToolBox as pd

# 机械臂端口设置
IPRob = "192.168.0.101"
PortNumberRob = 30003

# 相机内参
Intrinsic = np.array([[4316.75424290671, 0, 1951.55163153415],
                      [0, 4320.31434343874, 1059.78199192089],
                      [0, 0, 1]])
Distortion = np.array([-0.109832680769026, 0.0853581279335363, 0, 0])

# 示教初始位置
Pos0 = np.array([0.415, 0.105, 0.770, 2.902, 1.202, 0])

# 建立标准模板点
# 棋盘格行列数
ChessRow = 9
ChessCol = 12
# 棋盘格间隔
ChessDis = 5
ChessPoints = pd.getHandEyeCalibChessBoardModel(ChessRow, ChessCol, ChessDis)
# 生成过程-未封装
# ChessPoints = np.zeros(((ChessRow - 1) * (ChessCol - 1), 3), np.float32)
# ChessPoints[:, :2] = np.mgrid[0:(ChessCol - 1), 0:(ChessRow - 1)].T.reshape(-1, 2)
# ChessPoints = 5 * ChessPoints

# 创建存图目录
fileName = 'Logs_of_Images'
childFileName = 'Pic'
path0, pathList = pd.getDirPathOfLogsbYTime(fileName, childFileName, 2)

# 控制机械臂在当前位置为中心的3*3*3矩阵内运动,同时在每个平面各沿X,Y方向做±5°偏转
# 设置各方向步长
stepX = 0.020
stepY = 0.020
stepZ = 0.010
stepZAngle = 0.005
theta = 3 / 180 * math.pi
# 创建旋转矢量、平移矢量、机械臂位姿矢量矩阵
# 设定标定点数目
# TotalPosNum = 9*3 + 9*2
TotalPosNum = 45
ChessRotVec = np.zeros((TotalPosNum, 3))
ChessTransVec = np.zeros((TotalPosNum, 3))
RobertPosVec = np.zeros((TotalPosNum, 6))
# 计数参数
count = 0
# 到示教位
pd.getUR10EMove(IPRob, PortNumberRob, Pos0[0], Pos0[1], Pos0[2], Pos0[3], Pos0[4], Pos0[5])
# 进行运动,控制Z轴高度
for i in range(3):
    pd.getUR10EMove(IPRob, PortNumberRob, Pos0[0], Pos0[1], Pos0[2] + (i - 1) * stepZ, Pos0[3], Pos0[4], Pos0[5])
    Pos1 = pd.getUR10EPose(IPRob, PortNumberRob)
    # 平面内平移
    # 控制X轴
    for j in range(3):
        pd.getUR10EMove(IPRob, PortNumberRob, Pos1[0] + (j - 1) * stepX, Pos1[1], Pos1[2], Pos1[3], Pos1[4], Pos1[5])
        Pos2 = pd.getUR10EPose(IPRob, PortNumberRob)
        # 控制Y轴
        for k in range(3):
            pd.getUR10EMove(IPRob, PortNumberRob, Pos2[0], Pos2[1] + (k - 1) * stepY, Pos2[2], Pos2[3], Pos2[4],
                            Pos2[5])
            Pos3 = pd.getUR10EPose(IPRob, PortNumberRob)
            # 记录位姿
            RobertPosVec[count, :] = Pos3
            # 存图
            ImgTemp = pd.getSingleImageByBasler()
            cv2.imwrite(pathList[0] + '/' + str(count) + '.bmp', ImgTemp)
            # 获取旋转矢量，平移矢量
            ChessRot, ChessTrans = pd.getHandEyeCalibChessBoardPoseEstimate(ImgTemp, ChessPoints, ChessRow, ChessCol,
                                                                            Intrinsic, Distortion)
            ChessRotVec[count, :] = ChessRot
            ChessTransVec[count, :] = ChessTrans
            count = count + 1

    # # 旋转角度采图
    # # 获取当前姿态
    # # PosNow = pd.getUR10EPose(IPRob, PortNumberRob)
    # # 从旋转矢量到旋转角度,计算旋转矩阵是为了验证,后续为提高精度将一步到位
    # RotVec = np.array([Pos1[3], Pos1[4], Pos1[5]])
    # RotMat = pd.getRotVec2RotMAT(RotVec)
    # Rx, Ry, Rz = pd.getRotationMatrixToAngles(RotMat)
    #
    # # Rx方向旋转
    # for j in range(2):
    #     Rx1 = Rx + ((-1) ** j) * theta
    #     # 避免奇异,限定范围
    #     if Rx1 < -math.pi:
    #         Rx1 = Rx1 + 2 * math.pi
    #     elif Rx1 > math.pi:
    #         Rx1 = Rx1 - 2 * math.pi
    #     # 根据角度计算回旋转矢量
    #     RotMatNew = pd.getAnglesToRotaionMatrix(Rx1, Ry, Rz)
    #     RotVecNew = cv2.Rodrigues(RotMatNew)[0]
    #     pd.getUR10EMove(IPRob, PortNumberRob, Pos1[0], Pos1[1], Pos1[2], RotVecNew[0], RotVecNew[1], RotVecNew[2])
    #     # 机械臂旋转速度过快,导致存图虚影,需有时间间隔
    #     time.sleep(3)
    #     # 记录位姿
    #     Pos3 = pd.getUR10EPose(IPRob, PortNumberRob)
    #     RobertPosVec[count, :] = Pos3
    #     # 存图
    #     ImgTemp = pd.getSingleImageByBasler()
    #     cv2.imwrite(pathList[0] + '/' + str(count) + '.bmp', ImgTemp)
    #     # 获取旋转矢量，平移矢量
    #     ChessRot, ChessTrans = pd.getHandEyeCalibChessBoardPoseEstimate(ImgTemp, ChessPoints, ChessRow, ChessCol,
    #                                                                     Intrinsic, Distortion)
    #     ChessRotVec[count, :] = ChessRot
    #     ChessTransVec[count, :] = ChessTrans
    #     count = count + 1
    #     time.sleep(3)
    #
    # # Ry方向旋转
    # for j in range(2):
    #     Ry1 = Ry + ((-1) ** j) * theta
    #     # 避免奇异,限定范围
    #     if Ry1 < -math.pi:
    #         Ry1 = Ry1 + 2 * math.pi
    #     elif Ry1 > math.pi:
    #         Ry1 = Ry1 - 2 * math.pi
    #     # 控制Z方向不便进行转动
    #     RotMatNew = pd.getAnglesToRotaionMatrix(Rx, Ry1, Rz)
    #     RotVecNew = cv2.Rodrigues(RotMatNew)[0]
    #     pd.getUR10EMove(IPRob, PortNumberRob, Pos1[0], Pos1[1], Pos1[2], RotVecNew[0], RotVecNew[1], RotVecNew[2])
    #     time.sleep(3)
    #     Pos3 = pd.getUR10EPose(IPRob, PortNumberRob)
    #     # 记录位姿
    #     RobertPosVec[count, :] = Pos3
    #     ImgTemp = pd.getSingleImageByBasler()
    #     # 存图
    #     cv2.imwrite(pathList[0] + '/' + str(count) + '.bmp', ImgTemp)
    #     # 获取旋转矢量，平移矢量
    #     ChessRot, ChessTrans = pd.getHandEyeCalibChessBoardPoseEstimate(ImgTemp, ChessPoints, ChessRow, ChessCol,
    #                                                                     Intrinsic, Distortion)
    #     ChessRotVec[count, :] = ChessRot
    #     ChessTransVec[count, :] = ChessTrans
    #     count = count + 1
    #     time.sleep(3)

for i in range(2):
    # 控制机械臂在角度变换层面运动
    pd.getUR10EMove(IPRob, PortNumberRob, Pos0[0], Pos0[1], Pos0[2] + ((-1) ** i) * stepZ / 2, Pos0[3], Pos0[4],
                    Pos0[5])
    # 计算当前旋转角度
    Pos1 = pd.getUR10EPose(IPRob, PortNumberRob)
    RotVec = np.array([Pos1[3], Pos1[4], Pos1[5]])
    RotMat = pd.getRotVec2RotMAT(RotVec)
    Rx, Ry, Rz = pd.getRotationMatrixToAngles(RotMat)
    # 设定Rx变换值
    for j in range(3):
        Rx1 = Rx + (j - 1) * theta
        # 避免奇异,限定范围
        if Rx1 < -math.pi:
            Rx1 = Rx1 + 2 * math.pi
        elif Rx1 > math.pi:
            Rx1 = Rx1 - 2 * math.pi
        # 设定Ry变换值
        for k in range(3):
            Ry1 = Ry + (k - 1) * theta
            # 避免奇异,限定范围
            if Ry1 < -math.pi:
                Ry1 = Ry1 + 2 * math.pi
            elif Ry1 > math.pi:
                Ry1 = Ry1 - 2 * math.pi
            # 计算得到新变换矩阵
            RotMatNew = pd.getAnglesToRotaionMatrix(Rx1, Ry1, Rz)
            RotVecNew = cv2.Rodrigues(RotMatNew)[0]
            # 控制机械臂运动
            pd.getUR10EMove(IPRob, PortNumberRob, Pos1[0], Pos1[1], Pos1[2], RotVecNew[0], RotVecNew[1], RotVecNew[2])
            # 短暂暂停,避免速度过快抖动影响成像
            time.sleep(1)
            Pos3 = pd.getUR10EPose(IPRob, PortNumberRob)
            # 记录位姿
            RobertPosVec[count, :] = Pos3
            ImgTemp = pd.getSingleImageByBasler()
            # 存图
            cv2.imwrite(pathList[0] + '/' + str(count) + '.bmp', ImgTemp)
            # 获取旋转矢量，平移矢量
            ChessRot, ChessTrans = pd.getHandEyeCalibChessBoardPoseEstimate(ImgTemp, ChessPoints, ChessRow, ChessCol,
                                                                            Intrinsic, Distortion)
            ChessRotVec[count, :] = ChessRot
            ChessTransVec[count, :] = ChessTrans
            count = count + 1
            time.sleep(1)

# 采图完成，计算手眼
# 根据TOB = THB @ TCH @ TOC, 即是物体在基坐标系下的位姿 = 手在基坐标系下的位姿 * 相机在末端坐标系下的位姿 * 物体在相机中的位姿
# TOC计算
# 取旋转矩阵行数作为循环操作总数
row, col = ChessRotVec.shape
# 循环获得旋转矩阵元胞与平移矩阵元胞
RotMatObjToCam = []
TransMatObjToCam = []
for i in range(row):
    # 旋转矩阵转换
    RotMatTemp = pd.getRotVec2RotMAT(ChessRotVec[i, :])
    RotMatObjToCam.append(RotMatTemp)
    # 平移向量转换
    TransMatObjToCam.append(ChessTransVec[i, :].reshape((-1, 1)))
# TOC变形为增广矩阵,仅做验证用,验证已通过故不手机用
RTObjToCam = []
for i in range(row):
    RTTemp = np.column_stack((RotMatObjToCam[i], TransMatObjToCam[i]))
    RTTemp = np.row_stack((RTTemp, np.array([0, 0, 0, 1])))
    RTObjToCam.append(RTTemp)

# THB计算
row, col = RobertPosVec.shape
RotMatHandToBase = []
TransMatHandToBase = []
RTHandToBase = []
for i in range(row):
    # 旋转
    RotVec = np.array([RobertPosVec[i, 3], RobertPosVec[i, 4], RobertPosVec[i, 5]])
    RotMatTemp = pd.getRotVec2RotMAT(RotVec)
    RotMatHandToBase.append(RotMatTemp)
    # 平移
    TransVec = np.array([[RobertPosVec[i, 0]], [RobertPosVec[i, 1]], [RobertPosVec[i, 2]]])
    TransVec = 1000 * TransVec
    TransMatHandToBase.append(TransVec)
    # 增广,仅做验证用,验证已通过故不再计算
    RTTemp = np.column_stack((RotMatTemp, TransVec))
    RTTemp = np.row_stack((RTTemp, np.array([0, 0, 0, 1])))
    RTHandToBase.append(RTTemp)

# TCH求解
# 利用Opencv 手眼求解
RotMatCamToHand, TransMatCamToHand = cv2.calibrateHandEye(RotMatHandToBase, TransMatHandToBase, RotMatObjToCam,
                                                          TransMatObjToCam)
RTCamToHand = np.column_stack((RotMatCamToHand, TransMatCamToHand))
RTCamToHand = np.row_stack((RTCamToHand, np.array([0, 0, 0, 1])))

print('TCB标定结果：')
print(RTCamToHand)
# 标定结果
# [[-7.02780688e-01 -7.11396993e-01  3.69089232e-03  8.24687253e+01]
#  [ 7.11396534e-01 -7.02733763e-01  8.95705326e-03  3.90889146e+01]
#  [-3.77830610e-03  8.92053206e-03  9.99953073e-01  1.07119386e+02]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
RxTCH, RyTCH, RzTCH = pd.getRotationMatrixToAngles(RotMatCamToHand)
print('旋转角度, Rx =', RxTCH * 180 / np.pi, '° ', 'Ry =', RyTCH * 180 / np.pi, '° ', 'Rz =', RzTCH * 180 / np.pi,
      '°')

# TOB验证
# 通过TOB的变化量验证TCH的正确性
Rx = np.zeros((row, 1))
Ry = np.zeros((row, 1))
Rz = np.zeros((row, 1))
X = np.zeros((row, 1))
Y = np.zeros((row, 1))
Z = np.zeros((row, 1))
for i in range(row):
    RTObjToBase = RTHandToBase[i] @ RTCamToHand @ RTObjToCam[i]
    # 累计旋转角度
    RotObjToBase = RTObjToBase[0:3, 0:3]
    Rx[i], Ry[i], Rz[i] = pd.getRotationMatrixToAngles(RotObjToBase)
    # 位移
    X[i] = RTObjToBase[0, 3]
    Y[i] = RTObjToBase[1, 3]
    Z[i] = RTObjToBase[2, 3]
    # print(RTObjToBase)

print('TOB校验:')
print('Rx重复误差：', np.std(Rx, ddof=1), ' rad')
print('Ry重复误差：', np.std(Ry, ddof=1), ' rad')
print('Rz重复误差：', np.std(Rz, ddof=1), ' rad')
print('X重复误差：', np.std(X, ddof=1), ' mm')
print('Y重复误差：', np.std(Y, ddof=1), ' mm')
print('Z重复误差：', np.std(Z, ddof=1), ' mm')
