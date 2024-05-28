# -*- coding: utf-8 -*-

import cv2
import socket
import struct
import numpy as np
from time import sleep
import math


def getUR10EPose(IPAddress, PortNumber):
    """
        通过Socket调用访问UR10E当前位姿信息
    :param IPAddress: IP地址，应为字符串格式，形如 ”192.168.1.1“
    :param PortNumber: 端口号，UR机器人实时通讯端口号为30003
    :return:位姿6自由度参数，单位为m，弧度
    """
    # 办公室UR10E分配网络IP地址
    # HOST = "192.168.0.70"
    HOST = IPAddress
    # PORT = 30003
    PORT = PortNumber

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))

    dic = {'MessageSize': 'i', 'Time': 'd', 'q target': '6d', 'qd target': '6d', 'qdd target': '6d', 'I target': '6d',
           'M target': '6d', 'q actual': '6d', 'qd actual': '6d', 'I actual': '6d', 'I control': '6d',
           'Tool vector actual': '6d', 'TCP speed actual': '6d', 'TCP force': '6d', 'Tool vector target': '6d',
           'TCP speed target': '6d', 'Digital input bits': 'd', 'Motor temperatures': '6d', 'Controller Timer': 'd',
           'Test value': 'd', 'Robot Mode': 'd', 'Joint Modes': '6d', 'Safety Mode': 'd', 'empty1': '6d',
           'Tool Accelerometer values': '3d',
           'empty2': '6d', 'Speed scaling': 'd', 'Linear momentum norm': 'd', 'SoftwareOnly': 'd', 'softwareOnly2': 'd',
           'V main': 'd',
           'V robot': 'd', 'I robot': 'd', 'V actual': '6d', 'Digital outputs': 'd', 'Program state': 'd',
           'Elbow position': '3d', 'Elbow velocity': '3d', 'Safety Status': 'd'}

    data = s.recv(1500)
    # print(data)
    names = []
    ii = range(len(dic))
    for key, i in zip(dic, ii):
        fmtsize = struct.calcsize(dic[key])
        data1, data = data[0:fmtsize], data[fmtsize:]
        fmt = "!" + dic[key]
        names.append(struct.unpack(fmt, data1))
        dic[key] = dic[key], struct.unpack(fmt, data1)

    a = dic["Tool vector actual"]
    a2 = np.array(a[1])
    print('机械臂当前位姿：', a2)
    s.close()
    return a2


def getUR10EPoseWithoutPrint(IPAddress, PortNumber):
    """
        通过Socket调用访问UR10E当前位姿信息
    :param IPAddress: IP地址，应为字符串格式，形如 ”192.168.1.1“
    :param PortNumber: 端口号，UR机器人实时通讯端口号为30003
    :return:位姿6自由度参数，单位为m，弧度
    """
    # 办公室UR10E分配网络IP地址
    # HOST = "192.168.0.70"
    HOST = IPAddress
    # PORT = 30003
    PORT = PortNumber

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))

    dic = {'MessageSize': 'i', 'Time': 'd', 'q target': '6d', 'qd target': '6d', 'qdd target': '6d', 'I target': '6d',
           'M target': '6d', 'q actual': '6d', 'qd actual': '6d', 'I actual': '6d', 'I control': '6d',
           'Tool vector actual': '6d', 'TCP speed actual': '6d', 'TCP force': '6d', 'Tool vector target': '6d',
           'TCP speed target': '6d', 'Digital input bits': 'd', 'Motor temperatures': '6d', 'Controller Timer': 'd',
           'Test value': 'd', 'Robot Mode': 'd', 'Joint Modes': '6d', 'Safety Mode': 'd', 'empty1': '6d',
           'Tool Accelerometer values': '3d',
           'empty2': '6d', 'Speed scaling': 'd', 'Linear momentum norm': 'd', 'SoftwareOnly': 'd', 'softwareOnly2': 'd',
           'V main': 'd',
           'V robot': 'd', 'I robot': 'd', 'V actual': '6d', 'Digital outputs': 'd', 'Program state': 'd',
           'Elbow position': '3d', 'Elbow velocity': '3d', 'Safety Status': 'd'}

    data = s.recv(1500)
    names = []
    ii = range(len(dic))
    for key, i in zip(dic, ii):
        fmtsize = struct.calcsize(dic[key])
        data1, data = data[0:fmtsize], data[fmtsize:]
        fmt = "!" + dic[key]
        names.append(struct.unpack(fmt, data1))
        dic[key] = dic[key], struct.unpack(fmt, data1)

    a = dic["Tool vector actual"]
    a2 = np.array(a[1])
    # print('机械臂当前位姿：', a2)
    s.close()
    return a2


def getUR10EMove(IPAddress, PortNumber, x, y, z, rx, ry, rz):
    """
        通过Socket进行UR系列机器人连接
    :param IPAddress: IP地址，应为字符串格式，形如 ”192.168.1.1“
    :param PortNumber: 端口号，UR机器人实时通讯端口号为30003
    :param x: 单位 m
    :param y: 单位 m
    :param z: 单位 m
    :param rx: 单位 弧度
    :param ry: 单位 弧度
    :param rz: 单位 弧度
    :return:
    """
    # 办公室UR10E分配网络IP地址
    # HOST = "192.168.0.70"
    HOST = IPAddress
    # PORT = 30003
    PORT = PortNumber
    m = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    m.connect((HOST, PORT))
    dic = {'MessageSize': 'i', 'Time': 'd', 'q target': '6d', 'qd target': '6d', 'qdd target': '6d', 'I target': '6d',
           'M target': '6d', 'q actual': '6d', 'qd actual': '6d', 'I actual': '6d', 'I control': '6d',
           'Tool vector actual': '6d', 'TCP speed actual': '6d', 'TCP force': '6d', 'Tool vector target': '6d',
           'TCP speed target': '6d', 'Digital input bits': 'd', 'Motor temperatures': '6d', 'Controller Timer': 'd',
           'Test value': 'd', 'Robot Mode': 'd', 'Joint Modes': '6d', 'Safety Mode': 'd', 'empty1': '6d',
           'Tool Accelerometer values': '3d',
           'empty2': '6d', 'Speed scaling': 'd', 'Linear momentum norm': 'd', 'SoftwareOnly': 'd', 'softwareOnly2': 'd',
           'V main': 'd',
           'V robot': 'd', 'I robot': 'd', 'V actual': '6d', 'Digital outputs': 'd', 'Program state': 'd',
           'Elbow position': '3d', 'Elbow velocity': '3d', 'Safety Status': 'd'}

    # data = m.recv(1500)
    # names = []
    # ii = range(len(dic))
    # for key, i in zip(dic, ii):
    #     fmtsize = struct.calcsize(dic[key])
    #     data1, data = data[0:fmtsize], data[fmtsize:]
    #     fmt = "!" + dic[key]
    #     names.append(struct.unpack(fmt, data1))
    #     dic[key] = dic[key], struct.unpack(fmt, data1)
    print("移动目标：", x, ',', y, ',', z, ',', rx, ',', ry, ',', rz)
    # 预设姿态
    # print(("movel(p[" + str(x) + "," + str(y) + "," + str(z) + "," + str(rx) + "," + str(ry) + "," + str(
    #     rz) + "," + "0.1" + ",0.1)" + "\n").encode())
    # s.send(("movel(p[" + str(x) + "," + str(y) + ",0.7,-2.902,-1.202,0]," + "0.1" + ",0.1)" + "\n").encode())
    m.send(("movel(p[" + str(x) + "," + str(y) + "," + str(z) + "," + str(rx) + "," + str(ry) + "," + str(
        rz) + "]," + "0.1" + ",0.1)" + "\n").encode())
    # sleep(10)
    while (1):
        PosNow = getUR10EPoseWithoutPrint(IPAddress, PortNumber)
        Dis = math.sqrt((PosNow[0] - x) ** 2 + (PosNow[1] - y) ** 2 + (PosNow[2] - z) ** 2)
        # print(Dis)
        if Dis <= 0.0005:
            print('到位成功')
            break
    m.close()
    return


def getRotVec2RotMAT(RotVec):
    ROtMat = cv2.Rodrigues(RotVec)[0]
    return ROtMat


def getDataFromHikSmartCameraByTCP(Host, PortNumber, startMsg, Num):
    # Num为需要读取的数据个数，所有数据均需采用4.2长度设计，各数据含义需在SC MVS软件中自行注明
    # 创建客户端
    HikClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 链接客户端
    HikClient.connect((Host, PortNumber))
    # 读取客户端
    # Flag = True
    while 1:
        msgStart = startMsg
        HikClient.send(msgStart.encode('utf-8'))
        data = HikClient.recv(1024)
        # data1 = data.decode('utf-8')
        # print(data1[0])
        data = str(data, 'utf-8')
        # print(data[0])
        if data[0] == str(1):
            print('收到智能相机数据')
            break
        # else:
        # sleep(0.1)
    Data = np.zeros((Num, 1))
    # print(Data)
    for i in range(Num):
        Data[i] = data[8 * i + 2:8 * i + 9]
    # np.array(Data)
    HikClient.close()
    return Data


def getHikSwitchPlanByTCP(Host, PortNumber, SwitchCode, PlanName):
    """
        通过TCP切换海康智能相机方案
    :param Host: 相机IP地址
    :param PortNumber:端口号
    :param SwitchCode:切换语句
    :param PlanName:待切换方案名称
    :return:
    """
    # 创建客户端
    HikClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 链接服务端
    HikClient.connect((Host, PortNumber))
    # 整理发送数据
    msgSend = SwitchCode + ' ' + PlanName
    # 发送数据
    while 1:
        HikClient.send(msgSend.encode('utf-8'))
        data = HikClient.recv(1024)
        # data1 = data.decode('utf-8')
        # print(data1[0])
        data = str(data, 'utf-8')
        # print(data[0])
        if data == str('ok'):
            print('成功切换至方案： ', PlanName)
            break


def getRobertInsert(Host, PortNumber, Width, Height, disCamX, disCamY, RotationEndToCam, dRuler, PosNow, CirclePos,
                    DisForward):
    """
        根据相机检测位置,控制机械臂移动
    :param Host:
    :param PortNumber:
    :param Width:
    :param Height:
    :param disCamX:
    :param disCamY:
    :param RotationEndToCam:
    :param dRuler:
    :param PosNow:
    :param CirclePos:
    :param DisForward:
    :return:
    """
    # 计算旋转矩阵
    RotVec = np.array([PosNow[3], PosNow[4], PosNow[5]])
    RotBaseToEnd = getRotVec2RotMAT(RotVec)
    # 循环控制运动
    num = len(CirclePos[:, 0])
    for i in range(num):
        # 平面内移动
        MoveX = (CirclePos[i, 0] - Width / 2) * dRuler + disCamX
        MoveY = (CirclePos[i, 1] - Height / 2) * dRuler + disCamY
        Move1 = np.array([[MoveX], [MoveY], [0]])
        Move1 = Move1 / 1000
        MoveCam = RotBaseToEnd @ RotationEndToCam @ Move1
        getUR10EMove(Host, PortNumber, PosNow[0] + MoveCam[0], PosNow[1] + MoveCam[1], PosNow[2] + MoveCam[2],
                     PosNow[3], PosNow[4], PosNow[5])
        # sleep(20)  # 示教使用
        # 垂直方向移动
        Move2 = np.array([[0], [0], [DisForward]])
        Move2 = Move2 / 1000
        MoveGri = RotBaseToEnd @ Move2
        getUR10EMove(Host, PortNumber, PosNow[0] + MoveGri[0] + MoveCam[0], PosNow[1] + MoveGri[1] + MoveCam[1],
                     PosNow[2] + MoveGri[2] + MoveCam[2], PosNow[3], PosNow[4], PosNow[5])
        sleep(10)
        # 回示教平面
        getUR10EMove(Host, PortNumber, PosNow[0] + MoveCam[0], PosNow[1] + MoveCam[1], PosNow[2] + MoveCam[2],
                     PosNow[3], PosNow[4], PosNow[5])


def getRobertMoveInPlane(Host, PortNumber, Width, Height, disCamX, disCamY, RotationEndToCam, dRuler, PosNow,
                         CirclePos):
    """
        根据相机检测位置,控制机械臂在平面内移动
    :param Host:
    :param PortNumber:
    :param Width:
    :param Height:
    :param disCamX:
    :param disCamY:
    :param RotationEndToCam:
    :param dRuler:
    :param PosNow:
    :param CirclePos:
    :return:
    """
    # 计算旋转矩阵
    RotVec = np.array([PosNow[3], PosNow[4], PosNow[5]])
    RotBaseToEnd = getRotVec2RotMAT(RotVec)
    # 循环控制运动
    # print(CirclePos.shape[0])
    try:
        num = len(CirclePos[:, 0])
    except:
        # 仅存在1行数据时，会从数据变为列表，导致后续对CirclePos调用出错，因此需重新定义
        num = 1
        CirclePos = np.array([[CirclePos[0], CirclePos[1]]])
    for i in range(num):
        # print(i)
        # 平面内移动
        MoveX = (CirclePos[i, 0] - Width / 2) * dRuler + disCamX
        # print(MoveX)
        MoveY = (CirclePos[i, 1] - Height / 2) * dRuler + disCamY
        # print(MoveY)
        Move1 = np.array([[MoveX], [MoveY], [0]])
        Move1 = Move1 / 1000
        MoveCam = RotBaseToEnd @ RotationEndToCam @ Move1
        getUR10EMove(Host, PortNumber, PosNow[0] + MoveCam[0], PosNow[1] + MoveCam[1], PosNow[2] + MoveCam[2],
                     PosNow[3], PosNow[4], PosNow[5])
        sleep(2)


def getRobertMovePerpendicularToPlane(Host, PortNumber, PosNow, DisForward):
    """
        根据相机检测位置,控制机械臂进行垂直（往复）运动
    :param Host:
    :param PortNumber:
    :param PosNow:
    :param DisForward:
    :return:
    """
    # 计算旋转矩阵
    RotVec = np.array([PosNow[3], PosNow[4], PosNow[5]])
    RotBaseToEnd = getRotVec2RotMAT(RotVec)
    # 垂直方向移动
    Move2 = np.array([[0], [0], [DisForward]])
    Move2 = Move2 / 1000
    MoveGri = RotBaseToEnd @ Move2
    getUR10EMove(Host, PortNumber, PosNow[0] + MoveGri[0], PosNow[1] + MoveGri[1], PosNow[2] + MoveGri[2],
                 PosNow[3], PosNow[4], PosNow[5])
    sleep(5)
    # 回示教平面
    getUR10EMove(Host, PortNumber, PosNow[0], PosNow[1], PosNow[2], PosNow[3], PosNow[4], PosNow[5])


def getRotationMatrixToAngles(RotMat):
    """
        根据旋转矩阵输出旋转角
    :param RotMat: 旋转矩阵
    :return: Rx,Ry,Rz
    """
    sy = math.sqrt(RotMat[0, 0] * RotMat[0, 0] + RotMat[1, 0] * RotMat[
        1, 0])  # 矩阵元素下标都从0开始（对应公式中是sqrt(r11*r11+r21*r21)），sy=sqrt(cosβ*cosβ)

    singular = sy < 1e-6  # 判断β是否为正负90°

    if not singular:  # β不是正负90°
        x = math.atan2(RotMat[2, 1], RotMat[2, 2])
        y = math.atan2(-RotMat[2, 0], sy)
        z = math.atan2(RotMat[1, 0], RotMat[0, 0])
    else:  # β是正负90°
        x = math.atan2(-RotMat[1, 2], RotMat[1, 1])
        y = math.atan2(-RotMat[2, 0], sy)  # 当z=0时，此公式也OK，上面图片中的公式也是OK的
        z = 0

    return x, y, z


def getAnglesToRotaionMatrix(Rx, Ry, Rz):
    """
        利用弧度制的角度转换为旋转矩阵
    :param Rx:旋转角
    :param Ry:
    :param Rz:
    :return:
    """
    # 计算旋转矩阵的X分量
    RotMatX = np.array([[1, 0, 0],
                        [0, math.cos(Rx), -math.sin(Rx)],
                        [0, math.sin(Rx), math.cos(Rx)]])

    # 计算旋转矩阵的Y分量
    RotMatY = np.array([[math.cos(Ry), 0, math.sin(Ry)],
                        [0, 1, 0],
                        [-math.sin(Ry), 0, math.cos(Ry)]])

    # 计算旋转矩阵的Z分量
    RotMatZ = np.array([[math.cos(Rz), -math.sin(Rz), 0],
                        [math.sin(Rz), math.cos(Rz), 0],
                        [0, 0, 1]])
    RotMat = RotMatZ @ RotMatY @ RotMatX
    return RotMat

