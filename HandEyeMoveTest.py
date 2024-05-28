import time
import cv2
import math
import numpy as np
import PDDTVisionToolBox as pd

# 通讯设置
# 设定ur机械臂IP地址及端口号
IPRob = "192.168.0.101"
PortNumberRob = 30003

# 相机标定参数
Intrinsic = np.array([[4310.70617677370, 0, 1946.96880448200],
                      [0, 4310.22880285837, 1060.73807866207],
                      [0, 0, 1]])
Distortion = np.array([-0.109633688715862, 0.106474318809890, 0, 0])
# 初始位置
PosOri = np.array([[4.03781753e-01, 9.81425346e-02, 7.20470540e-01, -2.9022, -1.2000, 0]])
pd.getUR10EMove(IPRob, PortNumberRob, 4.03781753e-01, 9.81425346e-02, 7.20470540e-01, -2.9022, -1.2000, 0)
# 手动到合理位置
Pos0 = pd.getUR10EPose(IPRob, PortNumberRob)
# print(Pos0)
# 运动规划 3*3*3范围内格移动距离5~15mm
# 存储点规划
PosList = []
# 存图规划
fileName = 'Logs_of_Images'
childFileName = 'Pic'
path0, pathList = pd.getDirPathOfLogsbYTime(fileName, childFileName, 2)
# # 初始点
# ImgTemp = pd.getSingleImageByBasler()
# Img = cv2.undistort(ImgTemp, Intrinsic, Distortion)
count = 0
# cv2.imwrite(path0 + '/' + str(count) + '.bmp', Img)
# 循环运动
stepX = 0.015
stepY = 0.015
stepZ = 0.010
theta = 5 / 180 * math.pi
for i in range(3):
    # 控制Z轴
    # stepZ = 0.010
    pd.getUR10EMove(IPRob, PortNumberRob, Pos0[0], Pos0[1], Pos0[2] + (i - 1) * stepZ, Pos0[3], Pos0[4], Pos0[5])
    Pos1 = pd.getUR10EPose(IPRob, PortNumberRob)
    for j in range(3):
        # 控制X轴
        # stepX = 0.010
        pd.getUR10EMove(IPRob, PortNumberRob, Pos1[0] + (j - 1) * stepX, Pos1[1], Pos1[2], Pos1[3], Pos1[4], Pos1[5])
        Pos2 = pd.getUR10EPose(IPRob, PortNumberRob)
        for k in range(3):
            # 控制Y轴
            # stepY = 0.010
            pd.getUR10EMove(IPRob, PortNumberRob, Pos2[0], Pos2[1] + (k - 1) * stepY, Pos2[2], Pos2[3], Pos2[4],
                            Pos2[5])
            Pos3 = pd.getUR10EPose(IPRob, PortNumberRob)
            PosList.append(Pos3)
            ImgTemp = pd.getSingleImageByBasler()
            cv2.imwrite(pathList[0] + '/' + str(count) + '.bmp', ImgTemp)
            Img = cv2.undistort(ImgTemp, Intrinsic, Distortion)
            cv2.imwrite(pathList[1] + '/' + str(count) + '.bmp', Img)
            count = count + 1
            print('第', count, '次存图成功')

    # 旋转角度采图
    PosNow = pd.getUR10EPose(IPRob, PortNumberRob)
    RotVec = np.array([PosNow[3], PosNow[4], PosNow[5]])
    RotMat = pd.getRotVec2RotMAT(RotVec)
    Rx, Ry, Rz = pd.getRotationMatrixToAngles(RotMat)
    for j in range(2):
        # 控制Rx
        Rx1 = Rx + ((-1) ** j) * theta
        if Rx1 < -math.pi:
            Rx1 = Rx1 + 2 * math.pi
        elif Rx1 > math.pi:
            Rx1 = Rx1 - 2 * math.pi
        # 控制Z方向不便进行转动
        RotMatNew = pd.getAnglesToRotaionMatrix(Rx1, Ry, Rz)
        RotVecNew = cv2.Rodrigues(RotMatNew)[0]
        pd.getUR10EMove(IPRob, PortNumberRob, PosNow[0], PosNow[1], PosNow[2], RotVecNew[0], RotVecNew[1], RotVecNew[2])
        time.sleep(5)
        Pos3 = pd.getUR10EPose(IPRob, PortNumberRob)
        PosList.append(Pos3)
        time.sleep(3)
        ImgTemp = pd.getSingleImageByBasler()
        cv2.imwrite(pathList[0] + '/' + str(count) + '.bmp', ImgTemp)
        Img = cv2.undistort(ImgTemp, Intrinsic, Distortion)
        cv2.imwrite(pathList[1] + '/' + str(count) + '.bmp', Img)
        count = count + 1
        print('第', count, '次存图成功')
    time.sleep(5)
    for j in range(2):
        # 控制Rx
        Ry1 = Ry + ((-1) ** j) * theta
        if Ry1 < -math.pi:
            Ry1 = Ry1 + 2 * math.pi
        elif Ry1 > math.pi:
            Ry1 = Ry1 - 2 * math.pi
        # 控制Z方向不便进行转动
        RotMatNew = pd.getAnglesToRotaionMatrix(Rx, Ry1, Rz)
        RotVecNew = cv2.Rodrigues(RotMatNew)[0]
        pd.getUR10EMove(IPRob, PortNumberRob, PosNow[0], PosNow[1], PosNow[2], RotVecNew[0], RotVecNew[1], RotVecNew[2])
        time.sleep(5)
        Pos3 = pd.getUR10EPose(IPRob, PortNumberRob)
        PosList.append(Pos3)
        ImgTemp = pd.getSingleImageByBasler()
        cv2.imwrite(pathList[0] + '/' + str(count) + '.bmp', ImgTemp)
        Img = cv2.undistort(ImgTemp, Intrinsic, Distortion)
        cv2.imwrite(pathList[1] + '/' + str(count) + '.bmp', Img)
        count = count + 1
        print('第', count, '次存图成功')
        time.sleep(3)

print('PosList:', PosList)
