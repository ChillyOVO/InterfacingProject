import cv2
import socket
import numpy as np
import PDDTVisionToolBox as pd

# 设定海康相机IP地址及端口号
IPCam = "192.168.1.5"
PortNumberCam = 8192
# 通讯触发信号
msgStart = '123'
# 创建客户端
HikClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 链接客户端
HikClient.connect((IPCam, PortNumberCam))
# 读取客户端
# Flag = True
while 1:
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
Num = 1
Data = np.zeros((Num, 1))
# print(Data)
for i in range(Num):
    Data[i] = data[8 * i + 2:8 * i + 9]
# np.array(Data)
print(Data)
HikClient.close()
