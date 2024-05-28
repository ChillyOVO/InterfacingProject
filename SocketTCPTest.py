import socket
import numpy as np
import PDDTVisionToolBox as pd

# IP-本机
Host = '192.168.0.46'
# 端口号
PortNumber = 8192
# 创建TCP Server端
# HKTCPServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 连接服务器端
# HKTCPServer.connect((Host, PortNumber))
# 输入信息
# msg = "Str"
# HKTCPServer.send(msg)
# HKTCPServer.close()


code = 'switch'
PlanName1 = 'PlaneTest'
PlanName2 = 'CircleTest'
PlanName3 = 'Kongzhihe'
pd.getHikSwitchPlanByTCP(Host, PortNumber, code, PlanName3)
