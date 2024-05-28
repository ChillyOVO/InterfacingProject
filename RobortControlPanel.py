import numpy as np
import cv2
import PDDTVisionToolBox as pd

# 机械臂端口设置
IPRob = "192.168.0.101"
PortNumberRob = 30003

# 获取当前位姿
PosNow = pd.getUR10EPose(IPRob, PortNumberRob)

# 手动控制运动
# pd.getUR10EMove(IPRob, PortNumberRob, 0.415, 0.105, 0.770, 2.902, 1.202, 0)
