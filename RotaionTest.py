import numpy as np
import math
import cv2

import PDDTVisionToolBox as pd

Vec = np.array([[-2.901], [-1.202], [0]])
RotBE = pd.getRotVec2RotMAT(Vec)
# print(RotBE)
# 末端到相机
RotEndToCamera = np.array([[-0.707, -0.707, 0],
                           [0.707, -0.707, 0],
                           [0, 0, 1]])
# 相机绕轴旋转
cosTheta = math.cos(-45 / 180 * np.pi)
sinTheta = math.sin(-45 / 180 * np.pi)

RotAxisY = np.array([[cosTheta, 0, -sinTheta],
                     [0, 1, 0],
                     [sinTheta, 0, cosTheta]])
# BaseToNewEnd
R = RotBE @ RotEndToCamera @ RotAxisY @ (RotEndToCamera.T)
print(R)
Vec = cv2.Rodrigues(R)[0]
print(Vec)