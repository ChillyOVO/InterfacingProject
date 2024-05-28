import os
import cv2
import numpy as np
import HandEyeCalibTest


# HandEye = HandEyeCalibTest.EyeToHandCalibration


class ReadImagesByOrder:
    """
        按照顺序批量读图
    """

    def __init__(self):
        """
            初始化
        """
        # self.ImageArray = []
        # self.TransObjectToCamera = []
        # self.Rotation = []
        # self.Translation = []

    def getImages(self, FolderName):
        """
            在文件夹内读图
        :param FolderName: 文件夹名字
        :return:
        """
        # 图像所在文件夹，需在当前文件夹下
        ImgList = os.listdir(r"./" + FolderName)
        # 规定文件夹内图像命名格式必须是 0.bmp
        ImgList.sort(key=lambda x: int(x.split('.')[0]))
        # print(ImgList)
        # 初始化列表
        ImageArray = []
        # TransObjectToCamera = []
        # Rotation = []
        # Translation = []
        for count in range(0, len(ImgList)):
            FileName = ImgList[count]
            Img = cv2.imread(FolderName + "/" + FileName)
            # cv2.imshow("m", Img)
            # cv2.waitKey(0)
            ImageArray.append(Img)
            # TOC, R, t = HandEye.getTransObjectToCamera(self,Img)
            # TransObjectToCamera.append(TOC)
            # Rotation.append(R)
            # Translation.append(t)
        return ImageArray
