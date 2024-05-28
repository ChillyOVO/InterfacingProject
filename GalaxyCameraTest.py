import cv2
import math
import numpy as np
import gxipy as gx
# import open3d as o3d
# from skimage import morphology

def getImage(self):
        """
            使用大恒图像 Galaxy MER-500-14GC 双相机获取双目图像
        :return:leftImage,rightImage
        """
        # 设置曝光时间,单位 ns
        expTime = 50000.0
        # 设置图像增益,建议默认为1
        gain = 1.0
        # 依据SN码区分左右相机
        SNLeft = 'GX0170062878'
        SNRight = 'GX0170062876'
        # 初始化类,获取并检查相机
        dev_manager = gx.DeviceManager()
        devNum, devSNList = dev_manager.update_device_list()
        if devNum == 0:
            print("No Camera Founded, Please Check The Camera Link")
            return
        # 初始化相机
        camL = dev_manager.open_device_by_sn(SNLeft)
        camR = dev_manager.open_device_by_sn(SNRight)

        # 读取左相机
        # 设置参数
        camL.TriggerMode.set(gx.GxSwitchEntry.OFF)
        camL.ExposureTime.set(expTime)
        camL.Gain.set(gain)
        # 优化查找表
        if camL.GammaParam.is_readable():
            gamma_value = camL.GammaParam.get()
            gamma_lut = gx.Utility.get_gamma_lut(gamma_value)
        else:
            gamma_lut = None
        if camL.ContrastParam.is_readable():
            contrast_value = camL.ContrastParam.get()
            contrast_lut = gx.Utility.get_contrast_lut(contrast_value)
        else:
            contrast_lut = None
        if camL.ColorCorrectionParam.is_readable():
            color_correction_param = camL.ColorCorrectionParam.get()
        else:
            color_correction_param = 0
        # 开始读图像
        camL.stream_on()
        # get raw image
        raw_image = camL.data_stream[0].get_image()
        # get RGB image from raw image
        rgb_image = raw_image.convert("RGB")
        # improve image quality
        rgb_image.image_improvement(color_correction_param, contrast_lut, gamma_lut)
        # create numpy array with data from raw image
        ImgL = rgb_image.get_numpy_array()
        # 关相机
        camL.stream_off()

        # 读取右相机
        # 设置参数
        camR.TriggerMode.set(gx.GxSwitchEntry.OFF)
        camR.ExposureTime.set(expTime)
        camR.Gain.set(gain)
        # 优化查找表
        if camR.GammaParam.is_readable():
            gamma_value = camR.GammaParam.get()
            gamma_lut = gx.Utility.get_gamma_lut(gamma_value)
        else:
            gamma_lut = None
        if camR.ContrastParam.is_readable():
            contrast_value = camR.ContrastParam.get()
            contrast_lut = gx.Utility.get_contrast_lut(contrast_value)
        else:
            contrast_lut = None
        if camR.ColorCorrectionParam.is_readable():
            color_correction_param = camR.ColorCorrectionParam.get()
        else:
            color_correction_param = 0
        # 开始读图像
        camR.stream_on()
        raw_image = camR.data_stream[0].get_image()
        rgb_image = raw_image.convert("RGB")
        rgb_image.image_improvement(color_correction_param, contrast_lut, gamma_lut)
        ImgR = rgb_image.get_numpy_array()
        camR.stream_off()

        # 关相机
        camL.close_device()
        camR.close_device()
        return ImgL, ImgR