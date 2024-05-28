# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import stereoConfigure
import open3d as o3d


# 预处理
def preprocess(img1, img2):
    # 彩色图->灰度图
    if (img1.ndim == 3):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 通过OpenCV加载的图像通道顺序是BGR
    if (img2.ndim == 3):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 直方图均衡
    img1 = cv2.equalizeHist(img1)   # 直方图均衡化，不是很必须，而且影响结果
    img2 = cv2.equalizeHist(img2)
    # img1 = cv2.normalize(img1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # img2 = cv2.normalize(img2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return img1, img2


# 消除畸变
def undistortion(image, camera_matrix, dist_coeff):
    undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)

    return undistortion_image


# 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
# @param：config是一个类，存储着双目标定的参数:config = stereoconfig.stereoCamera()
def getRectifyTransform(height, width, config):
    # 读取内参和外参
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T

    # 计算校正变换
    # R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
    #                                                   (width, height), R, T, alpha=0)
    #
    # map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    # map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                      (width, height), R, T, alpha=1.1)

    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q


# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)

    return rectifyed_img1, rectifyed_img2


# 立体校正检验----画线
def draw_line(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    # 绘制等间距平行线
    line_interval = 500  # 直线间隔：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)

    return output


# 视差计算
def stereoMatchSGBM(left_image, right_image, down_scale=False):
    # SGBM匹配参数设置
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 15
    paraml = {'minDisparity': 0,
              'numDisparities': 64 * 16,
              'blockSize': blockSize,
              'P1': 8 * img_channels * blockSize ** 2,
              'P2': 32 * img_channels * blockSize ** 2,
              'disp12MaxDiff': -1,
              'preFilterCap': 0,
              'uniquenessRatio': 0,
              'speckleWindowSize': 100,
              'speckleRange': 15,
              # 'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
              'mode': cv2.STEREO_SGBM_MODE_HH
              }

    # 构建SGBM对象
    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)
    # 计算视差图
    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = left_matcher.compute(left_image, right_image)
        disparity_right = right_matcher.compute(right_image, left_image)
    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]

        disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right

    # 真实视差（因为SGBM算法得到的视差是×16的）
    # disparity_left = cv2.medianBlur(disparity_left, 5)
    trueDisp_left = disparity_left.astype(np.float32) / 16.0
    trueDisp_right = disparity_right.astype(np.float32) / 16.0

    # trueDisp_left = cv2.medianBlur(trueDisp_left, 7)
    # trueDisp_left, buf = cv2.filterSpeckles(trueDisp_left, 0, 100, 20)

    return trueDisp_left, trueDisp_right


# def getDepthMapWithQ(disparityMap: np.ndarray, Q: np.ndarray) -> np.ndarray:
#     points_3d = cv2.reprojectImageTo3D(disparityMap, Q)
#     depthMap = points_3d[:, :, 2]
#     reset_index = np.where(np.logical_or(depthMap < 4000.0, depthMap > 65535.0))
#     depthMap[reset_index] = 0
#
#     return depthMap.astype(np.float32)


def getDepthMapWithConfig(disparityMap: np.ndarray, config: stereoConfigure.stereoCamera) -> np.ndarray:
    fb = config.cam_matrix_left[0, 0] * (-config.T[0])
    doffs = config.doffs
    depthMap = np.divide(fb, disparityMap + doffs)
    reset_index = np.where(np.logical_or(depthMap < 100.0, depthMap > 500.0))  # 改动此行可使得深度图可显示部分 8000
    depthMap[reset_index] = 0
    reset_index2 = np.where(disparityMap < 0.0)
    depthMap[reset_index2] = 0
    return depthMap.astype(np.float32)


if __name__ == '__main__':
    # 办公室图像
    iml = cv2.imread('./20231023/SGBMTest/Left2.bmp', 1)  # 左图
    imr = cv2.imread('./20231023/SGBMTest/Right2.bmp', 1)  # 右图

    # 预处理
    # iml = cv2.GaussianBlur(iml, [3, 3], 0)
    # imr = cv2.GaussianBlur(imr, [3, 3], 0)

    if (iml is None) or (imr is None):
        print("Error: Images are empty, please check your image's path!")
        sys.exit(0)
    height, width = iml.shape[0:2]

    # 读取相机内参和外参
    # 使用之前先将标定得到的内外参数填写到stereoConfigure.py中的StereoCamera类中
    config = stereoConfigure.stereoCamera()
    # config.setMiddleBurryParams()
    print(config.cam_matrix_left)

    # 立体校正
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)  # 获取用于畸变校正和立体校正的映射矩阵以及用于计算像素空间坐标的重投影矩阵
    iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
    print(Q)

    # 绘制等间距平行线，检查立体校正的效果
    cv2.imwrite('./iml_rectified.png', iml_rectified)
    cv2.imwrite('./imr_rectified.png', imr_rectified)
    line = draw_line(iml_rectified, imr_rectified)
    cv2.imwrite('./rectification.png', line)

    # 立体匹配
    iml, imr = preprocess(iml_rectified, imr_rectified)  # 预处理，一般可以削弱光照不均的影响，不做也可以
    # iml = iml_rectified
    # imr = imr_rectified  # 预处理，一般可以削弱光照不均的影响，不做也可以

    disp, _ = stereoMatchSGBM(iml, imr, False)
    cv2.imwrite('./disaprity.png', disp * 4)

    # 计算深度图
    # depthMap = getDepthMapWithQ(disp, Q)
    depthMap = getDepthMapWithConfig(disp, config)
    minDepth = np.min(depthMap)
    maxDepth = np.max(depthMap)
    print(minDepth, maxDepth)
    depthMapVis = (255.0 * (depthMap - minDepth)) / (maxDepth - minDepth)
    depthMapVis = depthMapVis.astype(np.uint8)
    cv2.namedWindow("DepthMap", 0)
    cv2.resizeWindow("DepthMap", 800, 600)
    cv2.imshow("DepthMap", depthMapVis)
    cv2.waitKey(0)

    # 使用open3d库绘制点云
    colorImage = o3d.geometry.Image(iml)
    depthImage = o3d.geometry.Image(depthMap)
    rgbdImage = o3d.geometry.RGBDImage().create_from_color_and_depth(colorImage, depthImage, depth_scale=10000.0,
                                                                     depth_trunc=np.inf)

    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    # fx = Q[2, 3]
    # fy = Q[2, 3]
    # cx = Q[0, 3]
    # cy = Q[1, 3]
    fx = config.cam_matrix_left[0, 0]
    fy = config.cam_matrix_left[1, 1]
    cx = config.cam_matrix_left[0, 2]
    cy = config.cam_matrix_left[1, 2]
    print(fx, fy, cx, cy)
    intrinsics.set_intrinsics(width, height, fx=fx, fy=fy, cx=cx, cy=cy)
    extrinsics = np.array([[1., 0., 0., 0.],
                           [0., 1., 0., 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]])
    pointcloud = o3d.geometry.PointCloud().create_from_rgbd_image(rgbdImage, intrinsic=intrinsics, extrinsic=extrinsics)
    o3d.io.write_point_cloud("PointCloud.pcd", pointcloud=pointcloud)
    o3d.visualization.draw_geometries([pointcloud], width=720, height=480)
    sys.exit(0)

    # # 计算像素点的3D坐标（左相机坐标系下）
    # points_3d = cv2.reprojectImageTo3D(disp, Q)  # 参数中的Q就是由getRectifyTransform()函数得到的重投影矩阵
    #
    # # 构建点云--Point_XYZRGBA格式
    # pointcloud = DepthColor2Cloud(points_3d, iml)
    #
    # # 显示点云
    # view_cloud(points_3d)
