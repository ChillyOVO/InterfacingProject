# -*- coding:utf-8 -*-
import qrcode
import pyzbar.pyzbar as pyzbar
import sys
import time
import cv2

# 生成信息
info = '''
Number: 001
Position: (0000,0000)
'''

# 生成二维码，默认格式
timgCodingStart0 = time.time()
img = qrcode.make(info)
print('Code Cost: ', time.time() - timgCodingStart0, 's')
# 显示
img.show()
# 保存
img.save('Test1.jpg')
# 读入
codeImg = cv2.imread('Test1.jpg')

# opencv解码
timeDecodeOpenCVStart = time.time()
# 初始化Opencv解码器
Decoder = cv2.QRCodeDetector()
# 解码
codeInfo, points, straight_qrcode = Decoder.detectAndDecode(codeImg)
print('OpenCV Decode Cost:', time.time() - timeDecodeOpenCVStart, 's')
print('codeInfo:', codeInfo)

# pyzbar解码
timeDecodePyZbarStart = time.time()
# 初始化PyZbar解码器
decodedImg = pyzbar.decode(codeImg)
# 解码
codeData = decodedImg[0].data
print('PyZbar Decode Cost:', time.time() - timeDecodePyZbarStart, 's')
print('codeInfo:', codeData.decode())
