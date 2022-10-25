import cv2
import os
import numpy as np
import tensorflow as tf
import string

def addImage (backgroud, path):
    img1 = cv2.imread(backgroud)  # rows = 302 cols = 304
    img2 = cv2.imread(path)  # rows = 192 cols = 200


    img1 = img1[10:330, 10:330]  # 裁剪坐标为[y0:y1, x0:x1]
    img2 = img2[10:330, 10:330]
    #img1 = cv2.resize(img1, (512,512), interpolation=cv2.INTER_CUBIC)
    #img2 = cv2.resize(img2, (512, 512), interpolation=cv2.INTER_CUBIC)

    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]

    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # BGR到GRAY
    ret, mask = cv2.threshold(img2gray, 175, 255, cv2.THRESH_BINARY)  # 将灰度图img2gray中灰度值小于175的点置0，灰度值大于175的点置255
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)

    dst = cv2.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst

    #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #ret, img1 = cv2.threshold(img1, 220, 255, cv2.THRESH_BINARY)
    name = path.split('/')
    cv2.imwrite('dataset/dataset/test/b-'+ name[6], img1)

def get_img(file_dir):
    # step1：获取路径下所有的图片路径名，存放到对应的列表中，同时贴上标签，存放到label列表中。
    indir = []
    for file in os.listdir(file_dir + '/dataset/dataset/new'):
        indir.append(file_dir + '/dataset/dataset/new/' + file)
    return indir

def dct(path):
    img = cv2.imread(path)
    img = img[:, :, 0]  # 获取rgb通道中的一个
    ret, img = cv2.threshold(img, 175, 255, cv2.THRESH_BINARY)
    img = np.float32(img)  # 将数值精度调整为32位浮点型
    img_dct = cv2.dct(img)  # 使用dct获得img的频域图像
    img_dct = img_dct * 255
    name = path.split('/')
    cv2.imwrite('test/output/' + name[5], img_dct)

def idct(path):
    img_dct = cv2.imread(path)
    img_dct = img_dct[:, :, 0]  # 获取rgb通道中的一个
    img_dct = img_dct / 255
    img_recor2 = cv2.idct(img_dct)  # 使用反dct从频域图像恢复出原图像(有损)
    name = path.split('/')
    cv2.imwrite('test/reresult/' + name[5], 255-img_recor2*255)

def cut(path):
    img = cv2.imread(path)
    name = path.split('/')
    cv2.imwrite('dataset/dataset/output/'+name[5], img[10:330,10:330])

if __name__ ==  '__main__':
    indir = get_img('E:/Python project/add')
    backgroud = 'b.jpg'

    for path in indir:
        addImage(backgroud, path)
        #idct(path)
        #cut(path)

