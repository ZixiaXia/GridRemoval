# coding=UTF-8
import os
import tensorflow as tf
import cv2
import numpy as np
import pywt as pw

#��window������ʱ����룬Linux����ʱɾ��
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
#--------------------------------------

def get_img(file_dir):
    # step1����ȡ·�������е�ͼƬ·��������ŵ���Ӧ���б��У�ͬʱ���ϱ�ǩ����ŵ�label�б��С�
    inimg = []
    outimg = []
    for file in os.listdir(file_dir + 'dataset/input'):
        dir = file_dir + 'dataset/input/' + file
        img = read(dir)
        inimg.append(img)
    for file in os.listdir(file_dir + 'dataset/output'):
        dir = file_dir + 'dataset/output/' + file
        img = read(dir)
        outimg.append(img)
    return inimg, outimg

def normalization(input):
    max = tf.reduce_max(input)
    min = tf.reduce_min(input)
    return (input-min)/(max-min)

def read(dir):
    img = cv2.imread(dir, 0)
    #img = img[0:41, 0:41]
    '''tcoeffs = pw.dwt2(img, 'db1')
    tcA, (tcH, tcV, tcD) = tcoeffs
    tcA = tcA.astype(np.float32) / 255
    tcH = tcH.astype(np.float32) / 255
    tcV = tcV.astype(np.float32) / 255
    tcD = tcD.astype(np.float32) / 255
    test_temp = np.array([tcA, tcH, tcV, tcD])
    test_elem = np.rollaxis(test_temp, 0, 3)
    test_data = test_elem[np.newaxis, ...]
    return test_elem'''
    img = np.float32(img)  # ����ֵ���ȵ���Ϊ32λ������
    img_dct = cv2.dct(img)  # ʹ��dct���img��Ƶ��ͼ��
    img_dct = np.array([img_dct])
    img_dct = np.rollaxis(img_dct, 0, 3)
    img_dct = np.float32(img_dct)
    #img_dct = img_dct[np.newaxis, ...]
    #img_dct = np.asarray(img_dct)
    return img_dct

def get_batch(train_dir, h, w, batch_size, num_threads, capacity):
    inimg, outimg = get_img(train_dir)
    inimg = np.asarray(inimg)
    outimg = np.asarray(outimg)
    train = tf.data.Dataset.from_tensor_slices({'x': inimg,'y': outimg})
    train = train.shuffle(buffer_size=2).repeat(4).batch(16)
    return train

