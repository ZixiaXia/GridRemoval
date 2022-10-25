from __future__ import print_function
import cv2
import pywt as pw
import numpy as np
import glob, os
import sys
import time
import h5py
import random

print ('\nDeveloped by Tiantong for NTIER CVPR 2017 SR Competition, team iPAL\n'
       'Without running the training data generator, you CAN still use the training framework on demo training data set.\n'
       'Please refer to ../WvSR.py for training framework on demo training data set.\n')

Wv = 'db1'
HR_DATA_PATH = './output/'
##################################
SCALE = 2
LR_DATA_PATH = './input/'
DATA_800_PATH = './DATA_800_160_h5_f32_x2/'
###################################
# SCALE = 3
# LR_DATA_PATH = './DIV2K_train_LR_bX3X3_lum/'
# DATA_800_PATH = './DATA_800_160_h5_f32_x3/'
###################################
# SCALE = 4
# LR_DATA_PATH = './DIV2K_train_LR_bX4X4_lum/'
# DATA_800_PATH = './DATA_800_160_h5_f32_x4/'
###################################
print ('Before using python to generate training data:\n'
       'please use GenrateLRTrainData.m to store enlarged training LR in folder %s.\n' %
       LR_DATA_PATH)
STRIDE = 160
SIZE_INPUT = 160
SIZE_TARGET = SIZE_INPUT
PADDING = abs(int((SIZE_INPUT - SIZE_TARGET) / 2))
# PROCESSING EVERY TRAINING IMAGES################################################
i = 1
count = 0
print('>>Start processing single image:')
for HR_img_name in glob.glob(HR_DATA_PATH + '*.jpg'):
    # for HR_img_name in ['./DIV2K_train_HR_lum/]:
    h5fw = h5py.File(str('./DATA_800_' + str(SIZE_INPUT) + '_h5_f32_x'+str(SCALE)+'/Wv4_' + str(SIZE_INPUT) + '_' + str(i) + '_x' + str(SCALE) +'.h5'), 'w')
    print('>>Processing HR image' + str(HR_img_name))
    imgHR = cv2.imread(HR_img_name, 0)
    hcoeffs = pw.dwt2(imgHR, Wv)
    hcA, (hcH, hcV, hcD) = hcoeffs
    hcA = hcA.astype(np.float32) / 255
    hcH = hcH.astype(np.float32) / 255
    hcV = hcV.astype(np.float32) / 255
    hcD = hcD.astype(np.float32) / 255

    LR_img_name = HR_img_name.split('/')[2]
    print('	    LR image' + str(LR_DATA_PATH + LR_img_name))
    imgLR = cv2.imread(LR_DATA_PATH + LR_img_name, 0)
    lcoeffs = pw.dwt2(imgLR, Wv)
    lcA, (lcH, lcV, lcD) = lcoeffs 
    lcA = lcA.astype(np.float32) / 255
    lcH = lcH.astype(np.float32) / 255
    lcV = lcV.astype(np.float32) / 255
    lcD = lcD.astype(np.float32) / 255

    input_temp = np.array([lcA, lcH, lcV, lcD])
    input_elem = np.rollaxis(input_temp, 0, 3)
    # print(input_elem.shape) 
    #out_temp = np.array([hcA - lcA, hcH - lcH, hcV - lcV, hcD - lcD])
    out_temp = np.array([hcA, hcH, hcV, hcD])
    target_elem = np.rollaxis(out_temp, 0, 3)
    # print(target_elem.shape)

    (hei, wid) = input_elem.shape[0:2]
    INPUT = np.empty(shape=(0, SIZE_INPUT, SIZE_INPUT, 4))
    TARGET = np.empty(shape=(0, SIZE_TARGET, SIZE_TARGET, 4))
    # croping training patches
    '''for x in range(0, hei, STRIDE):
        for y in range(0, wid, STRIDE):
            subim_input = input_elem[x: x + SIZE_INPUT, y: y + SIZE_INPUT, :]
            subim_target = target_elem[x + PADDING: x + PADDING + SIZE_TARGET, y + PADDING: y + PADDING + SIZE_TARGET, :]
            INPUT = np.append(INPUT, subim_input[np.newaxis, ...], axis=0)
            TARGET = np.append(TARGET, subim_target[np.newaxis, ...], axis=0)
            count += 1
            # print(str(count))
            # break'''
    INPUT = input_elem[np.newaxis, ...]
    TARGET = target_elem[np.newaxis, ...]
    dset_input = h5fw.create_dataset(name='INPUT', shape=INPUT.shape, data=INPUT, dtype=np.float32)
    dset_target = h5fw.create_dataset(name='TARGET', shape=TARGET.shape, data=TARGET, dtype=np.float32)
    print(str(i) + '-INPUT' + str(INPUT.shape) + '-TARGET' + str(TARGET.shape) + '-Type' + str(dset_input.dtype))
    sys.stdout.flush()
    time.sleep(.2)
    h5fw.close()
    i += 1
    # break

# APPEND EVERY 100 IMAGES TOGETHER TO SHUFFLE################################################
SIZE_INPUT = 160
SIZE_TARGET = SIZE_INPUT

INPUT = np.empty(shape=(0, SIZE_INPUT, SIZE_INPUT, 4), dtype=np.float32)
TARGET = np.empty(shape=(0, SIZE_TARGET, SIZE_TARGET, 4), dtype=np.float32)
count = 1
i = 1
print ('>>Start shuffle images:')
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

for h5_file_name in glob.glob(DATA_800_PATH + '*.h5'):
    h5fr = h5py.File(h5_file_name, 'r')
    print(str(count) + ':' + h5_file_name)
    sys.stdout.flush()
    time.sleep(.2)
    input_ele = h5fr['INPUT'][...]
    INPUT = np.append(INPUT, h5fr['INPUT'][...], axis=0)
    TARGET = np.append(TARGET, h5fr['TARGET'][...], axis=0)

    if count % 1 == 0:
        INPUT, TARGET = unison_shuffled_copies(INPUT, TARGET)
        h5fw_b = h5py.File(str('Wv4_' + str(SIZE_INPUT) + '_b' + str(i) + '_x' + str(SCALE) + '.h5'), 'w')
        print('>>>>save b' + str(i) + str(str('Wv4_' + str(SIZE_INPUT) + '_b' + str(i) + '.h5')
                                          + '| INPUT' + str(INPUT.shape)
                                          + 'TARGET' + str(TARGET.shape)))
        dset_input = h5fw_b.create_dataset(name='INPUT', shape=INPUT.shape, data=INPUT, dtype=np.float32)
        dset_target = h5fw_b.create_dataset(name='TARGET', shape=TARGET.shape, data=TARGET, dtype=np.float32)
        h5fw_b.close()
        INPUT = np.empty(shape=(0, SIZE_INPUT, SIZE_INPUT, 4), dtype=np.float32)
        TARGET = np.empty(shape=(0, SIZE_TARGET, SIZE_TARGET, 4), dtype=np.float32)
        i += 1
    count += 1

# APPEND 8 BATCHES TOGETHER TO FORM ONE H5 FILE################################################
h5fr_1 = h5py.File('./Wv4_160_b1_x' + str(SCALE) + '.h5', 'r')
h5fr_2 = h5py.File('./Wv4_160_b2_x' + str(SCALE) + '.h5', 'r')
h5fr_3 = h5py.File('./Wv4_160_b3_x' + str(SCALE) + '.h5', 'r')
h5fr_4 = h5py.File('./Wv4_160_b4_x' + str(SCALE) + '.h5', 'r')
h5fr_5 = h5py.File('./Wv4_160_b5_x' + str(SCALE) + '.h5', 'r')
h5fr_6 = h5py.File('./Wv4_160_b6_x' + str(SCALE) + '.h5', 'r')
h5fr_7 = h5py.File('./Wv4_160_b7_x' + str(SCALE) + '.h5', 'r')
h5fr_8 = h5py.File('./Wv4_160_b8_x' + str(SCALE) + '.h5', 'r')
print ('>>Start creat final data set.')
print('b1IN%s%s OUT%s%s \nb2IN%s%s OUT%s%s \n'
      'b3IN%s%s OUT%s%s \nb4IN%s%s OUT%s%s \n'
      'b5IN%s%s OUT%s%s \nb6IN%s%s OUT%s%s \n'
      'b7IN%s%s OUT%s%s \nb8IN%s%s OUT%s%s '
      % (str(h5fr_1['INPUT'].shape), str(h5fr_1['INPUT'].dtype),
         str(h5fr_1['TARGET'].shape), str(h5fr_1['TARGET'].dtype),
         str(h5fr_2['INPUT'].shape), str(h5fr_2['INPUT'].dtype),
         str(h5fr_2['TARGET'].shape), str(h5fr_2['TARGET'].dtype),
         str(h5fr_3['INPUT'].shape), str(h5fr_3['INPUT'].dtype),
         str(h5fr_3['TARGET'].shape), str(h5fr_3['TARGET'].dtype),
         str(h5fr_4['INPUT'].shape), str(h5fr_4['INPUT'].dtype),
         str(h5fr_4['TARGET'].shape), str(h5fr_4['TARGET'].dtype),
         str(h5fr_5['INPUT'].shape), str(h5fr_5['INPUT'].dtype),
         str(h5fr_5['TARGET'].shape), str(h5fr_5['TARGET'].dtype),
         str(h5fr_6['INPUT'].shape), str(h5fr_6['INPUT'].dtype),
         str(h5fr_6['TARGET'].shape), str(h5fr_6['TARGET'].dtype),
         str(h5fr_7['INPUT'].shape), str(h5fr_7['INPUT'].dtype),
         str(h5fr_7['TARGET'].shape), str(h5fr_7['TARGET'].dtype),
         str(h5fr_8['INPUT'].shape), str(h5fr_8['INPUT'].dtype),
         str(h5fr_8['TARGET'].shape), str(h5fr_8['TARGET'].dtype),
         ))
h5fw_b = h5py.File(str('Wv4_160_ALL_x' + str(SCALE) + '.h5'), 'w')
print('>>Start concatenate INPUT')
print ('>>This may take a while...')
input_all = h5fw_b.create_dataset(name='INPUT', data=np.concatenate((h5fr_1['INPUT'][...],
                                                                     h5fr_2['INPUT'][...],
                                                                     h5fr_3['INPUT'][...],
                                                                     h5fr_4['INPUT'][...],
                                                                     h5fr_5['INPUT'][...],
                                                                     h5fr_6['INPUT'][...],
                                                                     h5fr_7['INPUT'][...],
                                                                     h5fr_8['INPUT'][...]), axis=0),
                                  dtype=np.float32)
print('INPUT ALL%s%s\nStart concatenate TARGET' % (str(input_all.shape), str(input_all.dtype)))
print('>>Start concatenate TARGET')
print ('>>This may take a while...')
target_all = h5fw_b.create_dataset(name='TARGET', data=np.concatenate((h5fr_1['TARGET'][...],
                                                                      h5fr_2['TARGET'][...],
                                                                      h5fr_3['TARGET'][...],
                                                                      h5fr_4['TARGET'][...],
                                                                      h5fr_5['TARGET'][...],
                                                                      h5fr_6['TARGET'][...],
                                                                      h5fr_7['TARGET'][...],
                                                                      h5fr_8['TARGET'][...]), axis=0),
                                   dtype=np.float32)
print ('TARGET ALL%s%s' % (str(target_all.shape), str(target_all.dtype)))
h5fw_b.close()
print ('Data preparation finished. Please use %s for training.' % str('Wv4_160_ALL_x' + str(SCALE) + '.h5'))
