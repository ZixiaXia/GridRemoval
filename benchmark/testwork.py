# coding=UTF-8
import os
import numpy as np
import network
import tensorflow as tf
import cv2 as cv

#在window上运行时需加入，Linux运行时删除
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

h=320#图片高
w=320#图片宽
training = False#训练时批归一化要求true，测试时要求false

def normalization(input):
    max = tf.reduce_max(input)
    min = tf.reduce_min(input)
    return (input-min)/(max-min)

#获取图片路径
def get_dir(file_dir):
    testdir = []
    for file in os.listdir(file_dir + 'test'):
        testdir.append(file_dir + 'test/' + file)
    return testdir

#生成形如tensor的四维图片矩阵
def get_img_array(file_dir):
    img = cv.imread(file_dir[0], 0)
    img = cv.resize(img, (h, w))
    img = np.reshape(img, [1, h, w, 1])
    for i in range(1,len(file_dir)):
        imgt = cv.imread(file_dir[i], 0)
        imgt = cv.resize(imgt, (h, w))
        imgt = np.reshape(imgt, [1, h, w, 1])
        img = tf.concat([img, imgt], 0)#矩阵在第0维叠加
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        img_aray = img.eval()#tensor转numpy
    return img_aray

# 测试图片
def evaluate_one_image(image_array, logs_train_dir):
    image = tf.image.convert_image_dtype(image_array, tf.float32)
    image = tf.reshape(image, [-1, h, w, 1])
    image = tf.cast(image, tf.complex64)
    image = tf.fft2d(image)
    image = tf.concat([normalization(tf.real(image)), normalization(tf.imag(image))], 3)
    logit = network.all_network(image, h, w, training)
    #logit = tf.image.convert_image_dtype(logit, tf.uint8)

    x = tf.placeholder(tf.float32, shape=[None, h, w, 2])#占位符
    saver = tf.train.Saver(tf.global_variables())#读取模型文件

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)

            #feed the data
            prediction = sess.run(logit, feed_dict={x: image.eval()})
            #拆分四维矩阵为n个三维矩阵（n个图片）
            for i in range(prediction.shape[0]):
                real = prediction[i:i+1, 0:h, 0:w, 0]
                imag = prediction[i:i+1, 0:h, 0:w, 1]
                real = np.reshape(real, [h, w, 1])
                imag = np.reshape(imag, [h, w, 1])
                max1 = 320
                min1 = -30
                max2 = 90
                min2 = -90
                output = tf.ifft2d(tf.complex(real*(max1-min1)+min1,imag*(max2-min2)+min2))
                print(output.shape)
                output = abs(output.eval())
                output = output.squeeze()
                cv.imwrite( 'test/' + str(i) + '.jpg', output*255)

        else:
            print('No checkpoint file found')


#main（）
if __name__ == '__main__':
    test_dir = '/home/pg/Public/dataGrid/'
    logs_train_dir = '/home/pg/Public/dataGrid/logs'
    testdir = get_dir(test_dir)
    img_aray = get_img_array(testdir)
    evaluate_one_image(img_aray, logs_train_dir)


