from __future__ import print_function
import os, glob, argparse, time
import tensorflow as tf
import numpy as np
from NET import model
import h5py
import math
import grid_image
from datetime import datetime
import cv2
import pywt as pw

print ('Developed by Tiantong for NTIER CVPR 2017 SR Competition, team iPAL')
# Resume
RESUME_EXPERIMENT = False
print('WARNING: the training data in WvDATA is not all the training data\n'
      '         The training data used in this script is to show the training procedure of the network.\n'
      '         Please refer to  WvDATA/TrainDataGenerator/')

RECORD_PATH = './logs/x2.ckpt'
DATA_PATH = "./dataset/Wv4_160_ALL_x2.h5"
TEST_PATH = './TestSets/Set14BicBicX2Lum/'
print ('For track 1: bicubic downsample x2')

# RECORD_PATH = './Weight/x3.ckpt'
# DATA_PATH = "./WvDATA/Wv4_41_demo_x3.h5"
# TEST_PATH = './TestSets/Set14BicBicX3Lum/'
# print ('For track 1: bicubic downsample x3')

# RECORD_PATH = './Weight/x4.ckpt'
# DATA_PATH = "./WvDATA/Wv4_41_demo_x4.h5"
# TEST_PATH = './TestSets/Set14BicBicX4Lum/'
# print ('For track 1: bicubic downsample x4')

# Image Size
IMG_SIZE = (160, 160)
BATCH_SIZE = 4
PRINT_INFO_FLG = True

# Weight Decay
USE_WEIGHT_DECAY = True
WEIGHT_DECAY_RATE = 1e-4

# Learning Rate
USE_ADAM_OPT = True
if USE_ADAM_OPT:
    BASE_LR = 1e-3
else:
    BASE_LR = 0.1
LR_DECAY_RATE = 0.5
LR_STEP_SIZE = 40   # epoch
MAX_EPOCH = 1000
WV = 'db1'

print('>>Loading data from %s' % DATA_PATH)
DATA_NAMES = ['INPUT', 'TARGET']
h5fr = h5py.File(DATA_PATH, 'r')
DATA_SIZE = h5fr[DATA_NAMES[0]].shape
TRAINING_SIZE = DATA_SIZE[0]
TOTAL_BATCH_NUM = math.trunc(TRAINING_SIZE / BATCH_SIZE)
print('>>DATA loading finish')

# Log Info.
#GRID_Y = [16, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 16]
#GRID_X = [16, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 16]
GRID_Y = [16, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 16]
GRID_X = [16, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 16]
# DISPLAY_INTERVAL = int(1)     # for DEBUG
# SAVE_INTERVAL = int(1)        # for DEBUG
# TEST_INTERVAL = int(1)        # for DEBUG
DISPLAY_INTERVAL = int(TOTAL_BATCH_NUM)
SAVE_INTERVAL = int(TOTAL_BATCH_NUM)
TEST_INTERVAL = int(TOTAL_BATCH_NUM)


def path_gen(date_time_started):
    curr_dir = os.getcwd()
    new_dir_name = str(date_time_started)
    new_dir_name = new_dir_name.replace(':', '-')
    new_dir_name = new_dir_name.replace('.', '-')
    dir_to_make = curr_dir + '/record/' + new_dir_name
    print(dir_to_make)
    if not os.path.exists(dir_to_make):
        os.makedirs(dir_to_make)
    log_and_save_path = dir_to_make
    print(log_and_save_path)
    return log_and_save_path

LOG_PATH = path_gen(datetime.now())

if __name__ == '__main__':
    # Input & Output & Global Step
    # train_input = tf.placeholder(shape=(BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1), dtype=np.float32)
    # train_gt = tf.placeholder(shape=(BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 4), dtype=np.float32)
    # with tf.device("/gpu:1"):
    print('>>Start construct network')
    train_input = tf.placeholder(np.float32)
    train_gt = tf.placeholder(np.float32)

    global_step = tf.Variable(0, trainable=False)

    # Feeding Forward
    # shared_model = tf.make_template('shared_model', model)
    train_output, weights, all_outputs = model(train_input)
    total_loss = tf.div(tf.reduce_sum(tf.nn.l2_loss(tf.subtract(train_output, train_gt))), BATCH_SIZE)
    #total_loss = tf.div(tf.reduce_mean(tf.square(tf.subtract(train_output, train_gt))), BATCH_SIZE)
    print('     >>Weights:')

    # Print W & logging W
    i = 0
    for w in weights:
        tf.add_to_collection('summary_collection', tf.summary.histogram(w.name, w))
        if len(w.get_shape()) == 4:
            print('         LayerName:%s' % (str(w.name)))
            grid = grid_image.put_kernels_on_grid(w, GRID_Y[i], GRID_X[i])
            i += 1
            tf.add_to_collection('summary_collection',
                                 tf.summary.image(w.name + '/filters', grid, max_outputs=1))
            # Add weight decay
            if USE_WEIGHT_DECAY:
                total_loss += tf.nn.l2_loss(w) * WEIGHT_DECAY_RATE

    # Logging activations
    for a in all_outputs:
        tf.add_to_collection('summary_collection', tf.summary.histogram(a.name, a))
        temp = a[:, :, :, 0]
        temp_shape = tf.shape(temp)
        temp = tf.reshape(temp, [BATCH_SIZE, temp_shape[1], temp_shape[2], -1])
        tf.add_to_collection('summary_collection', tf.summary.image(a.name, temp, max_outputs=3))

    # Learning Rate
    learning_rate = tf.train.exponential_decay(BASE_LR,
                                               global_step,
                                               LR_STEP_SIZE * TOTAL_BATCH_NUM,
                                               LR_DECAY_RATE,
                                               staircase=True)
    # learning_rate = tf.train.exponential_decay(BASE_LR, global_step*BATCH_SIZE,
    # len(train_list)*LR_STEP_SIZE, LR_RATE, staircase=True)
    if USE_ADAM_OPT:
        optimizer = tf.train.AdamOptimizer(learning_rate)   # tf.train.MomentumOptimizer(learning_rate, 0.9)
        # opt = optimizer.minimize(loss, global_step=global_step)
    else:
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    tvars = tf.trainable_variables()
    # Gradients
    gvs = zip(tf.gradients(total_loss, tvars), tvars)
    norm = 0.01
    # Clip Gradients
    capped_gvs = [(tf.clip_by_norm(grad, norm), var) for grad, var in gvs]
    train_opt = optimizer.apply_gradients(capped_gvs, global_step=global_step)
    print('>>Construct network finished.')

    with tf.Session(config=tf.ConfigProto()) as sess:
        print('>>Start initialization:')
        tf.initialize_all_variables().run()
        print('>>Initialization finished.')
        # add summary here:
        tf.add_to_collection('summary_collection', tf.summary.histogram(train_input.name, train_input))
        tf.add_to_collection('summary_collection', tf.summary.image('input_images', train_input, max_outputs=3))
        tf.add_to_collection('summary_collection', tf.summary.histogram(train_gt.name, train_gt))
        tf.add_to_collection('summary_collection', tf.summary.image('target_images', train_gt, max_outputs=3))
        tf.add_to_collection('summary_collection', tf.summary.histogram(train_output.name, train_output))
        tf.add_to_collection('summary_collection', tf.summary.image('output_images', train_output, max_outputs=3))
        tf.add_to_collection('summary_collection', tf.summary.scalar('learning_rate', learning_rate))
        tf.add_to_collection('summary_collection', tf.summary.scalar('loss', total_loss))
        tf.add_to_collection('summary_collection', tf.summary.scalar('global_step', global_step))
        tf.add_to_collection('summary_collection', tf.summary.scalar('learning_rate', learning_rate))
        summary_op = tf.summary.merge(tf.get_collection('summary_collection'))
        summary_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

        saver = tf.train.Saver(tf.all_variables())
        if RESUME_EXPERIMENT:
            print ('>>>>>>>>Resuming Experiments:')
            saver.restore(sess, RECORD_PATH)
            print('Start learning rate:%s and steps@:%d' % (str(learning_rate.eval()), global_step.eval()))
        else:
            print ('>>>>>>>>New Experiments:')

        if PRINT_INFO_FLG:
            print('DataPath%s\t|DataNames%s\t|TrainingSize:%s\n'
                  'ImageSize:%s\t|BatchSize:%s\t|TotalBatchNum:%s\n'
                  'IfUsingAdam:%s\t|BaseLearningRate:%s\t|LearningDecayRate&Step:%s&%s\n'
                  'IfUsingWeightDecay:%s\t|WeightDacayRate:%s\n'
                  % (DATA_PATH, str(DATA_NAMES), str(TRAINING_SIZE),
                     str(IMG_SIZE), str(BATCH_SIZE), str(TOTAL_BATCH_NUM),
                     str(USE_ADAM_OPT), str(BASE_LR), str(LR_DECAY_RATE), str(LR_STEP_SIZE),
                     str(USE_WEIGHT_DECAY), str(WEIGHT_DECAY_RATE)))

        # Train
        print('>>Start training:')
        step = 1
        start_time = time.time()
        for epoch in range(0, MAX_EPOCH):
            for batch_index in range(0, TOTAL_BATCH_NUM):
                start_index = batch_index * BATCH_SIZE
                end_index = (batch_index + 1) * BATCH_SIZE
                input_data = h5fr[DATA_NAMES[0]][start_index:end_index, :, :, :]
                gt_data = h5fr[DATA_NAMES[1]][start_index:end_index, :, :, :]
                feed_dict = {train_input: input_data, train_gt: gt_data}
                _, loss_now, output, lr, g_step = sess.run([train_opt, total_loss,
                                                            train_output, learning_rate, global_step],
                                                           feed_dict=feed_dict)
                step += 1
                if step % int(DISPLAY_INTERVAL) == 0:
                    duration = time.time() - start_time
                    second_per_batch = float(duration / DISPLAY_INTERVAL)
                    sample_per_second = (BATCH_SIZE * DISPLAY_INTERVAL) / duration
                    print("\n[epoch %2.4f step %d] loss %.4f\t lr %.10f\t sample/s %.4f\t s/batch %.4f" %
                          (epoch, step, loss_now, lr, sample_per_second, second_per_batch))
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
                    start_time = time.time()
                if step % int(SAVE_INTERVAL) == 0:
                    if not os.path.exists(str(str(LOG_PATH) + '/model/')):
                        os.makedirs(str(str(LOG_PATH) + '/model/'))
                    save_path = saver.save(sess, str(str(LOG_PATH) + '/model/tf' + str(g_step) + '.ckpt'))
                    print("Model saved in: %s" % save_path)
                '''if step % int(TEST_INTERVAL) == 0:
                    print('>>Start testing:')
                    text_index = 1
                    if not os.path.exists(str(str(LOG_PATH) + '/test/')):
                        os.makedirs(str(str(LOG_PATH) + '/test/'))
                    for testImgName in glob.glob(TEST_PATH + '*.bmp'):
                        print('     Test Image %s' % testImgName)
                        testBBImg = cv2.imread(testImgName, 0)
                        tcoeffs = pw.dwt2(testBBImg, WV)
                        tcA, (tcH, tcV, tcD) = tcoeffs
                        tcA = tcA.astype(np.float32) / 255
                        tcH = tcH.astype(np.float32) / 255
                        tcV = tcV.astype(np.float32) / 255
                        tcD = tcD.astype(np.float32) / 255
                        test_temp = np.array([tcA, tcH, tcV, tcD])
                        test_elem = np.rollaxis(test_temp, 0, 3)
                        test_data = test_elem[np.newaxis, ...]
                        output_data = sess.run([train_output], feed_dict={train_input: test_data})
                        dcA = output_data[0][0, :, :, 0]
                        dcH = output_data[0][0, :, :, 1]
                        dcV = output_data[0][0, :, :, 2]
                        dcD = output_data[0][0, :, :, 3]
                        srcoeffs = (dcA * 255 + tcA * 255,
                                    (dcH * 255 + tcH * 255,
                                     dcV * 255 + tcV * 255,
                                     dcD * 255 + tcD * 255))
                        sr_img = pw.idwt2(srcoeffs, WV)
                        # cv2.namedWindow('SR', cv2.WINDOW_NORMAL)
                        # cv2.imshow("SR", sr_img)                            # Show image
                        # cv2.waitKey(0)
                        cv2.imwrite(str(str(LOG_PATH) + '/test/' + str(text_index) +
                                        'Reconst_'+str(step) + '.bmp'),
                                    sr_img)
                        text_index += 1
                        # print('Test output %s' % str(output_data[0].shape))
                    # save_path = saver.save(sess, str(str(LOG_PATH) + '/test/tf' + str(step)))'''
    sess.close()
