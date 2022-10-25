import tensorflow as tf

#在window上运行时需加入，Linux运行时删除
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def conv2d(x, filter_shape, name, s, p):
    #生成标准权重filters
    filters = tf.get_variable(
        shape=filter_shape,
        name=name,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),#返回一个初始化权重
        trainable=True)
    return tf.nn.conv2d(x, filters, strides=[1, s, s, 1], padding=p)

def line_extraction(x_batch, training):
    b_conv1 = tf.layers.batch_normalization(conv2d(x_batch, [11, 11, 1, 32], 'w_conve1', 1, 'SAME'), training = training, name='b_conve1')
    h_conv1 = tf.nn.relu(b_conv1)

    b_conv2 = tf.layers.batch_normalization(conv2d(h_conv1, [3, 3, 32, 32], 'w_conve2', 1, 'SAME'), training = training, name='b_conve2')
    h_conv2 = tf.nn.relu(b_conv2)

    b_conv3 = tf.layers.batch_normalization(conv2d(h_conv2, [3, 3, 32, 32], 'w_conve3', 1, 'SAME'), training = training, name='b_conve3')
    h_conv3 = tf.nn.relu(b_conv3)

    b_conv4 = tf.layers.batch_normalization(conv2d(h_conv3, [3, 3, 32, 32], 'w_conve4', 1, 'SAME'), training = training, name='b_conve4')
    h_conv4 = tf.nn.relu(b_conv4)

    b_conv5 = tf.layers.batch_normalization(conv2d(h_conv4, [3, 3, 32, 32], 'w_conve5', 1, 'SAME'), training = training, name='b_conve5')
    h_conv5 = tf.nn.relu(b_conv5)

    b_conv6 = tf.layers.batch_normalization(conv2d(h_conv5, [3, 3, 32, 1], 'w_conve6', 1, 'SAME'), training = training, name='b_conve6')
    h_conv6 = tf.nn.sigmoid(b_conv6)

    return h_conv6

def cnn(x_batch, h, w, training):
    #批归一化函数tf.layers.batch_normalization
    b_conv1 = tf.layers.batch_normalization(conv2d(x_batch, [5, 5, 2, 32], 'w_conv1', 2, 'VALID'), training = training, name='b_conv1')
    h_conv1 = tf.nn.relu(b_conv1)
    size1=int((h-5)/2+1)

    b_conv2 = tf.layers.batch_normalization(conv2d(h_conv1, [3, 3, 32, 64], 'w_conv2', 2, 'VALID'), training = training,name = 'b_conv2')
    h_conv2 = tf.nn.relu(b_conv2)
    size2=int((size1-3)/2+1)

    b_conv3 = tf.layers.batch_normalization(conv2d(h_conv2, [3, 3, 64, 128], 'w_conv3', 1, 'SAME'), training = training, name = 'b_conv3')
    h_conv3 = tf.nn.relu(b_conv3)
    size3=size2

    b_conv4 = tf.layers.batch_normalization(conv2d(h_conv3, [3, 3, 128, 256], 'w_conv4', 2, 'VALID'), training = training, name = 'b_conv4')
    h_conv4 = tf.nn.relu(b_conv4)
    size4=int((size3-3)/2+1)

    b_conv5 = tf.layers.batch_normalization(conv2d(h_conv4, [3, 3, 256, 512], 'w_conv5', 1, 'SAME'), training = training, name = 'b_conv5')
    h_conv5 = tf.nn.relu(b_conv5)
    size5=size4

    b_conv6 = tf.layers.batch_normalization(conv2d(h_conv5, [3, 3, 512, 512], 'w_conv6', 1, 'SAME'), training = training, name = 'b_conv6')
    h_conv6 = tf.nn.relu(b_conv6)
    size6=size5

    b_conv7 = tf.layers.batch_normalization(conv2d(h_conv6, [3, 3, 512, 256], 'w_conv7', 1, 'SAME'), training = training, name = 'b_conv7')
    h_conv7 = tf.nn.relu(b_conv7)
    size7=size6

    b_conv8 = tf.layers.batch_normalization(conv2d(h_conv7, [3, 3, 256, 128], 'w_conv8', 1, 'SAME'), training = training, name = 'b_conv8')
    h_conv8 = tf.nn.relu(b_conv8)
    size8 = size7

    h_conv9_up = tf.image.resize_images(h_conv8, size=[size2, size2], method = 1)
    size9_up=size2

    b_conv9 = tf.layers.batch_normalization(conv2d(h_conv9_up, [3, 3, 128, 128], 'w_conv9', 1, 'SAME'), training = training, name = 'b_conv9')
    h_conv9 = tf.nn.relu(b_conv9)
    size9=size9_up

    b_conv10 = tf.layers.batch_normalization(conv2d(h_conv9, [3, 3, 128, 64], 'w_conv10', 1, 'SAME'), training = training, name = 'b_conv10')
    h_conv10 = tf.nn.relu(b_conv10)
    size10=size9

    h_conv11_up = tf.image.resize_images(h_conv10, size=[size1, size1], method = 1)

    b_conv11 = tf.layers.batch_normalization(conv2d(h_conv11_up, [3, 3, 64, 64], 'w_conv11', 1, 'SAME'), training = training, name = 'b_conv11')
    h_conv11 = tf.nn.relu(b_conv11)

    b_conv12 = tf.layers.batch_normalization(conv2d(h_conv11, [3, 3, 64, 32], 'w_conv12', 1, 'SAME'), training = training, name = 'b_conv12')
    h_conv12 = tf.nn.relu(b_conv12)

    h_conv13_up = tf.image.resize_images(h_conv12, size=[h, w], method = 1)

    b_conv13 = tf.layers.batch_normalization(conv2d(h_conv13_up, [3, 3, 32, 16], 'w_conv13', 1, 'SAME'), training = training, name = 'b_conv13')
    h_conv13 = tf.nn.relu(b_conv13)

    b_conv14 = tf.layers.batch_normalization(conv2d(h_conv13, [3, 3, 16, 8], 'w_conv14', 1, 'SAME'), training = training, name = 'b_conv14')
    h_conv14 = tf.nn.relu(b_conv14)

    b_conv15 = tf.layers.batch_normalization(conv2d(h_conv14, [3, 3, 8, 1], 'w_conv15', 1, 'SAME'), training = training, name = 'b_conv15')
    h_conv15 = tf.nn.tanh(b_conv15)

    return h_conv15

def all_network(x_batch, h, w, training):
    ex = line_extraction(x_batch, training)
    xex = tf.concat([x_batch, ex], 3)#两个四维矩阵矩阵在第三维叠加
    rx = cnn(xex, h, w, training)
    output = tf.add(rx, ex)
    output = ex
    a = tf.to_float(0)
    b = tf.to_float(1)
    fx=tf.maximum(a, tf.minimum(b, output))
    return fx
