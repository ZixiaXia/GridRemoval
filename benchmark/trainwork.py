import os
import tensorflow as tf
import prework
import network

#在window上运行时需加入，Linux运行时删除
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement=True
sess = tf.Session(config=config)

WEIGHT_DECAY_RATE = 1e-4
h=320
w=320
BATCH_SIZE=4
CAPACITY=80
MAX_STEP=10000
num_threads = 12
step = 0
training = True#训练时批归一化要求true，测试时要求false

train_dir='/home/pg/Public/dataGrid/'
logs_train_dir=train_dir + 'logs'
train = prework.get_batch(train_dir, h, w, BATCH_SIZE, num_threads, CAPACITY)
iterator = train.make_one_shot_iterator()
next_batch = iterator.get_next()


#MSE损失函数
fx = network.all_network(next_batch['x'], h, w, training)
train_loss = tf.reduce_mean(tf.square(tf.subtract(fx, next_batch['y'])))
'''tf.summary.scalar('loss',train_loss)'''

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)#批归一化固定操作
with tf.control_dependencies(update_ops):
    global_step = tf.get_variable('global_step', [], dtype=tf.int32,
                                      initializer=tf.constant_initializer(0), trainable=False)
    #加入退化学习率 初始值为learning_rate,让其每5000步衰减0.1，学习率 = learning_rate*0.9^(global_step/5000)
    # staircase默认值为False,当为True时，（global_step/decay_steps）则被转化为整数
    learning_rate = tf.train.exponential_decay(learning_rate=0.1, global_step=global_step, decay_steps=20000,decay_rate=0.95, staircase=True)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, name='optimizer')#梯度下降
    train_op = optimizer.minimize(train_loss, global_step=global_step)#损失优化


with tf.Session() as sess:
    # 产生一个saver来存储训练好的模型
    saver = tf.train.Saver(var_list=tf.global_variables())
    # 所有节点初始化
    sess.run(tf.global_variables_initializer())
    '''merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("log/", sess.graph)'''
    # 进行batch的训练
    while True:
        try:
            _, g_step, tra_loss= sess.run([train_op, global_step, train_loss])
            '''rs = sess.run(merged)
            writer.add_summary(rs, step)'''
            step = step + 1 
            if step % 2 == 0:
                print('Step %d, train loss = %f' % (step, tra_loss))
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
            checkpoint_path = os.path.join(logs_train_dir, 'thing.ckpt')
            #saver.save(sess, checkpoint_path, global_step=step)
            saver.save(sess, checkpoint_path)
            break





