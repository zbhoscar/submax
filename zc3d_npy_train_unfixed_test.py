import os.path as osp
import os
import tensorflow as tf
import numpy as np
import time
from pprint import pprint

import data_io.basepy as basepy
import zc3d_npy_base as base
import zfeatures_cliptxt2segmentnpy as io

# Basic model parameters as external flags.
tags = tf.app.flags
tags.DEFINE_integer('batch_size', 10, 'Batch size.')
tags.DEFINE_float('learning_rate_base', 0.01, 'learning rate base')
tags.DEFINE_float('moving_average_decay', 0.99, 'moving average decay')
tags.DEFINE_string('npy_file_path', '/absolute/datasets/anoma_1632_c3d_clips_features', 'npy file path')
F = tags.FLAGS

LAMBADA1 = 0.0005
LAMBADA2 = 0.0005
BATCH_SIZE = 20
C3D_FEATURE_LENGTH = 4096


def main(_):
    with tf.device('/cpu:0'):
        feature_dict = io.read_npy_file_path_list(basepy.get_1tier_file_path_list(F.npy_file_path, suffix='.npy'))
        train_txt = '/absolute/datasets/Anomaly-Detection-Dataset/Anomaly_Train.txt'
        train_list = basepy.read_txt_lines2list(train_txt, sep=' ')
        anomaly_keys, normal_keys = [], []
        for i in train_list:
            class_name = osp.dirname(i[0])
            if class_name != 'Training_Normal_Videos_Anomaly':
                video_name = osp.splitext(osp.basename(i[0]))[0]
                anomaly_keys.append(class_name + '@' + video_name)
            else:
                class_name = 'normal_train'
                video_name = osp.splitext(osp.basename(i[0]))[0]
                normal_keys.append(class_name + '@' + video_name)
        anomaly_list = basepy.repeat_list_for_epochs(anomaly_keys, epoch_num=F.batch_size, shuffle=True)
        normal_list = basepy.repeat_list_for_epochs(normal_keys, epoch_num=F.batch_size, shuffle=True)

    with tf.name_scope('input'):
        input_anomaly = [tf.placeholder(tf.float32, [None, C3D_FEATURE_LENGTH], name='an') for _ in range(BATCH_SIZE)]
        input_normal = [tf.placeholder(tf.float32, [None, C3D_FEATURE_LENGTH], name='no') for _ in range(BATCH_SIZE)]

    with tf.name_scope('forward-propagation'):
        score_anomaly = base.network_fn(input_anomaly)
        score_normal = base.network_fn(input_normal)

    with tf.name_scope('loss'):
        # batch_score = tf.convert_to_tensor([tf.reduce_max(i) for i in score_anomaly])
        restrict1 = tf.convert_to_tensor([tf.reduce_mean((i[1:] - i[:-1]) ** 2) for i in score_anomaly],
                                         dtype=tf.float32)
        restrict2 = tf.convert_to_tensor([tf.reduce_mean(i ** 2) for i in score_anomaly], dtype=tf.float32)
        mil_loss = tf.convert_to_tensor(
            [tf.reduce_max([0., 1. - tf.reduce_max(score_anomaly[i]) + tf.reduce_max(score_normal[i])])
             for i in range(BATCH_SIZE)], dtype=tf.float32)
        loss = tf.reduce_mean(mil_loss) + LAMBADA1 * tf.reduce_mean(restrict1) + LAMBADA2 * tf.reduce_mean(restrict2)

    global_step = tf.Variable(0, trainable=False)
    with tf.name_scope('moving_average'):
        variable_averages = tf.train.ExponentialMovingAverage(F.moving_average_decay, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.name_scope('train_step'):
        train_step = tf.train.AdamOptimizer(learning_rate=F.learning_rate_base).minimize(loss, global_step=global_step)
        with tf.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.no_op(name='train')

    print(('* ' + 'Variables to be trained' + ' *').center(60, '*'))
    pprint(tf.trainable_variables())

    init_op = tf.global_variables_initializer()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        sess.run(init_op)

        print('program begins, timestamp %s' % time.asctime(time.localtime(time.time())))

        go_on_training = True
        while go_on_training:
            step = sess.run(global_step)
            anomaly_in, normal_in = [], []
            for i in anomaly_list[step * F.batch_size:step * F.batch_size + F.batch_size]:
                anomaly_in.append(feature_dict[i])
            for i in normal_list[step * F.batch_size:step * F.batch_size + F.batch_size]:
                normal_in.append(feature_dict[i])

            l, _ = sess.run([loss, train_op], feed_dict={input_anomaly: anomaly_in, input_normal: normal_in})

            if step % 100 == 0 or (step % 10 == 0 and step < 100):
                print('After %d steps,loss = %g' % (step, l))

            go_on_training = True if step * F.batch_size + F.batch_size >= anomaly_list.__len__() else False

    print("done, at %s" % time.asctime(time.localtime(time.time())))
    print('debug symbol')


if __name__ == '__main__':
    tf.app.run()
