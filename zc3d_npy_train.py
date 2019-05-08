import os.path as osp
import os
import tensorflow as tf
import numpy as np
import time
from pprint import pprint

import data_io.basepy as basepy
import zc3d_npy_base as base
import zdefault_dict

# Basic model parameters as external flags.
timestamp = time.strftime("%y%m%d%H%M%S", time.localtime())
tags = tf.flags
# Net config
tags.DEFINE_integer('batch_size', 30, 'batch size.')
tags.DEFINE_integer('epoch_num', 600, 'epoch number.')
tags.DEFINE_float('learning_rate_base', 0.0003, 'learning rate base')
tags.DEFINE_float('moving_average_decay', 0.99, 'moving average decay')
tags.DEFINE_float('regularization_scale', 0.00003, 'regularization scale')
tags.DEFINE_string('npy_file_path', '/absolute/datasets/anoma_motion16_c3d_features', 'npy file path')
tags.DEFINE_string('fusion', 'standard', 'fusion ways in feature extraction')
# General
tags.DEFINE_string('set_gpu', '0', 'Single gpu version, index select')
tags.DEFINE_string('save_file_path', osp.join('/absolute/tensorflow_models', timestamp, timestamp + '.ckpt'),
                   'where to store tensorflow models')
F = tags.FLAGS

SAVE_FILE_PATH = F.save_file_path
JSON_FILE_PATH = osp.join(basepy.check_or_create_path(osp.dirname(SAVE_FILE_PATH), show=True), 'anoma_v08_keys.json')
D = basepy.DictCtrl(zdefault_dict.EXPERIMENT_KEYS).save2path(JSON_FILE_PATH,
                                                             batch_size=F.batch_size,
                                                             epoch_num=F.epoch_num,
                                                             learning_rate_base=F.learning_rate_base,
                                                             moving_average_decay=F.moving_average_decay,
                                                             regularization_scale=F.regularization_scale,
                                                             npy_file_path=F.npy_file_path,
                                                             set_gpu=F.set_gpu,
                                                             lambda1=0.00008,
                                                             lambda2=0.00008,
                                                             fusion=F.fusion,
                                                             )

print('D values:')
_ = [print(i, ":", D[i]) for i in D]


def main(_):
    # with tf.device('/cpu:0'):
    feature_path_list = basepy.get_1tier_file_path_list(D['npy_file_path'], suffix='.txt')
    train_txt = '/absolute/datasets/Anomaly-Detection-Dataset/Anomaly_Train.txt'
    train_list = basepy.read_txt_lines2list(train_txt, sep=' ')
    train_list = base.reform_train_list(train_list, feature_path_list)

    anomaly_keys = [i for i in train_list if 'normal' not in i.lower()]
    normal_keys = [i for i in train_list if 'normal' in i.lower()]

    anomaly_list = basepy.repeat_list_for_epochs(anomaly_keys, epoch_num=D['epoch_num'], shuffle=True)
    normal_list = basepy.repeat_list_for_epochs(normal_keys, epoch_num=D['epoch_num'], shuffle=True)

    with tf.name_scope('input'):
        input_anom = tf.placeholder(tf.float32, [D['batch_size'], D['segment_num'], D['feature_len']], name='anom')
        input_norm = tf.placeholder(tf.float32, [D['batch_size'], D['segment_num'], D['feature_len']], name='norm')

    with tf.name_scope('forward-propagation'):
        score_anomaly = base.network_fn(input_anom,
                                        fusion=D['fusion'],
                                        feature_len=D['feature_len'],
                                        segment_num=D['segment_num'])
        score_normal = base.network_fn(input_norm,
                                       fusion=D['fusion'],
                                       feature_len=D['feature_len'],
                                       segment_num=D['segment_num'])

    with tf.name_scope('loss'):
        # batch_score = tf.convert_to_tensor([tf.reduce_max(i) for i in score_anomaly])
        # restrict1 = tf.convert_to_tensor(
        #     [tf.reduce_mean((score_anomaly[i][1:] - score_anomaly[i][:-1]) ** 2) for i in range(D['batch_size'])],
        #     dtype=tf.float32)
        restrict2 = tf.convert_to_tensor(
            [tf.reduce_mean(score_anomaly[i] ** 2) for i in range(D['batch_size'])],
            dtype=tf.float32)
        mil_loss = tf.maximum(0., 1. - tf.reduce_max(score_anomaly, axis=1) + tf.reduce_max(score_normal, axis=1))
        regu = tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l2_regularizer(D['regularization_scale']), tf.trainable_variables())
        mean_mil = tf.reduce_mean(mil_loss)
        # l1, l2 = D['lambda1'] * tf.reduce_mean(restrict1), D['lambda2'] * tf.reduce_mean(restrict2)
        l2 = D['lambda2'] * tf.reduce_mean(restrict2)
        if D['fusion'] == 'standard':
            loss = mean_mil + l2 + regu
        elif D['fusion'] == 'segments' or D['fusion'] == 'average' or D['fusion'] == 'attention':
            loss = mean_mil + regu
        else:
            raise ValueError('Wrong fusion type: %s' % D['fusion'])

    global_step = tf.Variable(0, trainable=False)
    with tf.name_scope('moving_average'):
        variable_averages = tf.train.ExponentialMovingAverage(D['moving_average_decay'], global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.name_scope('train_step'):
        train_step = tf.train.AdamOptimizer(learning_rate=D['learning_rate_base']
                                            ).minimize(loss, global_step=global_step)
        with tf.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.no_op(name='train')

    print(('* ' + 'Variables to be trained' + ' *').center(60, '*'))
    pprint(tf.trainable_variables())
    print('Model .ckpt save path: %s' % SAVE_FILE_PATH)

    saver = tf.train.Saver(max_to_keep=15)

    init_op = tf.global_variables_initializer()
    os.environ["CUDA_VISIBLE_DEVICES"] = D['set_gpu']
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        sess.run(init_op)

        print('program begins, timestamp %s' % time.asctime(time.localtime(time.time())))

        try:
            while True:
                step = sess.run(global_step)
                anomaly_in, normal_in = [], []
                for i in anomaly_list[step * D['batch_size']:step * D['batch_size'] + D['batch_size']]:
                    anomaly_in.append(feature_dict[i])
                for i in normal_list[step * D['batch_size']:step * D['batch_size'] + D['batch_size']]:
                    normal_in.append(feature_dict[i])

                l, _, a, c, d = sess.run([loss, train_op, mean_mil, l2, regu],
                                            feed_dict={input_anom: np.array(anomaly_in, dtype='float32'),
                                                       input_norm: np.array(normal_in, dtype='float32')})

                if step % 100 == 0 or (step % 10 == 0 and step < 100):
                    print('After %d steps, loss = %g, mil = %g, l2 = %g, regu = %g.' % (step, l, a, c, d))
                if step % 1000 == 0:
                    print('Save tfrecords at step %s' % step)
                    saver.save(sess, SAVE_FILE_PATH, global_step=global_step)
        except ValueError:
            print("Training done, at %s" % time.asctime(time.localtime(time.time())))
    print('debug symbol')


if __name__ == '__main__':
    tf.app.run()
