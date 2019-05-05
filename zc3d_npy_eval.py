import os.path as osp
import os
import tensorflow as tf
import numpy as np
import time

import data_io.basepy as basepy
import data_io.basetf as basetf
import zc3d_npy_base as base
import zfeatures_cliptxt2segmentnpy as io
import zdefault_dict

# Basic model parameters as external flags.
tags = tf.flags
F = tags.FLAGS
tags.DEFINE_string('save_file_path', '/absolute/tensorflow_models/190425190541/190425190541.ckpt-1001',
                   'where to restore.')
tags.DEFINE_string('set_gpu', '0', 'Single gpu version, index select')
tags.DEFINE_integer('batch_size', 1, 'batch size.')

RESTORE_FILE_PATH = basetf.get_ckpt_path(F.save_file_path)
print('Restoring .ckpt from %s' % RESTORE_FILE_PATH)
JSON_FILE_PATH = osp.join(osp.dirname(RESTORE_FILE_PATH), 'keys.json')

D = basepy.DictCtrl(zdefault_dict.EXPERIMENT_KEYS).read4path(JSON_FILE_PATH)
D['batch_size'], D['set_gpu'] = 1, '0'

print('D values:')
_ = [print(i, ":", D[i]) for i in D]


def main(_):
    with tf.device('/cpu:0'):
        feature_dict = io.read_npy_file_path_list(
            basepy.get_1tier_file_path_list(D['npy_file_path'],
                                            suffix=D['embedding'] + '.npy'), class_name_in_keys=False)

        test_txt = '/absolute/datasets/Anomaly-Detection-Dataset/Temporal_Anomaly_Annotation_for_Testing_Videos.txt'
        test_list = basepy.read_txt_lines2list(test_txt, sep='  ')
        test_keys = [j[0].split('.')[0] + D['embedding'] for j in test_list]
        label_keys = [0 if 'Normal' in j[0] else 1 for j in test_list]

    with tf.name_scope('input'):
        input_test = tf.placeholder(tf.float32, [D['batch_size'], D['segment_num'], D['feature_len']], name='anom')

    with tf.name_scope('forward-propagation'):
        score_anomaly = base.network_fn(input_test,
                                        fusion=D['fusion'],
                                        feature_len=D['feature_len'],
                                        segment_num=D['segment_num'],
                                        attention_l=D['attention_l'],
                                        is_training=False)

    saver = tf.train.Saver()

    init_op = tf.global_variables_initializer()
    os.environ["CUDA_VISIBLE_DEVICES"] = D['set_gpu']
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        saver.restore(sess, RESTORE_FILE_PATH)

        print('program begins, timestamp %s' % time.asctime(time.localtime(time.time())))

        step, label_test = 0, []
        try:
            while True:
                test_in = []
                for i in test_keys[step * D['batch_size']:step * D['batch_size'] + D['batch_size']]:
                    test_in.append(feature_dict[i])

                np_test_in = np.array(test_in, dtype='float32')

                s = sess.run(score_anomaly, feed_dict={input_test: np_test_in})
                # in form of LIST:
                label_test.append(np.max(s))

                step += 1
        except ValueError:
            print("Test done, at %s" % time.asctime(time.localtime(time.time())))

    t, f, auc, auc_, precision_list = basepy.TPR_FPR(label_test, label_keys, bool_draw=True, sample_num=100)
    print(auc, auc_)
    print(t)
    print(f)
    print(precision_list)
    print(max(precision_list))

    print('debug symbol')


if __name__ == '__main__':
    tf.app.run()
