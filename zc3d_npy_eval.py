import os.path as osp
import os
import tensorflow as tf
import numpy as np
import time
import json

import data_io.basepy as basepy
import zc3d_npy_base as base
import zdefault_dict

TEST_LIST = ('/absolute/datasets/Anomaly-Detection-Dataset/Temporal_Anomaly_Annotation_for_Testing_Videos.txt',
             '/home/zbh/Desktop/absolute/datasets/UCFCrime2Local/Test_split_AD.txt',
             '/absolute/datasets/UCSDped2_split_list/10_fold_001/v01_test.txt')[0]
# Basic model parameters as external flags.
tags = tf.flags
F = tags.FLAGS
tags.DEFINE_string('save_file_path', '/absolute/tensorflow_models/190510191850',
                   'where to restore.')
tags.DEFINE_string('set_gpu', '0', 'Single gpu version, index select')
tags.DEFINE_integer('batch_size', 1, 'batch size.')
tags.DEFINE_string('testing_list', TEST_LIST, 'test samples from the list')

MODEL_FOLDER = F.save_file_path if osp.isdir(F.save_file_path) else osp.dirname(F.save_file_path)
JSON_FILE_PATH = basepy.get_1tier_file_path_list(MODEL_FOLDER, suffix='.json')[0]

D = basepy.DictCtrl(zdefault_dict.EXPERIMENT_KEYS).read4path(JSON_FILE_PATH)
D['batch_size'], D['set_gpu'] = 1, F.set_gpu

print('------ D values: ------')
_ = [print(i, ":", D[i]) for i in D]


def main(_):
    feature_path_list = basepy.get_1tier_file_path_list(D['npy_file_path'], suffix='.npy')

    # test_txt = '/absolute/datasets/Anomaly-Detection-Dataset/Temporal_Anomaly_Annotation_for_Testing_Videos.txt'
    test_list = basepy.read_txt_lines2list(F.testing_list, sep='  ')
    test_list = base.reform_train_list(test_list, feature_path_list)
    feature_dict = base.read_npy_file_path_list(test_list)

    test_keys = list(feature_dict.keys())
    label_keys = [0 if 'normal' in j.lower() else 1 for j in test_keys]

    if not os.path.isdir(F.save_file_path):
        _ = eval_one_ckpt(test_keys, label_keys, feature_dict, D, ckpt_file=F.save_file_path, sample_num=2500)
    else:
        # model_checkpoint_path: "/absolute/tensorflow_models/190516014752/190516014752.ckpt-16001"
        ckpt_check_list = [i[1][1:-1]
                           for i in basepy.read_txt_lines2list(osp.join(MODEL_FOLDER, 'checkpoint'), sep=': ')][1:]

        results = [eval_one_ckpt(
            test_keys, label_keys, feature_dict, D, ckpt_file=one_ckpt, if_draw=False, sample_num=2500)
            for one_ckpt in ckpt_check_list]

        _ = [print(result[3]) for result in results]

    print('------ Finish ------ Debug Symbol ------ %s ------' % time.asctime(time.localtime(time.time())))


def eval_one_ckpt(test_keys, label_keys, feature_dict, d, ckpt_file=None,
                  if_print=True, if_draw=True, sample_num=1000, npy_folder_suffix='_anoma_json'):
    with tf.name_scope('input'):
        input_test = tf.placeholder(tf.float32, [d['batch_size'], d['segment_num'], d['feature_len']], name='anom')

    with tf.name_scope('forward-propagation'):
        score_anomaly = base.network_fn(input_test,
                                        fusion=d['fusion'],
                                        feature_len=d['feature_len'],
                                        segment_num=d['segment_num'],
                                        is_training=False)
    saver = tf.train.Saver()

    init_op = tf.global_variables_initializer()
    os.environ["CUDA_VISIBLE_DEVICES"] = d['set_gpu']
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        saver.restore(sess, ckpt_file)

        print('Program begins, timestamp %s' % time.asctime(time.localtime(time.time())))

        step, label_test = 0, []
        try:
            while True:
                test_in, info_in, = [], []
                for i in test_keys[step * d['batch_size']:step * d['batch_size'] + d['batch_size']]:
                    test_in.append(base.reform_np_array(feature_dict[i], reform=d['segment_num']))
                    info_in.append(feature_dict[i][:, 4096:])

                np_test_in = np.array(test_in, dtype='float32')

                s = sess.run(score_anomaly, feed_dict={input_test: np_test_in})
                # in form of LIST:
                label_test.append(np.max(s))
                # make probability .npy
                if npy_folder_suffix:
                    eval_npy_path = basepy.check_or_create_path(ckpt_file + npy_folder_suffix)
                    ins_results = np.hstack((info_in[0], s[:, :len(info_in[0])].T))
                    list_results = [i.split('@') + clip for clip in ins_results.tolist()]
                    with open(osp.join(eval_npy_path, i + '.json'), 'w') as f:
                        json.dump(list_results, f)

                step += 1
        except ValueError:
            print("Test done, at %s" % time.asctime(time.localtime(time.time())))

    # t, f, auc, auc_, precision_list = basepy.TPR_FPR(label_test, label_keys, bool_draw=if_draw, sample_num=sample_num)
    # if if_print:
    #     print('%s evaluate results:' % ckpt_file)
    #     print('auc: %5f, auc_: %5f, max precision: %5f,' % (auc, auc_, max(precision_list)))
    #     print('TPR_points:')
    #     print(t)
    #     print('FPR_points:')
    #     print(f)
    #     print('corresponding precision:')
    #     print(precision_list)
    #
    # return t, f, auc, auc_, precision_list


if __name__ == '__main__':
    tf.app.run()
