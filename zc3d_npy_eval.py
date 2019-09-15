import os.path as osp
import os
import tensorflow as tf
import numpy as np
import time
import json
from itertools import chain

import data_io.basepy as basepy
import zc3d_npy_base as base
import zdefault_dict

# TEST_LIST = ('/absolute/datasets/Anomaly-Detection-Dataset/Temporal_Anomaly_Annotation_for_Testing_Videos.txt',
#              '/home/zbh/Desktop/absolute/datasets/UCFCrime2Local/Test_split_AD.txt',
#              '/absolute/datasets/UCSDped2_split_list/10_fold_001/v01_test.txt')[0]
# Basic model parameters as external flags.
tags = tf.flags
F = tags.FLAGS
tags.DEFINE_string('save_file_path',
                   '/absolute/tensorflow_models/190912162832_anoma_motion_4training_pyramid_80_56_4region_c3d_npy/190912162832.ckpt-1',
                   'model folder path, or model ckpt file path.')
tags.DEFINE_string('set_gpu', '0', 'Single gpu version, index select')
# tags.DEFINE_integer('batch_size', 1, 'batch size.')
tags.DEFINE_string('testing_list',
                   '/absolute/datasets/Anomaly-Detection-Dataset/Temporal_Anomaly_Annotation_for_Testing_Videos.txt',
                   'test samples from the list, default for UCFCrime,'
                   'else for USCD: e.g. /absolute/datasets/UCSDped2_split_list/10_fold_001/v01_test.txt')

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

    merged_keys, merged_features = merge_keys_and_features_in_one(feature_dict)
    merged_keys.append(False)

    print('Evaluating %s...' % F.save_file_path)
    print('Testing list %s.' % F.testing_list)
    if not os.path.isdir(F.save_file_path):
        _ = eval_one_ckpt(merged_keys, merged_features, D, ckpt_file=F.save_file_path)
    else:
        # model_checkpoint_path: "/absolute/tensorflow_models/190516014752/190516014752.ckpt-16001"
        ckpt_check_list = [i[1][1:-1]
                           for i in basepy.read_txt_lines2list(osp.join(MODEL_FOLDER, 'checkpoint'), sep=': ')][1:]

        _ = [eval_one_ckpt(merged_keys, merged_features, D, ckpt_file=one_ckpt)
            for one_ckpt in ckpt_check_list]

    print('------ Finish ------ Debug Symbol ------ %s ------' % time.asctime(time.localtime(time.time())))


def merge_keys_and_features_in_one(feature_dict):
    dict_keys = feature_dict.keys()
    return list(chain(*[[key] * feature_dict[key].shape[0] for key in dict_keys])), \
           np.concatenate([feature_dict[key] for key in dict_keys], axis=0)
    # a,b =
    # c= [print(b[j+1]) for j,i in enumerate(a) if i != a[j+1]]


def eval_one_ckpt(merged_keys, merged_features, d, ckpt_file=None, npy_folder_suffix='_eval_json'):
    with tf.name_scope('input'):
        input_test = tf.placeholder(tf.float32, [1, 1, d['feature_len']], name='anom')

    with tf.name_scope('forward-propagation'):
        score_anomaly = base.network_fn(input_test,
                                        fusion=d['fusion'],
                                        feature_len=d['feature_len'],
                                        segment_num=1,
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

        step, dict_key, dict_key_json, height = 0, None, [], None
        while merged_keys[step]:
            s = sess.run(score_anomaly,
                         feed_dict={
                             input_test: np.expand_dims(np.expand_dims(merged_features[step][:4096], axis=0), axis=0)})

            class_name, video_name = merged_keys[step].split('@')
            line_to_json = [class_name, video_name] + merged_features[step][4096:].tolist() + s.tolist()[0]

            if merged_keys[step] != dict_key:
                dict_key = merged_keys[step]
                height = [clip for clip in merged_keys[:-1] if clip == dict_key].__len__()
                dict_key_json.append(line_to_json)
            else:
                dict_key_json.append(line_to_json)
                if dict_key_json.__len__() == height:
                    eval_npy_path = basepy.check_or_create_path(ckpt_file + npy_folder_suffix)
                    with open(osp.join(eval_npy_path, dict_key + '.json'), 'w') as f:
                        json.dump(dict_key_json, f)
                    print('%s done.' % merged_keys[step])
                    dict_key, dict_key_json, height = None, [], None
            step += 1
        # END
        print("Test done, at %s" % time.asctime(time.localtime(time.time())))


if __name__ == '__main__':
    tf.app.run()
