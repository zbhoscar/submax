from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import zclips_npy2npy_reform as reform
import zc3d_npy_train as train
import zc3d_npy_eval as evaluation
import zresults_analysis as analysis


tags = tf.flags
F = tags.FLAGS
# step 3 DATA REFORM
tags.DEFINE_string('npy_file_path',
                   '/absolute/datasets/anoma_motion_pyramid_120_85_c3d_npy',
                   'npy file folder to be reformed.')
tags.DEFINE_string('testing_list',
                   '/absolute/datasets/Anomaly-Detection-Dataset/Temporal_Anomaly_Annotation_for_Testing_Videos.txt',
                   'default to UCFCrime, else specific to UCSD.')
tags.DEFINE_boolean('multiprocessing', True, 'choose multiprocessing or not.')
tags.DEFINE_integer('var0', 0, 'choose NPY_FILE_FOLDER, SEGMENT_NUM.')
tags.DEFINE_integer('var1', 0, 'choose MULTISCALE, MULTIREGION.')
# step 4 TRAINING
tags.DEFINE_string('set_gpu', '0', 'Single gpu version, index select')
tags.DEFINE_integer('batch_size', 64, 'batch size.')
tags.DEFINE_integer('epoch_num', 1202, 'epoch number.')
tags.DEFINE_float('learning_rate_base', 0.0008, 'learning rate base')
tags.DEFINE_float('moving_average_decay', 0.999, 'moving average decay')
tags.DEFINE_float('regularization_scale', 0.00003, 'regularization scale')
tags.DEFINE_string('fusion', 'standard', 'fusion ways in feature extraction')
tags.DEFINE_string('lasting', '', 'a TensorFlow model path for lasting')
tags.DEFINE_integer('saving_interval', 10, 'every ? epochs to save')
# step 5 FOR EVALUATION # step 6 FOR ANALYSIS
# '' or /absolute/tensorflow_models/190912162832_anoma_motion_4training_pyramid_80_56_4region_c3d_npy
tags.DEFINE_string('ckpt_path_to_eval', '', ' "" for brand new, or model folder path, or model ckpt file path.')
tags.DEFINE_string('spatial_annotation_path', '/absolute/datasets/anoma_spatial_annotations', 'spatial annotation')


def main(_):
    if not F.ckpt_path_to_eval:
        # step 3
        reform_type, reform_num = (('maxtop', 256), ('maxtop', 512), ('maxtop', 128), ('segment', 32))[F.var0]
        multiscale, multiregion = (('pyramid', 4), ('pyramid', 1), ('single', 4), ('single', 1), (None, None))[F.var1]

        npy_reformed_file_path = reform.npy_reform(F.npy_file_path,
                                                   multiscale, multiregion, reform_type, reform_num,
                                                   F.multiprocessing, F.testing_list)
        # step 4
        tf_model_path = train.network_train(F, npy_reformed_file_path)
    else:
        tf_model_path = 'DIRECT TO STEP5'
    # step 5
    ckpt_path_to_eval = F.ckpt_path_to_eval or tf_model_path
    _ = evaluation.network_eval(ckpt_path_to_eval, F.set_gpu)
    # step 6
    _ = analysis.results_evaluate(ckpt_path_to_eval, '', F.spatial_annotation_path)


if __name__ == '__main__':
    tf.app.run()
