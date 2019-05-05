import data_io.basepy as base
import data_io.basetf as base_tf
import tensorflow as tf
import os
import multiprocessing as mp

dataset_path = '/absolute/datasets/anoma'

segment_number = 32
clip_length = 16

# tfrecords_path = dataset_path + '_%d%d_tfrecords' % (clip_length, segment_number)
# _ = base.check_or_create_path(tfrecords_path, create=True, show=True)

sample_path_list = [i for i in range(23)]


def write_1632_tfrecords(sample_path_list, flag1=123):
    print(sample_path_list, flag1)
    return 666


def write_1632_tfrecords2(sample_path_list):
    print(sample_path_list)


# CORRECT usage
base.non_output_multiprocessing(write_1632_tfrecords, sample_path_list, num=int(mp.cpu_count()))
base.non_output_multiprocessing(write_1632_tfrecords, sample_path_list, 111, num=int(mp.cpu_count()))
base.non_output_multiprocessing(write_1632_tfrecords2, sample_path_list, num=int(mp.cpu_count()))

# WRONG usage
base.non_output_multiprocessing(write_1632_tfrecords2, sample_path_list, 111, 222, num=int(mp.cpu_count()))


# *args usage test
def function_with_two_stars(m, *d, t=123):
    arg = (m, d)
    print(d, type(d))
    print(arg)


function_with_two_stars(123, 2, 4)
