import numpy as np
import data_io.basepy as basepy
import z1632clips_frame2tfrecords as base_io
import os
import multiprocessing as mp
import time

FOLDER = {'c3d': '/absolute/datasets/anoma_1632_c3d_clips_features',
          'dyn': '/absolute/datasets/anoma_1632_dyn_clips_features'}['c3d']

EMBEDDING = ['_avg', '_max', '_min'][1]


def get_clip2segment_feature(sample_txt, data_type='float32', is_writing=True):
    clip_features = basepy.read_txt_lines2list(sample_txt)

    feature_list = [[] for _ in range(base_io.SEGMENT_NUMBER)]
    for i in clip_features:
        feature = eval(i[0])
        segment_index = eval(i[1])
        feature_list[segment_index].append(feature)

    feature_array = np.zeros((base_io.SEGMENT_NUMBER, feature_list[0][0].__len__()))
    if EMBEDDING == '_avg':
        func = np.mean
    elif EMBEDDING == '_max':
        func = np.max
    elif EMBEDDING == '_min':
        func = np.min
    else:
        raise ValueError('Wrong EMBEDDING method: %s' % EMBEDDING)

    for j in range(len(feature_list)):
        feature_array[j] = func(np.array(feature_list[j]), axis=0)
    output = feature_array.astype(data_type)

    if is_writing:
        np.save(os.path.splitext(sample_txt)[0] + EMBEDDING, output)

    return output


def multi_get_clip2segment_feature(sample_txt_list, data_type='float32', is_writing=True):
    for sample_txt in sample_txt_list:
        try:
            _ = get_clip2segment_feature(sample_txt, data_type=data_type, is_writing=is_writing)
        except all:
            print('something wrong with %s' % sample_txt)


def main():
    # write tfrecords
    sample_path_list = basepy.get_1tier_file_path_list(FOLDER, suffix='.txt')

    basepy.non_output_multiprocessing(multi_get_clip2segment_feature, sample_path_list, 'float32', True,
                                      num=int(mp.cpu_count()))
    print('writing done')


def read_npy_file_path_list(npy_file_path_list, class_name_in_keys=True, sep='@'):
    feed_data = {}
    for npy_file_path in npy_file_path_list:
        file_name = os.path.basename(os.path.splitext(npy_file_path)[0])
        if class_name_in_keys:
            feed_data[file_name] = np.load(npy_file_path)
        else:
            file_name = os.path.splitext(file_name.split(sep)[-1])[0]
            feed_data[file_name] = np.load(npy_file_path)
    return feed_data


def compare_time():
    # single processing compare
    txt_file = './guinea/Arrest@Arrest046_x264.txt'
    start = time.time()
    a = get_clip2segment_feature(txt_file, data_type='float32', is_writing=True)
    middle = time.time()
    npy_file = os.path.splitext(txt_file)[0] + '.npy'
    b = np.load(npy_file)
    end = time.time()
    print(middle - start, end - middle)
    return a, b


if __name__ == '__main__':
    main()
