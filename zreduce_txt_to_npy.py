import data_io.basepy as basepy
import os.path as osp
import random
import multiprocessing as mp
import numpy as np
import time

SRC_TXT_PATH = '/absolute/ext3t/anoma_motion16_c3d_features'
RED_NPY_PATH = basepy.check_or_create_path(SRC_TXT_PATH.replace('c3d_features', 'reduced_npy'))

REDUCE_MODEL, REDUCE_NUM = (('standard', 2019), ('max', 1001))[1]


def get_reduced_npy(sample_txt, data_type='float32', is_writing=True):
    clip_features = basepy.read_txt_lines2list(sample_txt)
    if REDUCE_MODEL == 'standard':
        clip_features = random.sample(clip_features, REDUCE_NUM) if len(clip_features) > REDUCE_NUM else clip_features
    elif REDUCE_MODEL == 'max':
        # clip_features = sorted()
        clip_features = clip_features

    feature_list = [[] for _ in range(len(clip_features))]
    for j, i in enumerate(clip_features):
        # i:
        # str '[num]*4096', str 'index', str 'nj', str 'c', str 'r', str 'w', str 'h', str' max', str 'min'
        feature = eval(i[0])
        frame_index, nj = eval(i[1]), eval(i[2])
        c, r, w, h = eval(i[3]), eval(i[4]), eval(i[5]), eval(i[6])
        max_value, min_value = eval(i[7]), eval(i[8])
        feature_list[j] = feature + [frame_index, nj, c, r, w, h, max_value, min_value]

    output = np.array(feature_list, dtype=data_type)

    if is_writing:
        np.save(osp.join(RED_NPY_PATH, osp.basename(sample_txt).split('.')[0]), output)

    return output


def multi_get_reduced_npy(sample_txt_list, data_type='float32', is_writing=True):
    for sample_txt in sample_txt_list:
        try:
            _ = get_reduced_npy(sample_txt, data_type=data_type, is_writing=is_writing)
        except all:
            print('something wrong with %s' % sample_txt)


def main():
    # write tfrecords
    txt_path_list = basepy.get_1tier_file_path_list(SRC_TXT_PATH, suffix='.txt')
    npy_done_list = basepy.get_1tier_file_path_list(RED_NPY_PATH, suffix='.npy')
    npy_done_str = str(npy_done_list)

    remaining_txt_list = [i for i in txt_path_list if osp.basename(i).split('.')[0] not in npy_done_str]
    print('%d in all, %d .npy exist, %d .txt remaining' %
          (len(txt_path_list), len(npy_done_list), len(remaining_txt_list)), '...')

    random.shuffle(remaining_txt_list)

    basepy.non_output_multiprocessing(multi_get_reduced_npy, remaining_txt_list, 'float32', True,
                                      num=int(mp.cpu_count()))
    print('writing done')


def get_remaining_list(path1=SRC_TXT_PATH, suffix1='.txt', path2=RED_NPY_PATH, suffix2='.npy'):
    # write tfrecords
    txt_path_list = basepy.get_1tier_file_path_list(path1, suffix=suffix1)
    npy_done_list = basepy.get_1tier_file_path_list(path2, suffix=suffix2)
    npy_done_str = str(npy_done_list)

    remaining_txt_list = [i for i in txt_path_list if osp.basename(i).split('.')[0] not in npy_done_str]
    print('%d in all, %d .npy exist, %d .txt remaining' %
          (len(txt_path_list), len(npy_done_list), len(remaining_txt_list)), '...')
    return txt_path_list, npy_done_list, remaining_txt_list


def read_npy_file_path_list(npy_file_path_list, class_name_in_keys=True, sep='@'):
    feed_data = {}
    for npy_file_path in npy_file_path_list:
        file_name = osp.basename(osp.splitext(npy_file_path)[0])
        if class_name_in_keys:
            feed_data[file_name] = np.load(npy_file_path)
        else:
            file_name = osp.splitext(file_name.split(sep)[-1])[0]
            feed_data[file_name] = np.load(npy_file_path)
    return feed_data


def compare_time():
    # single processing compare
    txt_file = './guinea/Arrest@Arrest046_x264.txt'
    start = time.time()
    a = get_reduced_npy(txt_file, data_type='float32', is_writing=True)
    middle = time.time()
    npy_file = osp.splitext(txt_file)[0] + '.npy'
    b = np.load(npy_file)
    end = time.time()
    print(middle - start, end - middle)
    return a, b


if __name__ == '__main__':
    main()
