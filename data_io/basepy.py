import random
import cv2
import os
import numpy as np
import multiprocessing as mp
import math
import matplotlib.pyplot as plt
import json
import copy


def get_2tier_folder_path_list(dataset_path):
    """
    data structure:     DATASET/CLASS/SAMPLE/frames_from_videos
    eg. video dataset_path: '/absolute/datasets/anoma'
        video imagelized:   '/absolute/datasets/anoma/Abuse/Abuse001_x264/00001.jpg', '.../00002.jpg', ...
    :param dataset_path: dataset path
    :return:           ['/absolute/datasets/anoma/Stealing075_x264',
                        '/absolute/datasets/anoma/Stealing061_x264',
                        '/absolute/datasets/anoma/Stealing108_x264', ...]
    """
    out_list = []
    for class_folder_name in os.listdir(dataset_path):
        class_folder_path = os.path.join(dataset_path, class_folder_name)
        for video_folder_name in os.listdir(class_folder_path):
            video_folder_path = os.path.join(class_folder_path, video_folder_name)
            out_list.append(video_folder_path)
    return out_list


def get_1tier_file_path_list(path, suffix=''):
    """
    Get file_path_list in a folder, suffix='' means no limit to the suffix
    :param path:    path/files.tfrecord
    :param suffix:  '.txt' of 'text.txt'
    :return:        ['/absolute/datasets/anoma_1632/Abuse_Abuse001_x264.tfrecord',
                     '/absolute/datasets/anoma_1632/Abuse_Abuse002_x264.tfrecord', ...]
    """
    out_list = []
    for file_name in os.listdir(path):
        if file_name.endswith(suffix):
            file_path = os.path.join(path, file_name)
            out_list.append(file_path)
    return out_list


def get_2tier_dict_list(dict_file):
    """
    Get the 2-tier dict keys in one:
    :param dict_file:   {'video1': {1: [np.random.rand(4096), np.random.rand(4096)],
                                    2: [np.random.rand(4096), np.random.rand(4096), np.random.rand(4096)]},
                         'video2': {3: [np.random.rand(4096), np.random.rand(4096), np.random.rand(4096),
                                        np.random.rand(4096)],
                                    4: [np.random.rand(4096), np.random.rand(4096), np.random.rand(4096),
                                        np.random.rand(4096), np.random.rand(4096)]},
                         }
    :return:        [['video1', 1], ['video1', 2], ['video2', 3], ['video2', 4]]
    """
    video_segment_list = []
    for class_video_name in dict_file.keys():
        for segment_index in dict_file[class_video_name].keys():
            video_segment_list.append([class_video_name, segment_index])
    return video_segment_list


def sort_list_by_name(file_list, sep_sign='.', sep_index=0, reverse=False):
    """
    After os.listdir or something else, elements contain number as index, make them in number order
    :param file_list:    ['00448.jpg', '01878.jpg', '02230.jpg', '00058.jpg', '00486.jpg', '02118.jpg', ..., ]
                     or  [  '448.jpg',  '1878.jpg',  '2230.jpg',    '58.jpg',   '486.jpg',  '2118.jpg', ..., ]
    :param sep_sign:    sep the str， get the SPECIFIC str to sort
    :param sep_index:   the index of SPECIFIC str
    :param reverse:     False: 0,1,2,3,...      ; True: ...,3,2,1,...
    :return:        ['00001.jpg', '00002.jpg', '00003.jpg', '00004.jpg', '00005.jpg', '00006.jpg', ..., ]
                or  [    '1.jpg',     '2.jpg',     '3.jpg',     '4.jpg',     '5.jpg',     '6.jpg', ..., ]
    """
    return sorted(file_list, key=lambda x: int(x.split(sep_sign)[sep_index]), reverse=reverse)


def repeat_list_for_epochs(sample_list, epoch_num=10000, shuffle=True):
    """
    Repeat list for epoch_num times, especially for feed_dict
    :param sample_list:     sample list
    :param epoch_num:       number of epoch
    :param shuffle:         make each epoch in orig or shuffle model
    :return:                (sample_list with shuffle or not) * epoch_num
    """
    if shuffle:
        shuffle_index = [i for i in range(len(sample_list))]
        sample_queue = []
        for i in range(epoch_num):
            random.shuffle(shuffle_index)
            sample_queue.extend([sample_list[i] for i in shuffle_index])
    else:
        sample_queue = sample_list * epoch_num
    return sample_queue


def read_txt_lines2list(file_path, sep=',,'):
    """
    EXAMPLE:            str1,,str2,,str3\n
                        str1,,str2,,str3\n
    :param file_path:   TXT path
    :param sep:         separate symbol between elements
    :return:            [[str1, str2, str3],
                         [str1, str2, str3]]
    """
    with open(file_path, 'r') as f:
        contents = f.readlines()
    return [i.strip().split(sep) for i in contents]


def write_txt_add_lines(file_path, *args, sep=',,'):
    """
    Add string-lized elements to the new line of a txt: *ARGS MUST ALL BE STR
    :param file_path:   txt file path
    :param args:        a set of string-lized elements
    :param sep:         separate string elements in the same line
    :return:
    """
    line_string = ''
    for j, i in enumerate(args):
        line_string = line_string + i + '\n' if j == len(args) - 1 else line_string + i + sep

    with open(file_path, 'a') as f:
        f.writelines(line_string)

    return line_string


def divide_list(full_list, num):
    """
    Divide a list in to NUM pieces, for parallel processing
    :param full_list:   [a,b,c,d,e,f,g]
    :param num:         [3]
    :return:            [[a,d,g],[b,e],[c,f]
    """
    split_list = [[] for _ in range(num)]
    _ = [split_list[j % num].append(full_list[j]) for j in range(len(full_list))]
    return split_list


def check_or_create_path(path, create=True, show=False):
    """
    Given a path, check if the path exists or create it
    :param path:        given path
    :param create:      create or not
    :param show:        print the path status
    :return:            path in one str
    """
    if not os.path.exists(path) and create:
        os.makedirs(path)
        _ = print('Path %s does not exist and has just been created' % path) if show else None
    elif not os.path.exists(path) and not create:
        _ = print('Path %s does not exist' % path) if show else None
    else:
        _ = print('Path %s already exists' % path) if show else None
    return path


def cv2_imread_astype(i, path='./', astype='float32'):
    """ get class numpy after cv2.imread """
    np_img = cv2.imread(os.path.join(path, i))
    if astype == 'float32':
        np_img = np_img.astype(astype) / 255
    elif astype == 'uint8':
        np_img = np_img
    else:
        raise ValueError('Wrong astype of %s' % astype)
    return np_img


def np_stackimg_crop(innp, crop_size=(224, 224), method='randomcrop'):
    """ for eg. [height 224, weight 224, channel maybe > 3 eg=9]
    :method:    normdistcrop:   crop around central by normal distribution
                randomcrop:     just random for each edge
    """
    if method == 'normdistcrop':
        s = np.random.standard_normal(2) + 2.5  # junzhi 2.5, 2wei biaozhun zhengtai fenbu
        crop_start = [int(round(max(min(5, s[0]), 0) * (innp.shape[0] - crop_size[0]) / 5)),
                      int(round(max(min(5, s[1]), 0) * (innp.shape[1] - crop_size[1]) / 5))]
    elif method == 'randomcrop':
        crop_start = [random.randint(0, innp.shape[j] - i) for j, i in enumerate(crop_size)]
    else:
        raise ValueError('Wrong method in np_stackimg_crop: %s' % method)
    return innp[crop_start[0]:crop_start[0] + crop_size[0], crop_start[1]:crop_start[1] + crop_size[1], :]


def np_stackimg_resize(innp, resize=(224, 224), method='minlenbyratio'):
    """ for eg. [height 224, weight 224, channel maybe > 3 eg=9]
    :method: minlenbyratio: set edge's min len for orig img, keep ratio. eg. min=5, [5,1]->[25,5], [1,5]->[5,25]
             maxlenbyratio: set edge's max len for orig img, keep ratio. eg. max=5, [10,15]->[3,5], [15,10]->[5,3]
             absolute:      set absolute shape for orig img. eg. abs=[5,5], [10,15]->[5,5]
    """
    shape = innp.shape
    if method == 'minlenbyratio':
        ratio = max(max([resize[i] / shape[i] for i in [0, 1]]), 1)
        new_shape = (int(shape[0] * ratio), int(shape[1] * ratio))
    elif method == 'maxlenbyratio':
        ratio = min(min([resize[i] / shape[i] for i in [0, 1]]), 1)
        new_shape = (int(shape[0] * ratio), int(shape[1] * ratio))
    elif method == 'absolute':
        ratio, new_shape = 331, resize
    else:
        raise ValueError('Wrong method for np_stackimg_resize: %s' % method)
    return innp if ratio == 1 else cv2.resize(innp, new_shape[::-1])


def non_output_multiprocessing(func, todo_list, *args, num=int(mp.cpu_count())):
    """
    Do not need func output, use this to parallel the func
    usage:
        non_output_multiprocessing(func1, divided_list, num=int(mp.cpu_count()))
        non_output_multiprocessing(func1, divided_list, var1, var2, num=int(mp.cpu_count()))
        non_output_multiprocessing(func2, divided_list, num=int(mp.cpu_count()))
    :param func:        MUST be in the FORM to suit uncertain length of *args:
                            def func1(divided_list, flag1=var1, flag2=var2):
                            def func2(divided_list):
    :param todo_list:   full list to process
    :param num:         parallel N times
    :return:            none
    """
    ext_num = min(len(todo_list), num)
    split_list = divide_list(todo_list, ext_num)
    p = mp.Pool(ext_num)
    for em in split_list:
        params = [em]
        params.extend(args)
        p.apply_async(func, args=tuple(params))
    p.close()
    p.join()


def get_overlap_start_index(list_length, clip_length=16, overlap=0):
    """
    get clip start index in a list, overlap is the minimum overlap num
    :param list_length:     the length of the list
    :param clip_length:     the length of the clip
    :param overlap:         minimum overlap in this processing
    :return:                a list of clip start index, unfixed index number
    """
    clips_number = max(1, math.ceil((list_length - clip_length * overlap) / (clip_length - clip_length * overlap)))
    overlap_start = [round(i * (list_length - clip_length) / max(1, (clips_number - 1))) for i in range(clips_number)]
    return overlap_start


def get_segment_start_index(list_length, segment_num=32, clip_length=16):
    """
    Separate list_length in to segment_num of segments, each as least clip_length
    :param list_length:     480
    :param segment_num:     32
    :param clip_length:     16
    :return:                the start index in the list
    """
    if list_length >= segment_num * clip_length:
        segment_start_index = [round(i * list_length / segment_num)
                               for i in range(segment_num)] + [list_length]
    else:
        segment_start_index = [round(i * (list_length - clip_length) / (segment_num - 1))
                               for i in range(segment_num)] + [list_length]
    return segment_start_index


def TPR_FPR(test_label_list, real_label_list, bool_draw=True, sample_num=100):
    """
    TPR, FPR, ROC, AUC analysis
    :param test_label_list:     LIST of probability on detection for each sample, [0 < probability < 1]
                                    [0.123, 0.99, 0.002, 0.56, ..., 0.88]
    :param real_label_list:     LIST of real binary label for each sample
                                    [0, 1, 0, 0, ..., 1]
    :param bool_draw:           True for drawing ROC curve, False for none
    :param sample_num:          related to the num of threshold, from 0< threshold <1
    :return:                    TPR in different thresholds,
                                FPR in different thresholds,
                                ROC figure,
                                AUC rate,
                                precision.
    """
    # threshold_list = [(i + 1) / sample_num for i in range(sample_num)]
    threshold_list = [0.5]
    for i in range(2, sample_num):
        gap = 1 / 1.28 ** i
        threshold_list.extend([gap, 1 - gap])

    TPR_points, FPR_points, precisions = [], [], []
    for threshold in sorted(threshold_list):
        test_binarized = [0 if i < threshold else 1 for i in test_label_list]
        TP, FN, FP, TN = 0, 0, 0, 0
        for j, i in enumerate(real_label_list):
            if test_binarized[j] == 1 and i == 1:
                TP += 1
            elif test_binarized[j] == 1 and i == 0:
                FP += 1
            elif test_binarized[j] == 0 and i == 1:
                FN += 1
            else:
                TN += 1
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        precision = (TP + TN) / (TP + TN + FP + FN)
        TPR_points.append(TPR)
        FPR_points.append(FPR)
        precisions.append(precision)

    TPR_points_ = [1] + TPR_points + [0]
    FPR_points_ = [1] + FPR_points + [0]

    AUC = np.trapz(TPR_points[::-1], x=FPR_points[::-1])
    AUC_ = np.trapz(TPR_points_[::-1], x=FPR_points_[::-1])

    if bool_draw:
        plt.subplot(1, 2, 1)
        plt.plot(FPR_points, TPR_points)
        plt.axis([0, 1, 0, 1])
        plt.subplot(1, 2, 2)
        plt.plot(FPR_points_, TPR_points_)
        plt.axis([0, 1, 0, 1])
        plt.show()

    return TPR_points, FPR_points, AUC, AUC_, precisions


def Ht(t):
    """
    t-th Harmonic number
    :param t:   0 ~ ∞
    :return:    Σ1~n 1/t, special: Ht(0) = 0
                eg. 0, 1, 1.5, 1.8333, 2.0833
    """
    out = sum([1 / k for k in range(1, t + 1)])
    return out


class DictCtrl(object):
    """
    Use default_dict to add keys in a program:
        test = DictCtrl(default_dict)
    encode and save:
        encoded = test.save2path(json_path=None, **kwargs)      # None for do not save, else to a path
    read and decode:
        decoded = test.read4path(json_path, decode=True)        # decode for a new dict_dict, else just json_path
    """
    def __init__(self, default_dict):
        self.default_dict = default_dict

    def update_keys(self, kwargs):
        temp = copy.deepcopy(self.default_dict)
        for i in kwargs.keys():
            temp[i] = kwargs[i]
        return temp

    def encode2dict(self, **kwargs):
        return self.update_keys(kwargs)

    def decode2dict(self, kwargs):
        return self.update_keys(kwargs)

    def dict2json(self, d, json_path):
        with open(json_path, 'w') as f:
            json.dump(d, f)

    def json2dict(self, json_path):
        with open(json_path, 'r') as f:
            return json.load(f)

    def save2path(self, json_path=None, **kwargs):
        d = self.encode2dict(**kwargs)
        if json_path is not None:
            self.dict2json(d, json_path)
        return d

    def read4path(self, json_path, decode=True):
        d = self.json2dict(json_path)
        if decode:
            d = self.decode2dict(d)
        return d
