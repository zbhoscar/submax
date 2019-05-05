import os
import os.path as osp
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np
import multiprocessing as mp
from collections import Counter

import data_io.basepy as baseio


def sample_frames_in_scope(frame_scope, sample_num, if_odd, consecutive, sample_type):
    """
    :param frame_scope: sub or not of ['1.jpg', '2.jpg',...,'end.jpg']
    :param sample_num:  sample how many from frame_scope
    :param if_odd:      if make the odd one, else even one
    :param consecutive: sample [1,3,5,7,...] '*.jpg'
    :param sample_type: only for odd one, how to make the odd
    :return:
        get sampled ['*.jpg', ..., '*.jpg'] for a stream of siamese network
    """
    if if_odd:
        if sample_type == 'backwards':
            if not int(consecutive):
                out_tmp = random.sample(frame_scope, sample_num)
                out_tmp = sorted(out_tmp, key=lambda x: int(x.split('.')[0]), reverse=True)
            else:
                sample_len = (sample_num - 1) * max(1, int(consecutive)) + 1
                start = random.randint(0, len(frame_scope) - sample_len)
                out_tmp = frame_scope[start:start + sample_len:int(consecutive)][::-1]
        elif sample_type == 'random':
            sample_again, out_tmp = True, []
            while sample_again is True:
                out_tmp = random.sample(frame_scope, sample_num)
                sample_again = False if out_tmp != sorted(out_tmp, key=lambda x: int(x.split('.')[0])) else True
        else:
            raise ValueError('wrong sample_type: %s' % sample_type)
    else:
        if not int(consecutive):
            out_tmp = random.sample(frame_scope, sample_num)
            out_tmp = sorted(out_tmp, key=lambda x: int(x.split('.')[0]))
        else:
            sample_len = (sample_num - 1) * max(1, int(consecutive)) + 1
            start = random.randint(0, len(frame_scope) - sample_len)
            out_tmp = frame_scope[start:start + sample_len:int(consecutive)]
    return out_tmp


def sample_siamese_list(video_dpath, sample_num=6, siamese_task=(False, False, True), constrain=0.0, consecutive=0.0,
                        sample_type='backwards'):
    """
    :param video_dpath: video directory path, eg.
                            '/absolute/datasets/UCF101pic/CricketBowling/v_CricketBowling_g01_c01'
    :param sample_num: sample how many frames in video. eg. video_dpath =
        /absolute/datasets/UCF101pic/CricketBowling/v_CricketBowling_g01_c01 has frames['1.jpg',...,'20.jpg']
        if sample_num = 6, then get 6 of the ['1.jpg',...,'20.jpg'], for instance ['1.jpg',...,'6.jpg']
    :param siamese_task: [False, False, True] or [False, False, False] for compare or o3n
        eg. [False, False, True] for compare, 3rd stream is aways fault.
            [False, False, False] for o3n, random select a stream to make odd one
            False for even one, True for odd one
    :param constrain:   if constrain, sample frames in a subset, length = constrain * sample_num
        eg. sample_num = 6, constrain = 1.5, subset =  ['1.jpg',...,'9.jpg'], then sample 6 in the subset
    :param consecutive: if consecutive, sample frames in a 【 fixed 】 step. step = int(consecutive) 【 integer part 】
        eg. consecutive = 2.2, get ['1.jpg','3.jpg','5.jpg',...,'11.jpg']   else no 【 fixed 】 step
    :param sample_type: 'backwards' or 'random'
    :return:
        list_out:   get each stream of siamese network a sampled ['*.jpg', ..., '*.jpg']
                    form like [ ['*.jpg', ..., '*.jpg'] * sample_num ]
        label_out:  int for which one is odd
    """
    frame_orig = sorted(os.listdir(video_dpath), key=lambda x: int(x.split('.')[0]))
    if True not in siamese_task:
        label_out = random.randint(0, len(siamese_task) - 1)
        siamese_this = [i if j != label_out else True for j, i in enumerate(siamese_task)]
    else:
        siamese_this = siamese_task
        label_out = siamese_this.index(True)
    if int(constrain):
        sub_length = int(constrain * ((sample_num - 1) * max(1, int(consecutive)) + 1)) + 1
        if sub_length > len(frame_orig):
            raise ValueError('constrain %s, sample_num %s, consecutive %s not suitable for %s' % constrain, sample_num,
                             consecutive, video_dpath)
        frame_start = random.randint(0, len(frame_orig) - sub_length)
        frame_scope = frame_orig[frame_start:frame_start + sub_length]
    else:
        frame_scope = frame_orig
    list_out = [sample_frames_in_scope(frame_scope, sample_num, if_odd, consecutive, sample_type) for if_odd in
                siamese_this]
    return list_out, label_out


def Ht(t):
    out = sum([1 / k for k in range(1, t + 1)])
    return out


def evaluate_siamese_data(video_dpath, siam_samples, crop_size=(224, 224), encoder='dynamic', size_reform='randomcrop',
                          if_show=False, if_save=False):
    """
    :param video_dpath:     video directory path, eg.
                                '/absolute/datasets/UCF101pic/CricketBowling/v_CricketBowling_g01_c01'
    :param siam_samples:    [siamese1[6frames], siamese2[6frames], siamese3[6frames]]
    :param crop_size:       get specific size for input array
    :param encoder:         how to sum up the 6 frames for each siamese
    :param if_show:         kan yi xia
    :param if_save:         bao cun zai ./project
    :param size_reform:     make data to the specific size for the model input
    :return:                [siamese1[encode1], siamese2[encode2], siamese3[encode3]]
    """
    if len(Counter([len(i) for i in siam_samples]).most_common()) != 1:
        raise ValueError('Wrong siam_samples in sample: %s' % video_dpath)
    T = len(siam_samples[0])
    if encoder == 'dynamic':
        xi_shu = [2 * (T - t + 1) - (T + 1) * (Ht(T) - Ht(t - 1)) for t in range(1, T + 1)]
        # FOR sample_to IN multi_sample_to_siamese:
        encoder_out = [sum([xi_shu[j] * baseio.cv2_imread_astype(i, video_dpath) for j, i in enumerate(one_sample)]) for
                       one_sample in siam_samples]
    elif encoder == 'stack_frames':
        encoder_out = [np.concatenate([baseio.cv2_imread_astype(i, video_dpath) for i in one_sample], axis=2) for
                       one_sample in siam_samples]
    else:
        raise ValueError('unknown type of encoder： %s' % encoder)

    # ### get specific shape of input ###
    if size_reform == 'randomcrop':
        encoder_out = [baseio.np_stackimg_crop(baseio.np_stackimg_resize(i, resize=(240, 240), method='minlenbyratio'),
                                               crop_size=(224, 224), method='randomcrop') for i in encoder_out]
    elif size_reform == 'resize':
        encoder_out = [cv2.resize(i, crop_size) for i in encoder_out]
    else:
        raise ValueError('unknown type of size_reform： %s' % size_reform)

    # show or save or both?
    if if_show or if_save:
        video_relative = os.path.split(video_dpath)[-1]
        plt.figure(video_relative, figsize=(16, 10))
        plt.suptitle(video_relative)
        w = T + encoder_out[0].shape[2] // 3
        for i in range(len(encoder_out)):
            for j in range(w):
                plt.subplot(len(encoder_out), w, i * w + j + 1)
                if j < T:
                    plt.title(siam_samples[i][j])
                    img = cv2.imread(os.path.join(video_dpath, siam_samples[i][j]))
                else:
                    plt.title(encoder)
                    img = encoder_out[i][:, :, 3 * (j - T):3 * (j - T) + 3]
                plt.imshow(img[:, :, [2, 1, 0]])
                plt.axis('off')
        if if_save:
            plt.savefig(video_relative + '.png')
        if if_show:
            plt.show()
    return encoder_out


def one(sample, data_dpath, siamese_task=(False, False, True, False), sample_num=6, constrain=1.5, consecutive=1.2,
        sample_type='backwards', encoder='dynamic', crop_size=(224, 224), size_reform='randomcrop', if_show=False,
        if_save=False):
    """
    :data_dpath:            '/absolute/datasets/UCF101pic'
    :sample:                'CricketBowling/v_CricketBowling_g01_c01'
    :sample_siamese_list:   sample names of siam_frames in sample first
    :evaluate_siamese_data: encode siam_frames
    :return:                data.np.float32.array [task_n, 224, 224, 3], int.label 2
    """
    video_dpath = osp.join(data_dpath, sample)
    siam_frames, label = sample_siamese_list(video_dpath, siamese_task=siamese_task, sample_num=sample_num,
                                             constrain=constrain, consecutive=consecutive, sample_type=sample_type)
    arrays = evaluate_siamese_data(video_dpath, siam_frames, crop_size=crop_size, encoder=encoder,
                                   size_reform=size_reform, if_show=if_show, if_save=if_save)
    return np.stack(arrays, axis=0), label, sample


def ones(samples, data_dpath, siamese_task=(False, False, True, False), sample_num=6, constrain=1.5, consecutive=1.2,
         sample_type='backwards', encoder='dynamic', crop_size=(224, 224), size_reform='randomcrop', if_show=False,
         if_save=False):
    """
    :return:    data.np.float32.array [sample_n, task_n, 224, 224, 3], int.label [sample_n]
    """
    wow_arrays, wow_labels = [], []
    for sample in samples:
        arrays, label, _ = one(sample, data_dpath, siamese_task=siamese_task, sample_num=sample_num,
                               constrain=constrain,
                               consecutive=consecutive, sample_type=sample_type, encoder=encoder, crop_size=crop_size,
                               size_reform=size_reform, if_show=if_show, if_save=if_save)
        wow_arrays.append(arrays)
        wow_labels.append(label)
    return np.stack(wow_arrays, axis=0), wow_labels, samples


def ones_multiprocessing(sample_all, data_dpath, siamese_task=(False, False, True, False), sample_num=6, constrain=1.5,
                         consecutive=1.2, sample_type='backwards', encoder='dynamic', crop_size=(224, 224),
                         size_reform='randomcrop', num=int(mp.cpu_count() / 2)):
    ext_num = min(len(sample_all), num)
    split_list = baseio.divide_list(sample_all, ext_num)
    p = mp.Pool(ext_num)
    results = []
    for em in split_list:
        results.append(p.apply_async(
            ones, args=(
                em, data_dpath, siamese_task, sample_num, constrain, consecutive, sample_type, encoder, crop_size,
                size_reform)))
    p.close()
    p.join()
    feed_arrays = np.concatenate([i.get()[0] for i in results], axis=0)
    feed_labels = np.concatenate([i.get()[1] for i in results], axis=0)
    cor_samples = [x for j in [i.get()[2] for i in results] for x in j]
    return feed_arrays, feed_labels, cor_samples
