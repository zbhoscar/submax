import json
import data_io.basepy as basepy
import os.path as osp
import numpy as np
import zdefault_dict
from sklearn import metrics
from sklearn import manifold
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import cv2


def main(_):
    tags = tf.flags
    F = tags.FLAGS
    tags.DEFINE_string('results_json_path',
                       '/home/zbh/Desktop/absolute/tensorflow_models/191007174553_anoma_motion_reformed_single_120_85_1region_maxtop_256_c3d_npy/191007174553.ckpt-14302_eval_json',
                       'model folder path, or model ckpt file path:'
                       '/absolute/tensorflow_models/190918230353_anoma_motion_reformed_pyramid_120_85_1region_maxtop_1000_c3d_npy/190918230353.ckpt-9619_eval_json'
                       '/absolute/tensorflow_models/190918230353_anoma_motion_reformed_pyramid_120_85_1region_maxtop_1000_c3d_npy')
    tags.DEFINE_string('save_plot', '', 'where to save figs, default: "./temp/test_savefig", "" for NO FIG SAVE.')

    results_evaluate(F.results_json_path, F.save_plot)


def results_evaluate(results_json_path, save_plot):
    if 'anoma' in results_json_path:
        spatial_annotation_path = '/absolute/datasets/anoma_spatial_annotations'
    elif 'UCSD' in results_json_path:
        spatial_annotation_path = '/absolute/datasets/UCSDped2_spatial_annotation'
    else:
        raise ValueError('Wrong results_json_path: %s' % results_json_path)

    keys_json_path = basepy.get_1tier_file_path_list(results_json_path, suffix='keys.json')[0] \
        if basepy.get_1tier_file_path_list(results_json_path, suffix='keys.json') \
        else basepy.get_1tier_file_path_list(osp.dirname(results_json_path), suffix='keys.json')[0]

    d = basepy.DictCtrl(zdefault_dict.EXPERIMENT_KEYS).read4path(keys_json_path)

    results_json_path, temporal_annotation_file = results_json_path, d['testing_list']
    # SUFFIX = '.jpg' if 'anoma' in results_json_path else '.tif'
    inflate = 16 if 'anoma' in results_json_path else 8

    print('RESULTS_JSON_PATH'.ljust(25), results_json_path)
    print('TEMPORAL_ANNOTATION_FILE'.ljust(25), temporal_annotation_file)
    # print('SUFFIX'.ljust(25), SUFFIX)
    print('INFLATE'.ljust(25), inflate)
    print('------ Debug This To Choose Different Evaluation ------')

    if '_eval_json' in results_json_path[-10:]:
        results_of_one = analysis_in_one_ckpt(results_json_path,
                                              temporal_annotation_file, inflate, save_plot, spatial_annotation_path)
        # fpr, tpr, thresholds, auc, all_videos_map,  \
        # vfpr, vtpr, vthresholds, vauc, videos_order, \
        # fpr_s, tpr_s, thresholds_s, auc_s, vfpr_s, \
        # vtpr_s, vthresholds_s, vauc_s, spatial_p, spatial_r, \
        # spatial_threholds, recover_rate, info = results_of_one
        print(results_of_one[3], results_of_one[8], results_of_one[13], results_of_one[17], results_json_path)
        print(metrics.auc(results_of_one[19], results_of_one[18]), results_of_one[21])
        rr = results_of_one[21]
        r = [rr] + [i* rr for i in results_of_one[19]]
        p = [0] + [j for j in results_of_one[18]]
        plt.plot(r, p)
        plt.show()
        np.savetxt('ours_fpr.txt', results_of_one[10])
        np.savetxt('ours_tpr.txt', results_of_one[11])
        np.savetxt('ours_r.txt', np.array(r))
        np.savetxt('ours_p.txt', np.array(p))
        a = 0
        n = 0
        for i in results_of_one[-1]:
            if 'normal' in i[1].lower() and i[-1] > 0.5:
                a += 1
            elif 'normal' in i[1].lower() and i[-1] <= 0.5:
                n += 1
    else:
        _eval_json_list = basepy.get_1tier_file_path_list(results_json_path, suffix='_eval_json')
        _eval_json_list = sorted(_eval_json_list, key=lambda x: int(x.split('_eval_json')[0].split('ckpt-')[1]))
        results_all_in_one = [
            analysis_in_one_ckpt(i, temporal_annotation_file, inflate, save_plot, spatial_annotation_path)
            for i in _eval_json_list]
        _ = [print(i[3], i[8], i[13], i[17], metrics.auc(i[19], i[18]), i[21], osp.basename(_eval_json_list[j]))
             for j, i in enumerate(results_all_in_one)]
        print('RESULTS_JSON_PATH'.ljust(25), results_json_path)

    print('ANALYSIS DONE ------ Debug Symbol ------ %s ------' % time.asctime(time.localtime(time.time())))


def analysis_in_one_ckpt(results_json_path, temporal_annotation_file, inflate, save_plot, annotation_folder_path):
    path_if_save = osp.join(save_plot, osp.basename(osp.dirname(results_json_path)), osp.basename(results_json_path))
    # video_name, deflated_length, temporal_score, temporal_truth
    video_temporal_results, info_all = get_temporal_duration_in_folder(
        results_json_path, temporal_annotation_file, inflate, suffix='.json',
        score_and_truth_fig_save_path= path_if_save if save_plot else None)
    all_frames_score, all_frames_score_select, all_frames_truth, all_videos_map = [], [], [], []
    videos_score, video_score_select, videos_truth, videos_order = [], [], [], []
    for video_name, deflated_length, temporal_score, temporal_score_select, temporal_truth in video_temporal_results:
        all_frames_score.extend(temporal_score)
        all_frames_score_select.extend(temporal_score_select)
        all_frames_truth.extend(temporal_truth)
        all_videos_map.append([video_name] * deflated_length)
        videos_score.append(max(temporal_score))
        video_score_select.append(max(temporal_score_select))
        videos_truth.append(max(temporal_truth))
        videos_order.append(video_name)
    # t, f, auc, auc_, precision_list = basepy.TPR_FPR_calc(all_frames_score, all_frames_truth, bool_draw=bool_draw)
    # return t, f, auc, auc_, precision_list, osp.basename(results_json_path), all_videos_map
    fpr, tpr, thresholds = metrics.roc_curve(all_frames_truth, all_frames_score, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    vfpr, vtpr, vthresholds = metrics.roc_curve(videos_truth, videos_score, pos_label=1)
    vauc = metrics.auc(vfpr, vtpr)

    fpr_s, tpr_s, thresholds_s = metrics.roc_curve(all_frames_truth, all_frames_score_select, pos_label=1)
    auc_s = metrics.auc(fpr_s, tpr_s)
    vfpr_s, vtpr_s, vthresholds_s = metrics.roc_curve(videos_truth, video_score_select, pos_label=1)
    vauc_s = metrics.auc(vfpr_s, vtpr_s)

    spatial_p, spatial_r, spatial_threholds, recover_rate = \
        get_spatial_pr_curve(info_all, annotation_folder_path, temporal_annotation_file, inflate)

    if save_plot:
        save_one_fig(fpr, tpr, 'AUC = %f', 'False Positive Rate', 'True Positive Rate',
                     path_if_save.split('_eval_json')[0] + '_orig.png', for_title=auc)
        save_one_fig(vfpr, vtpr, 'AUC = %f', 'False Positive Rate', 'True Positive Rate',
                     path_if_save.split('_eval_json')[0] + '_video.png', for_title=vauc)
        save_one_fig(fpr_s, tpr_s, 'AUC = %f', 'False Positive Rate', 'True Positive Rate',
                     path_if_save.split('_eval_json')[0] + '_smooth.png', for_title=auc_s)
        save_one_fig(vfpr_s, vtpr_s, 'AUC = %f', 'False Positive Rate', 'True Positive Rate',
                     path_if_save.split('_eval_json')[0] + '_video_smooth.png', for_title=vauc_s)
        save_one_fig(spatial_r, spatial_p, 'RECOVER = %f', 'Recall', 'Precision',
                     path_if_save.split('_eval_json')[0] + '_pr_curve.png', for_title=recover_rate)
        if 'ucsd' in results_json_path.lower():
            data_path = '/absolute/datasets/UCSDped2_reform'
        elif 'anoma' in results_json_path.lower():
            data_path = '/absolute/datasets/anoma'
        else:
            raise ValueError('Wrong data_path %s' % results_json_path)

        draw_spatial(info_all, data_path=data_path, save_path=path_if_save)

    return fpr, tpr, thresholds, auc, all_videos_map, \
           vfpr, vtpr, vthresholds, vauc, videos_order, \
           fpr_s, tpr_s, thresholds_s, auc_s, vfpr_s, \
           vtpr_s, vthresholds_s, vauc_s, spatial_p, spatial_r, \
           spatial_threholds, recover_rate, info_all


def save_one_fig(x, y, title_str, xlabel_str, ylabel_str, save_file_path, for_title=''):
    # 'RECOVER%s' % '' == 'RECOVER'
    plt.plot(x, y)
    plt.title(title_str % for_title)
    plt.xlabel(xlabel_str)
    plt.ylabel(ylabel_str)
    plt.savefig(save_file_path)
    plt.close()


def get_spatial_pr_curve(results_all_in_one, annotation_folder_path, temporal_annotation_file, inflate,
                         iou_threshold=0.14):
    annotation_in_all = basepy.read_txt_lines2list(temporal_annotation_file, '  ')
    image_size = (240, 320) if 'Anomaly-Detection-Dataset' in temporal_annotation_file else (240, 360)
    wei_shu = 5 if 'Anomaly-Detection-Dataset' in temporal_annotation_file else 3
    spatial_annotation, all_annotation_num = {}, 0
    for i in annotation_in_all:
        video_name_temp = i[0].split('.')[0]
        index_list = list(range(int(i[2]), int(i[3]))) + list(range(int(i[4]), int(i[5])))
        in_one_video = {}
        for j in index_list:
            if j % inflate == 0:
                spatial_annotation_txt = osp.join(annotation_folder_path, video_name_temp,
                                                  str(j).zfill(wei_shu) + '.txt')
                if not osp.exists(spatial_annotation_txt):
                    raise ValueError('Not Exists: annotation txt path %s' % spatial_annotation_txt)
                in_one_video[j] = [[yoloLine2Shape(image_size, k[1], k[2], k[3], k[4]), 0] for k in
                                   basepy.read_txt_lines2list(spatial_annotation_txt, ' ')]
                all_annotation_num = all_annotation_num + len(in_one_video[j])
        spatial_annotation[video_name_temp] = in_one_video

    spatial_groud_truth, covered_num = get_spatial_groud_truth(results_all_in_one, spatial_annotation, scale_id=1,
                                                               iou_threshold=iou_threshold)
    spatial_anomaly_score = [i[-1] for i in results_all_in_one]
    p, r, thresholds = metrics.precision_recall_curve(spatial_groud_truth, spatial_anomaly_score)

    return p, r, thresholds, covered_num / all_annotation_num


def get_spatial_groud_truth(results_all_in_one, spatial_annotation, scale_id=2, iou_threshold=0.5):
    spatial_groud_truth = [0] * len(results_all_in_one)
    for j, one in enumerate(results_all_in_one):
        try:
            _, video_name, frame_index, _, a1_c, a1_r, a1_w, a1_h, a2_c, a2_r, a2_w, a2_h, _, _, _ = one
        except ValueError:
            _, video_name, frame_index, _, a1_c, a1_r, a1_w, a1_h, _, _, _ = one
            a2_c, a2_r, a2_w, a2_h = a1_c, a1_r, a1_w, a1_h
        if scale_id == 2:
            area = (int(a2_c), int(a2_r), int(a2_c + a2_w), int(a2_r + a2_h))
        elif scale_id == 1:
            area = (int(a1_c), int(a1_r), int(a1_c + a1_w), int(a1_r + a1_h))
        else:
            raise ValueError('Wrong select in results %d' % scale_id)

        frame_index = int(frame_index)
        # print(j, video_name, frame_index, area)
        if video_name in spatial_annotation.keys() and frame_index in spatial_annotation[video_name].keys():
            # print(video_name, frame_index)
            for id, [area_gt, _] in enumerate(spatial_annotation[video_name][frame_index]):
                if compute_iou(area, area_gt) >= iou_threshold:
                    spatial_groud_truth[j] = 1
                    spatial_annotation[video_name][frame_index][id][1] = 1

    covered_num = 0
    for video in spatial_annotation:
        for index in spatial_annotation[video]:
            for area, selected in spatial_annotation[video][index]:
                covered_num = covered_num + selected

    return spatial_groud_truth, covered_num


def get_temporal_duration_in_folder(results_json_path, temporal_annotation_file, inflate,
                                    suffix='.json', score_and_truth_fig_save_path=None):
    json_file_list = basepy.get_1tier_file_path_list(results_json_path, suffix=suffix)
    videos_temporal_results = []
    info_all_in_one = []
    if score_and_truth_fig_save_path:
        print('drawing to %s...' % score_and_truth_fig_save_path)

    for json_file in json_file_list:
        video_name, deflated_length, temporal_score, temporal_score_select, temporal_truth, info = \
            get_temporal_duration(json_file, inflate, temporal_annotation_file)
        if score_and_truth_fig_save_path:
            show_something(osp.join(score_and_truth_fig_save_path, 'temporal_orig'), deflated_length, inflate,
                           temporal_score, temporal_truth, video_name)
            show_something(osp.join(score_and_truth_fig_save_path, 'temporal_select'), deflated_length, inflate,
                           temporal_score_select, temporal_truth, video_name)
        videos_temporal_results.append([video_name, deflated_length,
                                        temporal_score, temporal_score_select, temporal_truth])
        info_all_in_one.extend(info)
    return videos_temporal_results, info_all_in_one


def convert_list2dict(results_list, select=2):
    new_dict = {}
    for _, video_name, frame_index, _, a1_c, a1_r, a1_w, a1_h, a2_c, a2_r, a2_w, a2_h, _, _, _ in results_list:
        if select == 2:
            area = (a2_c, a2_r, a2_c + a2_w, a2_r + a2_h)
        elif select == 1:
            area = (a1_c, a1_r, a1_c + a1_w, a1_r + a1_h)
        else:
            raise ValueError('Wrong select in results %d' % select)

        if video_name not in new_dict.keys():
            new_dict[video_name] = {int(frame_index): [area]}
        elif int(frame_index) not in new_dict[video_name].keys():
            new_dict[video_name][int(frame_index)] = [area]
        else:
            new_dict[video_name][int(frame_index)].append(area)
    return new_dict


def show_something(score_and_truth_fig_save_path, deflated_length, inflate,
                   temporal_score, temporal_truth, video_name):
    _ = basepy.check_or_create_path(score_and_truth_fig_save_path)
    x = np.linspace(0, deflated_length * inflate, deflated_length)
    y1 = np.array(temporal_score)
    y2 = np.array(temporal_truth)
    # plt.figure()
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.ylim(0, 1.1)
    plt.xlim(0, deflated_length * inflate + 4)
    plt.title(video_name)
    plt.xlabel('Frame number')
    plt.ylabel('Anomaly score')
    plt.savefig(osp.join(score_and_truth_fig_save_path, video_name) + '.png')
    plt.close()


def get_temporal_duration(json_file, inflate, temporal_annotation_file):
    video_name = json_file.split('@')[-1].split('.')[0]
    # print('getting temporal duration: %s' % video_name)
    # json_file = '/absolute/tensorflow_models/191007174553_anoma_motion_reformed_single_180_127_4region_maxtop_256_c3d_npy/191007174553.ckpt-15188_eval_json/normal_test@Normal_Videos_876_x264.json'
    with open(json_file, 'r') as f:
        info = json.load(f)
    last_at_start_index = max([i[2] for i in info])
    # get max frame index in .json
    zero_to_last_index = range(int(last_at_start_index + 1))
    annotation_in_all = basepy.read_txt_lines2list(temporal_annotation_file)
    video_in_annotation = [i[0] for i in annotation_in_all if video_name in i[0]]
    if video_in_annotation.__len__() != 1:
        raise ValueError('Too many %s in %s' % (video_name, video_in_annotation))
    else:
        video_in_annotation = video_in_annotation[0]
    video_mp4_name, video_class, start1, final1, start2, final2 = video_in_annotation.split('  ')
    start1, final1, start2, final2 = int(start1), int(final1), int(start2), int(final2)
    # get annotation in all frames (max is max in info)
    frame_annotation = [1 if start1 <= index <= final1 or start2 <= index <= final2 else 0
                        for index in zero_to_last_index]

    temporal_truth = [i for j, i in enumerate(frame_annotation) if j % inflate == 0]
    temporal_score = [-1] * len(temporal_truth)
    for line in info:
        frame_index, anomaly_score = line[2], line[-1]
        if frame_index % inflate == 0:
            index_deflated = int(frame_index // inflate)
            temporal_score[index_deflated] = max(temporal_score[index_deflated], anomaly_score)
    if -1 in temporal_score:
        raise ValueError('Missing index in %s' % json_file)
    if len(temporal_score) != len(temporal_truth):
        raise ValueError('temporal_score and temporal_truth do not match in number in %s' % json_file)

    temporal_score_select = [0] * len(temporal_truth)
    select_num = int(osp.dirname(osp.dirname(json_file)).split('_')[-3])
    select_num = select_num * 4 if '4region' in json_file else select_num
    info2 = sorted(info, key=lambda x: x[-3], reverse=True)[:select_num]
    for line in info2:
        frame_index, anomaly_score = line[2], line[-1]
        if frame_index % inflate == 0:
            index_deflated = int(frame_index // inflate)
            temporal_score_select[index_deflated] = max(temporal_score_select[index_deflated], anomaly_score)

    window_length = min(2 * int((len(temporal_score) - 1) / 2 / 2) + 1, 9)
    window = [1/window_length] * window_length
    temporal_score_select_smooth = np.convolve(temporal_score_select, window, mode='same')
    temporal_score_select_smooth = np.minimum(temporal_score_select_smooth, 0.9999)
    # info_smooth = info
    # for j, one_clip in enumerate(info_smooth):
    #     index = int(one_clip[2] // inflate)
    #     if one_clip[-1] == 0:
    #         one_clip[-1] = temporal_score_select_smooth[index] / 100
    #     elif temporal_score_select[index] == 0:
    #         one_clip[-1] = temporal_score_select_smooth[index]
    #     else:
    #         # if temporal_score_select[index] == 0 :
    #         #     print(json_file, index)
    #         one_clip[-1] = (one_clip[-1] / temporal_score_select[index]) * temporal_score_select_smooth[index]
    # info_smooth = sorted(info_smooth, key=lambda x: x[-3], reverse=True)[:select_num]

    return video_name, len(temporal_score), temporal_score, temporal_score_select_smooth, temporal_truth, info


def gaussian(x, u, sigma):
    return np.exp(-(x - u) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)


def yoloLine2Shape(image_size, xcen, ycen, w, h):
    # xcen, ycen, w, h = float(xcen), float(ycen), float(w), float(h)
    xmin = max(float(xcen) - float(w) / 2, 0)
    xmax = min(float(xcen) + float(w) / 2, 1)
    ymin = max(float(ycen) - float(h) / 2, 0)
    ymax = min(float(ycen) + float(h) / 2, 1)

    xmin = int(image_size[1] * xmin)
    xmax = int(image_size[1] * xmax)
    ymin = int(image_size[0] * ymin)
    ymax = int(image_size[0] * ymax)

    return xmin, ymin, xmax, ymax


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def draw_spatial(info_all, data_path='/absolute/datasets/UCSDped2_reform', save_path='./temp/UCSDped2_spatial', select_id=2):
    suffix = '.tif' if 'ucsd' in data_path.lower() else '.jpg'
    inflate = 8 if 'ucsd' in data_path.lower() else 16
    for one_info in info_all:
        class_name, video_name, frame_index, _, \
        a1_c, a1_r, a1_w, a1_h, a2_c, a2_r, a2_w, a2_h, \
        _, _, anomaly_score = one_info

        if select_id == 2:
            area = (a2_c, a2_r, a2_c + a2_w, a2_r + a2_h)
        elif select_id == 1:
            area = (a1_c, a1_r, a1_c + a1_w, a1_r + a1_h)
        else:
            raise ValueError('Wrong select in results %d' % select_id)
        frame_index, area = int(frame_index), [int(i) for i in area]

        video_orig_path = osp.join(data_path,class_name,video_name)
        # print(video_orig_path)
        frame_sorted = sorted(basepy.get_1tier_file_path_list(video_orig_path, suffix=suffix),
                              key=lambda x: int(osp.basename(x).split('.')[0]))

        which_frame = frame_sorted[frame_index+int(inflate/2)]
        save_frame = osp.join(save_path, video_name, osp.basename(which_frame))
        # save_frame = which_frame.replace(data_path,save_path)
        _ = basepy.check_or_create_path(osp.dirname(save_frame))

        img = cv2.imread(save_frame) if osp.isfile(save_frame) else cv2.imread(which_frame)

        cv2.rectangle(img, (area[0],area[1]), (area[2],area[3]), (int(255*anomaly_score),0,0),1)
        font = cv2.FONT_HERSHEY_PLAIN
        text = '%.4f' % anomaly_score
        cv2.putText(img,text,(area[0],area[1]),font,1,(int(255*anomaly_score),0,0),1)
        cv2.imwrite(save_frame, img)


def tsne(class_norm, clase_anom, title='tsne of anomalous and normal features'):
    in_one = np.vstack((class_norm, clase_anom))
    class1_num = len(class_norm)
    tsne = manifold.TSNE(n_components=2, )
    X_tsne = tsne.fit_transform(in_one)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure(title)
    ax = plt.gca()
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.scatter(X_norm[class1_num:, 0], X_norm[class1_num:, 1], c='r', alpha=0.5)
    ax.scatter(X_norm[:class1_num, 0], X_norm[:class1_num, 1], c='b', alpha=0.5)
    plt.show()



# def tsne3(class_norm, clase_anom):
#     in_one = np.vstack((class_norm, clase_anom))
#     class1_num = len(class_norm)
#     tsne = manifold.TSNE(n_components=3)
#     X_tsne = tsne.fit_transform(in_one)
#     x_min, x_max = X_tsne.min(0), X_tsne.max(0)
#     X_norm = (X_tsne - x_min) / (x_max - x_min)
#     plt.figure('tsne of anomalous and normal features')
#     ax = plt.gca()
#     # ax.set_xlabel('x')
#     # ax.set_ylabel('y')
#     ax.scatter(X_norm[:class1_num, 0], X_norm[:class1_num, 1], X_norm[:class1_num, 2], c='b')
#     ax.scatter(X_norm[class1_num:, 0], X_norm[class1_num:, 1], X_norm[class1_num:, 2], c='r')
#     ax.view_init(4, -72)
#     plt.show()



def collect_features(temporal_annotation_file, original_c3d_path, event_proposal_c3d_path, spatial_annotation_path, max_num=None):
    annotation_in_all = basepy.read_txt_lines2list(temporal_annotation_file)
    original_c3d_anomaly_all, original_c3d_normal_all, event_proposal_c3d_anomaly_all, event_proposal_c3d_normal_all = [],[],[],[]
    for each_line in annotation_in_all:
        original_c3d_anomaly, original_c3d_normal, event_proposal_c3d_anomaly, event_proposal_c3d_normal = \
            collect_each_video(each_line,original_c3d_path, event_proposal_c3d_path, spatial_annotation_path, max_num=max_num)
        original_c3d_anomaly_all.extend(original_c3d_anomaly)
        original_c3d_normal_all.extend(original_c3d_normal)
        event_proposal_c3d_anomaly_all.extend(event_proposal_c3d_anomaly)
        event_proposal_c3d_normal_all.extend(event_proposal_c3d_normal)

    return np.array(original_c3d_anomaly_all), np.array(original_c3d_normal_all), \
           np.array(event_proposal_c3d_anomaly_all), np.array(event_proposal_c3d_normal_all)


def collect_each_video(each_line, original_c3d_path, event_proposal_c3d_path, spatial_annotation_path, max_num=None):
    from random import choice
    print(each_line[0])
    video_name, video_class, start1, final1, start2, final2 = each_line[0].split('  ')
    start1, final1, start2, final2 = int(start1), int(final1), int(start2), int(final2)
    video_name, video_class = (video_name.replace('.mp4',''), video_class.replace('Normal','normal_test')) if '.mp4' in video_name else (video_name, video_class)

    original_c3d_npy_path = osp.join(original_c3d_path, video_class+"@"+video_name+".npy")
    if not osp.exists(original_c3d_npy_path):
        raise ValueError('No existing %s' % original_c3d_npy_path)
    original_c3d_npy = np.load(original_c3d_npy_path)
    np.random.shuffle(original_c3d_npy)
    original_c3d_npy = original_c3d_npy[:max_num]

    anomaly_index= [i[-8] for i in original_c3d_npy if start1 <= i[-8] <= final1 or start2 <= i[-8] <= final2]
    normal_index = [i[-8] for i in original_c3d_npy if not start1 <= i[-8] <=final1 and not start2 <= i[-8] <=final2]

    original_c3d_anomaly= [i[:4096] for i in original_c3d_npy if i[-8] in anomaly_index]
    original_c3d_normal = [i[:4096] for i in original_c3d_npy if i[-8] in normal_index]

    event_proposal_c3d_npy_path = osp.join(event_proposal_c3d_path, video_class+"@"+video_name+".npy")
    if not osp.exists(event_proposal_c3d_npy_path):
        raise ValueError('No existing %s' % event_proposal_c3d_npy_path)
    event_proposal_c3d_npy = np.load(event_proposal_c3d_npy_path)

    video_spatial_annotation_path = osp.join(spatial_annotation_path, video_name)
    # if not osp.exists(video_spatial_annotation_path):
    #     raise ValueError('No existing %s' % video_spatial_annotation_path)

    event_proposal_c3d_anomaly= [get_iou_one([j for j in event_proposal_c3d_npy if j[-12] == i], video_spatial_annotation_path) for i in anomaly_index]
    event_proposal_c3d_anomaly = [j for j in event_proposal_c3d_anomaly if j is not False]
    event_proposal_c3d_normal = [choice([j[:4096] for j in event_proposal_c3d_npy if j[-12] == i]) for i in normal_index]

    return original_c3d_anomaly, original_c3d_normal, event_proposal_c3d_anomaly, event_proposal_c3d_normal


def get_iou_one(c3d_list_in_one_frame, video_spatial_annotation_path):
    image_size, wei_shu = ((240, 360), 3) if 'ucsdped2' in video_spatial_annotation_path.lower() else ((240, 320),5)
    iou_list_in_one_frames, frame_index = [], c3d_list_in_one_frame[0][-12]
    spatial_annotation_txt = osp.join(video_spatial_annotation_path, str(int(frame_index)).zfill(wei_shu) + '.txt')
    spatial_annotations = [yoloLine2Shape(image_size, k[1], k[2], k[3], k[4]) for k in
                           basepy.read_txt_lines2list(spatial_annotation_txt, ' ')]
    for i in c3d_list_in_one_frame:
        event_proposal = (int(i[-10]), int(i[-9]), int(i[-10] + i[-8]), int(i[-9] + i[-7]))
        iou_list_in_one_frames.append(max([compute_iou(event_proposal, bx) for bx in spatial_annotations]))

    print(max(iou_list_in_one_frames))
    if max(iou_list_in_one_frames) > 0.1:
        select_one = iou_list_in_one_frames.index(max(iou_list_in_one_frames))
        # return (c3d_list_in_one_frame[select_one][:4096] + c3d_list_in_one_frame[select_one][4096:8192])/2
        return np.maximum(c3d_list_in_one_frame[select_one][:4096], c3d_list_in_one_frame[select_one][4096:8192])
    else:
        return False


def recall_iou_all(temporal_annotation_file, video_spatial_annotation_path, event_proposal_json_path):
    annotation_in_all = basepy.read_txt_lines2list(temporal_annotation_file)
    multi_all, single_all, frame_all = [], [], []
    for each_line in annotation_in_all:
        multi_region, one_region, iou_frame = recall_iou_video(each_line, video_spatial_annotation_path, event_proposal_json_path)
        multi_all.extend(multi_region)
        single_all.extend(one_region)
        frame_all.extend(iou_frame)

    return multi_all, single_all, frame_all


def recall_iou_video(each_line, video_spatial_annotation_path, event_proposal_json_path):
    print(each_line[0])
    video_name, video_class, start1, final1, start2, final2 = each_line[0].split('  ')
    start1, final1, start2, final2 = int(start1), int(final1), int(start2), int(final2)
    video_name, video_class, inflate = (video_name.replace('.mp4',''), video_class.replace('Normal','normal_test'), 16) if '.mp4' in video_name else (video_name, video_class, 8)

    index_ep_vs_annotation = [[i, int(i + inflate/2)] for i in list(range(start1, final1)) + list(range(start2, final2)) if i % inflate == 0]
    ep_json_file = osp.join(event_proposal_json_path, video_class + '@' + video_name + '.json')
    with open(ep_json_file, 'r') as f:
        info = json.load(f)
    annotation_path = osp.join(video_spatial_annotation_path, video_name)
    image_size, frame_region, wei_shu = ((240, 360), (0,0,360,240), 3) if 'ucsdped2' in video_spatial_annotation_path.lower() else ((240, 320), (0,0,320,240),5)
    multi_all, single_all, frame_all = [], [], []
    for ep_index, at_index in index_ep_vs_annotation:
        try:
            spatial_annotation_txt = osp.join(annotation_path, str(int(at_index)).zfill(wei_shu) + '.txt')
            spatial_annotations = [yoloLine2Shape(image_size, k[1], k[2], k[3], k[4]) for k in
                                   basepy.read_txt_lines2list(spatial_annotation_txt, ' ')]
        except FileNotFoundError:
            spatial_annotation_txt = osp.join(annotation_path, str(int(ep_index)).zfill(wei_shu) + '.txt')
            spatial_annotations = [yoloLine2Shape(image_size, k[1], k[2], k[3], k[4]) for k in
                                   basepy.read_txt_lines2list(spatial_annotation_txt, ' ')]

        ep_in_frame = [i for i in info if i[2] == ep_index]

        if ep_in_frame:
            ep_iou_on_at = [[compute_iou((int(i[-9]), int(i[-8]), int(i[-9] + i[-7]), int(i[-8] + i[-6])), j) for i in ep_in_frame] for j in spatial_annotations]
            multi_region = [max(k) for k in ep_iou_on_at]
            one_region = [s[[k[-1] for k in ep_in_frame].index(max([k[-1] for k in ep_in_frame]))] for s in ep_iou_on_at]
            iou_frame = [compute_iou(j, frame_region) for j in spatial_annotations]

            multi_all.extend(multi_region)
            single_all.extend(one_region)
            frame_all.extend(iou_frame)

    return multi_all, single_all, frame_all


if __name__ == '__main__':
    tf.app.run()
