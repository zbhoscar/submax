import json
import data_io.basepy as basepy
import os.path as osp
import numpy as np
import zdefault_dict
from sklearn import metrics
import time
import tensorflow as tf
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


def main(_):
    tags = tf.flags
    F = tags.FLAGS
    tags.DEFINE_string('results_json_path',
                       '/absolute/tensorflow_models/191007174553_anoma_motion_reformed_single_120_85_1region_maxtop_256_c3d_npy/191007174553.ckpt-12404_eval_json',
                       'model folder path, or model ckpt file path:'
                       '/absolute/tensorflow_models/190918230353_anoma_motion_reformed_pyramid_120_85_1region_maxtop_1000_c3d_npy/190918230353.ckpt-9619_eval_json'
                       '/absolute/tensorflow_models/190918230353_anoma_motion_reformed_pyramid_120_85_1region_maxtop_1000_c3d_npy')
    tags.DEFINE_string('save_plot', './temp/test_savefig', 'where to save figs, default: "./temp/test_savefig", "" for NO FIG SAVE.')
    tags.DEFINE_string('spatial_annotation_path', '/absolute/datasets/anoma_spatial_annotations', 'spatial annotation')

    results_evaluate(F.results_json_path, F.save_plot, F.spatial_annotation_path)


def results_evaluate(results_json_path, save_plot, spatial_annotation_path):
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
        # fpr, tpr, thresholds, auc, all_videos_map, vfpr, vtpr, vthresholds, vauc, videos_order, \
        # fpr_s, tpr_s, thresholds_s, auc_s, vfpr_s, vtpr_s, vthresholds_s, vauc_s = results_of_one
        print(results_of_one[3], results_of_one[8], results_of_one[13], results_of_one[17], results_json_path)
    else:
        _eval_json_list = basepy.get_1tier_file_path_list(results_json_path, suffix='_eval_json')
        _eval_json_list = sorted(_eval_json_list, key=lambda x: int(x.split('_eval_json')[0].split('ckpt-')[1]))
        results_all_in_one = [
            analysis_in_one_ckpt(i, temporal_annotation_file, inflate, save_plot, spatial_annotation_path)
            for i in _eval_json_list]
        _ = [print(i[3], i[8], i[13], i[17], osp.basename(_eval_json_list[j]))
             for j, i in enumerate(results_all_in_one)]
        print('RESULTS_JSON_PATH'.ljust(25), results_json_path)

    print('ANALYSIS DONE ------ Debug Symbol ------ %s ------' % time.asctime(time.localtime(time.time())))


def analysis_in_one_ckpt(results_json_path, temporal_annotation_file, inflate, save_plot, annotation_folder_path):
    # video_name, deflated_length, temporal_score, temporal_truth
    video_temporal_results, info_smooth_all_in_one = get_temporal_duration_in_folder(
        results_json_path, temporal_annotation_file, inflate, suffix='.json',
        score_and_truth_fig_save_path=osp.join(save_plot, osp.basename(results_json_path)) if save_plot else None)
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
        get_spatial_pr_curve(info_smooth_all_in_one, annotation_folder_path, temporal_annotation_file, inflate)

    if save_plot:
        save_one_fig(fpr, tpr, 'AUC = %f', 'False Positive Rate', 'True Positive Rate',
                     osp.join(save_plot, osp.basename(results_json_path)).split('_eval_json')[0] + '_orig.png',
                     for_title=auc)
        save_one_fig(vfpr, vtpr, 'AUC = %f', 'False Positive Rate', 'True Positive Rate',
                     osp.join(save_plot, osp.basename(results_json_path)).split('_eval_json')[0] + '_video.png',
                     for_title=vauc)
        save_one_fig(fpr_s, tpr_s, 'AUC = %f', 'False Positive Rate', 'True Positive Rate',
                     osp.join(save_plot, osp.basename(results_json_path)).split('_eval_json')[0] + '_smooth.png',
                     for_title=auc_s)
        save_one_fig(vfpr_s, vtpr_s, 'AUC = %f', 'False Positive Rate', 'True Positive Rate',
                     osp.join(save_plot, osp.basename(results_json_path)).split('_eval_json')[0] + '_video_smooth.png',
                     for_title=vauc_s)
        save_one_fig(spatial_r, spatial_p, 'RECOVER = %f', 'Recall', 'Precision',
                     osp.join(save_plot, osp.basename(results_json_path)).split('_eval_json')[0] + '_pr_curve.png',
                     for_title=recover_rate)

    return fpr, tpr, thresholds, auc, all_videos_map, vfpr, vtpr, vthresholds, vauc, videos_order, \
           fpr_s, tpr_s, thresholds_s, auc_s, vfpr_s, vtpr_s, vthresholds_s, vauc_s, \
           spatial_p, spatial_r, spatial_threholds, recover_rate


def save_one_fig(x, y, title_str, xlabel_str, ylabel_str, save_file_path, for_title=''):
    # 'RECOVER%s' % '' == 'RECOVER'
    plt.plot(x, y)
    plt.title(title_str % for_title)
    plt.xlabel(xlabel_str)
    plt.ylabel(ylabel_str)
    plt.savefig(save_file_path)
    plt.close()


def get_spatial_pr_curve(results_all_in_one, annotation_folder_path, temporal_annotation_file, inflate, iou_threshold=0.10):
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
                spatial_annotation_txt = osp.join(annotation_folder_path, video_name_temp, str(j).zfill(wei_shu) + '.txt')
                if not osp.exists(spatial_annotation_txt):
                    raise ValueError('Not Exists: annotation txt path %s' % spatial_annotation_txt)
                in_one_video[j] = [[yoloLine2Shape(image_size, k[1], k[2], k[3], k[4]), 0] for k in
                                   basepy.read_txt_lines2list(spatial_annotation_txt, ' ')]
                all_annotation_num = all_annotation_num + len(in_one_video[j])
        spatial_annotation[video_name_temp] = in_one_video

    spatial_groud_truth, covered_num = get_spatial_groud_truth(results_all_in_one, spatial_annotation, iou_threshold=iou_threshold)
    spatial_anomaly_score = [i[-1] for i in results_all_in_one]
    p, r, thresholds = metrics.precision_recall_curve(spatial_groud_truth, spatial_anomaly_score)

    return p, r, thresholds, covered_num/all_annotation_num


def get_spatial_groud_truth(results_all_in_one, spatial_annotation, scale_id=2, iou_threshold=0.1):
    spatial_groud_truth = [0] * len(results_all_in_one)
    for j, [_, video_name, frame_index, _, a1_c, a1_r, a1_w, a1_h, a2_c, a2_r, a2_w, a2_h, _, _, _] in enumerate(results_all_in_one):
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
    info_smooth_all_in_one = []
    if score_and_truth_fig_save_path:
        print('drawing to %s...' % score_and_truth_fig_save_path)

    for json_file in json_file_list:
        video_name, deflated_length, temporal_score, temporal_score_select, temporal_truth, info_smooth = \
            get_temporal_duration(json_file, inflate, temporal_annotation_file)
        if score_and_truth_fig_save_path:
            show_something(score_and_truth_fig_save_path, deflated_length, inflate,
                           temporal_score_select, temporal_truth, video_name)
        videos_temporal_results.append([video_name, deflated_length,
                                        temporal_score, temporal_score_select, temporal_truth])
        info_smooth_all_in_one.extend(info_smooth)
    return videos_temporal_results, info_smooth_all_in_one


def convert_list2dict(results_list, select=2):
    new_dict = {}
    for _, video_name, frame_index, _, a1_c, a1_r, a1_w, a1_h, a2_c, a2_r, a2_w, a2_h, _, _, _ in results_list:
        if select == 2:
            area = (a2_c, a2_r, a2_c+a2_w, a2_r+a2_h)
        elif select == 1:
            area = (a1_c, a1_r, a1_c+a1_w, a1_r+a1_h)
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
    temporal_score_select_smooth = list(savgol_filter(temporal_score_select, min(len(temporal_score_select), 11), 0))
    info_smooth = info
    for j, one_clip in enumerate(info_smooth):
        index = int(one_clip[2] // inflate)
        if one_clip[-1] == 0:
            one_clip[-1] = min(temporal_score_select_smooth[index] / 100, 0.995)
        elif temporal_score_select[index] == 0:
            one_clip[-1] = min(temporal_score_select_smooth[index], 0.995)
        else:
            # if temporal_score_select[index] == 0 :
            #     print(json_file, index)
            one_clip[-1] = min((one_clip[-1] / temporal_score_select[index]) * temporal_score_select_smooth[index], 0.995)

    return video_name, len(temporal_score), temporal_score, temporal_score_select_smooth, temporal_truth, info_smooth


def yoloLine2Shape(image_size, xcen, ycen, w, h):
    xcen, ycen, w, h = float(xcen), float(ycen), float(w), float(h)
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


if __name__ == '__main__':
    tf.app.run()
