import json
import data_io.basepy as basepy
import os.path as osp
import numpy as np
import zdefault_dict
from sklearn import metrics
import time
import tensorflow as tf


def main(_):
    tags = tf.flags
    F = tags.FLAGS
    tags.DEFINE_string('results_json_path',
                       '/absolute/tensorflow_models/191005201358_UCSDped2_reform_motion_reformed_single_120_85_4region_segment_32_c3d_npy',
                       'model folder path, or model ckpt file path:'
                       '/absolute/tensorflow_models/190918230353_anoma_motion_reformed_pyramid_120_85_1region_maxtop_1000_c3d_npy/190918230353.ckpt-9619_eval_json'
                       '/absolute/tensorflow_models/190918230353_anoma_motion_reformed_pyramid_120_85_1region_maxtop_1000_c3d_npy')
    tags.DEFINE_string('save_plot', "", './temp/test_savefig'
                                        'where to save figs, default: "./temp/test_savefig", "" for NO FIG SAVE.')

    results_evaluate(F.results_json_path, F.save_plot)


def results_evaluate(results_json_path, save_plot):
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
        fpr, tpr, thresholds, auc, all_videos_map, vfpr, vtpr, vthresholds, vauc, videos_order, \
        fpr_s, tpr_s, thresholds_s, auc_s, vfpr_s, vtpr_s, vthresholds_s, vauc_s = \
            analysis_in_one_ckpt( results_json_path, temporal_annotation_file, inflate, save_plot)
    else:
        _eval_json_list = basepy.get_1tier_file_path_list(results_json_path, suffix='_eval_json')
        _eval_json_list = sorted(_eval_json_list, key=lambda x: int(x.split('_eval_json')[0].split('ckpt-')[1]))
        results_all_in_one = [analysis_in_one_ckpt(i, temporal_annotation_file, inflate, save_plot)
                              for i in _eval_json_list]
        _ = [print(i[3], i[8], i[13], i[17], osp.basename(_eval_json_list[j]))
             for j, i in enumerate(results_all_in_one)]
        print('RESULTS_JSON_PATH'.ljust(25), results_json_path)

    print('ANALYSIS DONE ------ Debug Symbol ------ %s ------' % time.asctime(time.localtime(time.time())))


def analysis_in_one_ckpt(results_json_path, temporal_annotation_file, inflate, save_plot):
    # video_name, deflated_length, temporal_score, temporal_truth
    video_temporal_results = get_temporal_duration_in_folder(
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

    return fpr, tpr, thresholds, auc, all_videos_map, vfpr, vtpr, vthresholds, vauc, videos_order, \
           fpr_s, tpr_s, thresholds_s, auc_s, vfpr_s, vtpr_s, vthresholds_s, vauc_s


def get_temporal_duration_in_folder(results_json_path, temporal_annotation_file, inflate,
                                    suffix='.json', score_and_truth_fig_save_path=None):
    json_file_list = basepy.get_1tier_file_path_list(results_json_path, suffix=suffix)
    videos_temporal_results = []
    if score_and_truth_fig_save_path:
        print('drawing to %s...' % score_and_truth_fig_save_path)

    for json_file in json_file_list:
        video_name, deflated_length, temporal_score, temporal_score_select, temporal_truth = \
            get_temporal_duration(json_file, inflate, temporal_annotation_file)
        if score_and_truth_fig_save_path:
            show_something(score_and_truth_fig_save_path, deflated_length, inflate,
                           temporal_score, temporal_truth, video_name)
        videos_temporal_results.append([video_name, deflated_length,
                                        temporal_score, temporal_score_select, temporal_truth])
    return videos_temporal_results


def show_something(score_and_truth_fig_save_path, deflated_length, inflate,
                   temporal_score,temporal_truth, video_name):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    _ = basepy.check_or_create_path(score_and_truth_fig_save_path)
    x = np.linspace(0, deflated_length * inflate, deflated_length)
    y1 = np.array(temporal_score)
    y2 = np.array(temporal_truth)
    plt.figure()
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.ylim(0, 1.1)
    plt.xlim(0, deflated_length * inflate + 4)
    plt.title(video_name)
    plt.xlabel('Frame number')
    plt.ylabel('Anomaly score')
    plt.savefig(osp.join(score_and_truth_fig_save_path, video_name) + '.png')



def get_temporal_duration(json_file, inflate, temporal_annotation_file):
    video_name = json_file.split('@')[-1].split('.')[0]
    # print('getting temporal duration: %s' % video_name)
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

    return video_name, len(temporal_score), temporal_score, temporal_score_select, temporal_truth


if __name__ == '__main__':
    tf.app.run()
