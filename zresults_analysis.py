import json
import data_io.basepy as basepy
import os.path as osp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import metrics

matplotlib.use('Agg')

tags = tf.flags
F = tags.FLAGS
tags.DEFINE_string('results_json_path',
                   '/absolute/tensorflow_models/190915203428_anoma_motion_4training_original_c3d_npy',
                   'model folder path, or model ckpt file path:'
                   '/absolute/tensorflow_models/190912162832_anoma_motion_4training_pyramid_80_56_4region_c3d_npy/190914232215.ckpt-9619_eval_json'
                   '/absolute/tensorflow_models/190912162832_anoma_motion_4training_pyramid_80_56_4region_c3d_npy')
tags.DEFINE_string('testing_list',
                   '/absolute/datasets/Anomaly-Detection-Dataset/Temporal_Anomaly_Annotation_for_Testing_Videos.txt',
                   'test samples from the list, default for UCFCrime,'
                   'default is /absolute/datasets/Anomaly-Detection-Dataset/Temporal_Anomaly_Annotation_for_Testing_Videos.txt'
                   'else for USCD: e.g. /absolute/datasets/UCSDped2_split_list/10_fold_001/v01_test.txt')
tags.DEFINE_string('save_plot',
                   '',
                   'where to save figs, default: "./temp/test_savefig", "" for NO FIG SAVE.')
RESULTS_JSON_PATH, TEMPORAL_ANNOTATION_FILE = F.results_json_path, F.testing_list
SUFFIX = '.jpg' if 'anoma' in RESULTS_JSON_PATH else '.tif'
INFLATE = 16 if 'anoma' in RESULTS_JSON_PATH else 8

print('RESULTS_JSON_PATH'.ljust(25), RESULTS_JSON_PATH)
print('TEMPORAL_ANNOTATION_FILE'.ljust(25), TEMPORAL_ANNOTATION_FILE)
print('SUFFIX'.ljust(25), SUFFIX)
print('INFLATE'.ljust(25), INFLATE)


def main():
    print('------ Debug This To Choose Different Evaluation ------')

    if '_eval_json' in RESULTS_JSON_PATH[-10:]:
        fpr, tpr, thresholds, auc, all_videos_map = analysis_in_one_ckpt(RESULTS_JSON_PATH)
    else:
        _eval_json_list = basepy.get_1tier_file_path_list(RESULTS_JSON_PATH, suffix='_eval_json')
        _eval_json_list = sorted(_eval_json_list, key=lambda x: int(x.split('_eval_json')[0].split('ckpt-')[1]))
        results_all_in_one = [analysis_in_one_ckpt(i) for i in _eval_json_list]
        _ = [print(i[3], osp.basename(_eval_json_list[j])) for j, i in enumerate(results_all_in_one)]
        print('RESULTS_JSON_PATH'.ljust(25), RESULTS_JSON_PATH)

    print('------ Debug This To Choose Different Evaluation ------')


def analysis_in_one_ckpt(results_json_path):
    save_plot = osp.join(F.save_plot, osp.basename(results_json_path)) if F.save_plot else None
    # video_name, deflated_length, temporal_score, temporal_truth
    video_temporal_results = get_temporal_duration_in_folder(results_json_path=results_json_path,
                                                             suffix='.json',
                                                             score_and_truth_fig_save_path=save_plot)
    all_frames_score, all_frames_truth, all_videos_map = [], [], []
    for video_name, deflated_length, temporal_score, temporal_truth in video_temporal_results:
        all_frames_score.extend(temporal_score)
        all_frames_truth.extend(temporal_truth)
        all_videos_map.append([video_name] * deflated_length)

    # t, f, auc, auc_, precision_list = basepy.TPR_FPR_calc(all_frames_score, all_frames_truth, bool_draw=bool_draw)
    # return t, f, auc, auc_, precision_list, osp.basename(results_json_path), all_videos_map
    fpr, tpr, thresholds = metrics.roc_curve(all_frames_truth, all_frames_score, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return fpr, tpr, thresholds, auc, all_videos_map


def get_temporal_duration_in_folder(results_json_path=RESULTS_JSON_PATH,
                                    suffix='.json',
                                    score_and_truth_fig_save_path=None):
    json_file_list = basepy.get_1tier_file_path_list(results_json_path, suffix=suffix)
    videos_temporal_results = []
    if score_and_truth_fig_save_path:
        print('drawing to %s...' % score_and_truth_fig_save_path)

    for json_file in json_file_list:
        video_name, deflated_length, temporal_score, temporal_truth = get_temporal_duration(json_file)
        if score_and_truth_fig_save_path:
            _ = basepy.check_or_create_path(score_and_truth_fig_save_path)
            x = np.linspace(0, deflated_length * INFLATE, deflated_length)
            y1 = np.array(temporal_score)
            y2 = np.array(temporal_truth)
            plt.figure()
            plt.plot(x, y1)
            plt.plot(x, y2)
            plt.ylim(0, 1.1)
            plt.xlim(0, deflated_length * INFLATE + 4)
            plt.title(video_name)
            plt.xlabel('Frame number')
            plt.ylabel('Anomaly score')
            plt.savefig(osp.join(score_and_truth_fig_save_path, video_name) + '.png')
        videos_temporal_results.append([video_name, deflated_length, temporal_score, temporal_truth])
    return videos_temporal_results


def get_temporal_duration(json_file, temporal_annotation_file=TEMPORAL_ANNOTATION_FILE):
    video_name = json_file.split('@')[-1].split('.')[0]
    print('getting temporal duration: %s' % video_name)
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

    temporal_truth = [i for j, i in enumerate(frame_annotation) if j % INFLATE == 0]
    temporal_score = [-1] * len(temporal_truth)
    for _, _, frame_index, _, _, _, _, _, _, _, anomaly_score in info:
        if frame_index % INFLATE == 0:
            index_deflated = int(frame_index // INFLATE)
            temporal_score[index_deflated] = max(temporal_score[index_deflated], anomaly_score)

    if -1 in temporal_score:
        raise ValueError('Missing index in %s' % json_file)
    if len(temporal_score) != len(temporal_truth):
        raise ValueError('temporal_score and temporal_truth do not match in number in %s' % json_file)

    return video_name, len(temporal_score), temporal_score, temporal_truth


if __name__ == '__main__':
    main()
