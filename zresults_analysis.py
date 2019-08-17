import json
import data_io.basepy as basepy
import os.path as osp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from matplotlib.pyplot import plot, savefig
import numpy as np

RESULTS_JSON_PATH, TEMPORAL_ANNOTATION_FILE, DATASET_PATH, SUFFIX, INFLATE = \
    (('/absolute/tensorflow_models/190601162431/190601162431.ckpt-1266_anoma_json',
      '/absolute/datasets/Anomaly-Detection-Dataset/Temporal_Anomaly_Annotation_for_Testing_Videos.txt',
      '/absolute/datasets/anoma', '.jpg', 16),
     ('/absolute/tensorflow_models/190601162431',
      '/absolute/datasets/Anomaly-Detection-Dataset/Temporal_Anomaly_Annotation_for_Testing_Videos.txt',
      '/absolute/datasets/anoma', '.jpg', 16),
     ('/absolute/tensorflow_models/190813231653/190813231653.ckpt-52_anoma_json',
      '/absolute/datasets/UCSDped2_split_list/temproal_annotation.txt',
      '/absolute/datasets/UCSDped2_reform', '.tif', 8),
     )[2]


def get_temporal_duration(json_file,
                          dataset_path=DATASET_PATH,
                          temporal_annotation_file=TEMPORAL_ANNOTATION_FILE):
    # json_file = '/absolute/tensorflow_models/190813140440/190813140440.ckpt-49_anoma_json/Anomaly@Anomaly008.json'
    video_name = json_file.split('@')[-1].split('.')[0]
    print('%s' % video_name)
    video_list = basepy.get_2tier_folder_path_list(dataset_path)
    video_path = [i for i in video_list if video_name in i]
    if video_path.__len__() != 1:
        raise ValueError('Too many %s in %s' % (video_name, video_path))
    else:
        video_path = video_path[0]

    video_frames = basepy.get_1tier_file_path_list(video_path, suffix=SUFFIX)
    video_frames = sorted(video_frames, key=lambda x: int(osp.basename(x).split('.')[0]))
    # need a check
    video_length = video_frames.__len__()

    annotation_in_all = basepy.read_txt_lines2list(temporal_annotation_file)
    video_in_annotation = [i[0] for i in annotation_in_all if video_name in i[0]]
    if video_in_annotation.__len__() != 1:
        raise ValueError('Too many %s in %s' % (video_name, video_in_annotation))
    else:
        video_in_annotation = video_in_annotation[0]
    video_in_annotation = video_in_annotation.split('  ')
    temporal_annotation = [0 for _ in range(video_length)]
    for i in [2, 4]:
        start = int(video_in_annotation[i])
        final = int(video_in_annotation[i + 1])
        for j in range(start, final):
            try:
                temporal_annotation[j] = 1
            except:
                print('index %d out of range %s, %d' % (j, video_name, video_length))

    with open(json_file, 'r') as f:
        info = json.load(f)

    # select info by INFLATE
    info = [i for i in info if i[2] % 8 == 0]
    info_max_frame_index = int(max([i[2] for i in info]) / INFLATE + 1)
    temporal_prediction = [0 for _ in range(info_max_frame_index)]
    for area_tube in info:
        frame_index, anomaly_score = int(area_tube[2] / INFLATE), area_tube[-1]
        temporal_prediction[frame_index] = max(temporal_prediction[frame_index], anomaly_score)

    temporal_annotation = [i for j, i in enumerate(temporal_annotation) if j % INFLATE == 0][:len(temporal_prediction)]

    temporal_prediction=[min(0.998,score)+ 0.0019*np.random.rand(1)[0] for score in temporal_prediction]
    temporal_prediction=[max(0.00001,score) - + 0.0019*np.random.rand(1)[0] for score in temporal_prediction]

    return video_name, video_length, temporal_prediction, temporal_annotation


def get_temporal_duration_in_folder(results_json_path=RESULTS_JSON_PATH,
                                    dataset_path=DATASET_PATH,
                                    suffix='.json',
                                    save_plot=None):
    json_file_list = basepy.get_1tier_file_path_list(results_json_path, suffix=suffix)
    video_temporal_results = []
    if save_plot:
        print('%s...' % save_plot)

    for json_file in json_file_list:
        video_name, video_length, temporal_prediction, temporal_annotation = \
            get_temporal_duration(json_file, dataset_path=dataset_path)

        comparision_points = len(temporal_prediction)

        if save_plot:
            _ = basepy.check_or_create_path(save_plot)
            # temporal_annotation[-1] = 0
            # temporal_annotation[0] = 0
            x = np.linspace(0, comparision_points * INFLATE, comparision_points)
            y = np.array(temporal_prediction)
            t = np.array(temporal_annotation)
            plt.figure()
            plt.plot(x, y)
            plt.plot(x, t)
            plt.ylim(0, 1.1)
            plt.xlim(0, comparision_points * INFLATE + 5)
            plt.title(video_name)
            plt.xlabel('Frame number')
            plt.ylabel('Anomaly score')
            plt.savefig(osp.join(save_plot, video_name) + '.png')

        video_temporal_results.append([video_name, video_length, temporal_prediction, temporal_annotation])

    return video_temporal_results


def main():
    print('------ Debug This To Choose Different Evaluation ------')
    # only fig save
    if '_json' == RESULTS_JSON_PATH[-5:]:
        _ = get_temporal_duration_in_folder(results_json_path=RESULTS_JSON_PATH,
                                            dataset_path=DATASET_PATH,
                                            suffix='.json',
                                            save_plot=osp.join('./temp/test_savefig', osp.basename(RESULTS_JSON_PATH)))
    else:
        _anoma_json_list = basepy.get_1tier_file_path_list(osp.dirname(RESULTS_JSON_PATH), suffix='_json')
        for i in _anoma_json_list:
            _ = get_temporal_duration_in_folder(results_json_path=i,
                                                dataset_path=DATASET_PATH,
                                                suffix='.json',
                                                save_plot=osp.join('./temp/test_savefig', osp.basename(i)))

    # only to evaluation, single RESULTS_JSON_PATH
    video_temporal_results = get_temporal_duration_in_folder(results_json_path=RESULTS_JSON_PATH,
                                                             dataset_path=DATASET_PATH,
                                                             suffix='.json')

    label_test, label_keys = [], []
    for i in video_temporal_results:
        [video_name, video_length, temporal_prediction, temporal_annotation] = i
        label_test.extend(temporal_prediction)
        label_keys.extend(temporal_annotation)

    t, f, auc, auc_, precision_list = basepy.TPR_FPR(label_test, label_keys, bool_draw=True, sample_num=2500)
    if True:
        print('%s evaluate results:' % RESULTS_JSON_PATH)
        print('auc: %5f, auc_: %5f, max precision: %5f,' % (auc, auc_, max(precision_list)))
        print('TPR_points:')
        print(t)
        print('FPR_points:')
        print(f)
        print('corresponding precision:')
        print(precision_list)


if __name__ == '__main__':
    main()
