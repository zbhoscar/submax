import data_io.basepy as basepy
import os.path as osp
import json

crime2local_videos = '/absolute/datasets/UCFCrime2Local/Videos_from_UCFCrime.txt'
datasets_path = '/absolute/datasets/anoma'
json_path = '/absolute/datasets/UCFCrime2Local_motion_all_json'

normal_list = [i[0] for i in basepy.read_txt_lines2list(crime2local_videos, sep='.') if 'normal' in i[0].lower()]

video_folder_list = basepy.get_2tier_folder_path_list(datasets_path)

normal_video_list = []
for normal_video_name in normal_list:
    normal_video_list.extend([i for i in video_folder_list if normal_video_name in i])

for one_video_path in normal_video_list:
    video_name = osp.basename(one_video_path)
    class_name = osp.basename(osp.dirname(one_video_path))

    tfrecord_name = '%s@%s' % (class_name, video_name)
    tfrecord_path = osp.join(json_path, tfrecord_name + '.json')

    frames_path = basepy.get_1tier_file_path_list(one_video_path, suffix='.jpg')
    frame_list = sorted(frames_path, key=lambda x: int(osp.basename(x).split('.')[0]))

    print('%s   has   %d   frames in all' % (tfrecord_name, frame_list.__len__()))

    nj, c, r, w, h, mean_value = 0, 0, 0, 320, 240, 0
    clips_info = [[class_name, video_name, j, nj, c, r, w, h, mean_value] for j, one_frame in enumerate(frame_list)][:-16]

    with open(tfrecord_path, 'w') as f:
        json.dump(clips_info, f)
