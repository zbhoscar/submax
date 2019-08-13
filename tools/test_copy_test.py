import shutil
import data_io.basepy as basepy

test_list_txt = '/absolute/datasets/Anomaly-Detection-Dataset/Temporal_Anomaly_Annotation_for_Testing_Videos.txt'
data_set_path = '/absolute/datasets/anoma'
to_test_path = '/absolute/datasets/anoma_all_test'

video_folder_list = basepy.get_2tier_folder_path_list(data_set_path)

test_video_list = basepy.read_txt_lines2list(test_list_txt, sep='  ')
# test_video_list = [i[0].split('.')[0] for i in test_video_list]

frame_in_all = 0
anoma_in_all = 0

for one_line_test_video in test_video_list:
    a_test_video, class_name, an_start_1, an_end_1, an_start_2, an_end_2 = one_line_test_video
    a_test_video = a_test_video.split('.')[0]
    # print(a_test_video, an_start_1, an_end_1, an_start_2, an_end_2)
    a_test_video_folder_path_list = [j for j in video_folder_list if a_test_video in j]
    if a_test_video_folder_path_list.__len__() != 1:
        raise ValueError('NOT only %s in %s' % (a_test_video, a_test_video_folder_path_list))
    a_test_video_folder_path = a_test_video_folder_path_list[0]

    frame_in_all = frame_in_all + basepy.get_1tier_file_path_list(a_test_video_folder_path).__len__()
    anoma_in_all = anoma_in_all + int(an_end_1) - int(an_start_1) + int(an_end_2) - int(an_start_2)

    # shutil.copytree(a_test_video_folder_path, a_test_video_folder_path.replace(data_set_path, to_test_path))

print((frame_in_all-anoma_in_all)/frame_in_all)

print('DEBUG')
