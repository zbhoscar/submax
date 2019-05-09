import data_io.basepy as basepy
import os.path as osp
import random

tfrecord_path = '/absolute/datasets/anoma_motion16_tfrecords'
dst_txt_path = '/absolute/ext3t/anoma_motion16_c3d_features'
divide_num = 3

tfrecord_path = basepy.get_1tier_file_path_list(tfrecord_path, suffix='.tfr')
already_path = basepy.get_1tier_file_path_list(dst_txt_path, suffix='.txt')
already_str = str(already_path)

remaining_tfr_list = [i for i in tfrecord_path if osp.basename(i).split('.')[0] not in already_str]
random.shuffle(remaining_tfr_list)

name_list_in_num = basepy.divide_list(remaining_tfr_list, divide_num)

_ = [[basepy.write_txt_add_lines('./temp/to_c3d_%d.txt' % index, line) for line in name_list]
     for index, name_list in enumerate(name_list_in_num)]

print(tfrecord_path.__len__())
print(already_path.__len__())
print(remaining_tfr_list.__len__())
_ = [print(i.__len__()) for i in name_list_in_num]
print('wow')
