import data_io.basepy as basepy
import os.path as osp
import random

TFRECORD_PATH = '/absolute/datasets/anoma_motion_tfr_type_1_simple_1001'
DST_TXT_PATH = TFRECORD_PATH.replace('tfr', 'c3d_features')
LIST_PATH = './temp/tfr2c3d/to_c3d_%d.txt'
DIVIDE_NUM = 10


def make_multi_tfr_list(todo_file_list, done_file_list, list_txt_path=None, divide_num=1, if_print=False):
    already_str = str(done_file_list)
    ###
    #     Basename in tfrecord_path SAME == in already_path,
    #           or in tfrecord_path PART OF in already_path.
    ###
    remaining_tfr_list = [i for i in todo_file_list if osp.basename(i).split('.')[0] not in already_str]
    random.shuffle(remaining_tfr_list)

    name_list_in_num = basepy.divide_list(remaining_tfr_list, divide_num)

    if list_txt_path:
        _ = [[basepy.write_txt_add_lines(list_txt_path % index, line) for line in name_list]
             for index, name_list in enumerate(name_list_in_num)]

    if if_print:
        print('Files: %d in all, %d is done, %d remaining' %
              (len(todo_file_list), len(done_file_list), len(remaining_tfr_list)), '...')
        if list_txt_path:
            print('Split list in %d .txt, writen in %s' % (divide_num, list_txt_path))
        else:
            print('Split list in %d .txt, no writing.' % divide_num)

    return remaining_tfr_list, name_list_in_num


if __name__ == '__main__':
    make_multi_tfr_list(todo_file_list=basepy.get_1tier_file_path_list(TFRECORD_PATH, suffix='.tfr'),
                        done_file_list=basepy.get_1tier_file_path_list(
                            basepy.check_or_create_path(DST_TXT_PATH), suffix='.npy'),
                        list_txt_path=LIST_PATH, divide_num=DIVIDE_NUM, if_print=True)
