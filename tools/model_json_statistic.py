import data_io.basepy as basepy
import json
import time
import os.path as osp
from pprint import pprint


TF_MODEL_PATH = '/absolute/tensorflow_models'
WRITE2TXT = '../temp/results.txt'


def write_json_in_txt(json_file_path, if_pprint=False, write2txt=None):
    model_id = osp.basename(osp.dirname(json_file_path))
    with open(json_file_path, 'r') as f:
        model_info = json.load(f)
    if if_pprint:
        pprint(model_info)
    if write2txt:
        str2write = str(model_info)[1:-1].replace("'",'')
        basepy.write_txt_add_lines(write2txt, model_id, str2write)
    return model_info


def main():
    # find json file in TensorFlow models path
    model_json_list = basepy.get_2tier_folder_path_list(TF_MODEL_PATH, suffix_in_2tier='.json')
    _ = [write_json_in_txt(i, if_pprint=False, write2txt=WRITE2TXT) for i in model_json_list]

    print('------ Finish ------ Debug Symbol ------ %s ------' % time.asctime(time.localtime(time.time())))


if __name__ == '__main__':
    main()
