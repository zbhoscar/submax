import numpy as np
import json
import copy

var = np.random.rand(4096).tolist()

ditt = {'video1': {1: [var, var],
                   2: [var, var, var]},
        'video2': {3: [var, var, var, var],
                   4: [var, var, var, var, var]}}


def get_2tier_dict_list():
    video_segment_list = []
    for class_video_name in ditt.keys():
        for segment_index in ditt[class_video_name].keys():
            video_segment_list.append([class_video_name, segment_index])
    return video_segment_list
    # clip_average_feature = np.average(dict[class_video_name][segment_index], axis=0)
    # print(class_video_name, segment_index, dict[class_video_name][segment_index])


json_file = 'somefile.json'

with open(json_file, 'w') as f:
    json.dump(ditt, f)

with open(json_file, 'r') as f:
    dread = json.load(f)


def encode2dict(**kwargs):
    return update_keys(kwargs)


def decode2dict(kwargs):
    return update_keys(kwargs)


# Default keys
EXPERIMENT_KEYS = {'batch_size': 64,
                   'learning_rate_base': 0.001,
                   'new_key': False}


def update_keys(kwargs):
    temp = copy.deepcopy(EXPERIMENT_KEYS)
    for i in kwargs.keys():
        temp[i] = kwargs[i]
    return temp


def dict2json(d, json_path):
    with open(json_path, 'w') as f:
        json.dump(d, f)


def json2dict(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)
