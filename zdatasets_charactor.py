import cv2

def get_charactor(dataset_path):
    if 'ucsd' in dataset_path.lower():
        return dataset_path, '.tif', (240, 360), 16, 8, 2, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1), 3
    elif 'anoma' in dataset_path.lower():
        return dataset_path, '.jpg', (240, 320), 16, 16, 2, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1), 5
    else:
        raise ValueError('Analysis failure for dataset path: %s' % dataset_path)
    # DATASET_PATH, FRAME_SUFFIX, FRAME_SIZE, CLIP_LEN, STEP, OPTICAL, CRITERIA, zfill_num


def get_dataset_path(json_path):
    if 'ucsd' in json_path.lower():
        return '/absolute/datasets/UCSDped2_reform'
    elif 'anoma' in json_path.lower():
        return '/absolute/datasets/anoma'
    else:
        raise ValueError('Analysis failure for json path: %s' % json_path)
    # DATASET_PATH, FRAME_SUFFIX, FRAME_SIZE, CLIP_LEN, STEP, OPTICAL, CRITERIA
