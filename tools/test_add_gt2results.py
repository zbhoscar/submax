import data_io.basepy as basepy
from zdatasets_charactor import *
from zresults_analysis import *
import os.path as osp
import cv2

RESULTS_DRAW = [
    '/home/zbh/PycharmProjects/anoma_v0.8/temp/test_savefig/191017225848_UCSDped2_reform_motion_reformed_pyramid_80_56_4region_segment_32_c3d_npy/191017225848.ckpt-47_eval_json/Anomaly012',
    '/home/zbh/PycharmProjects/anoma_v0.8/temp/test_savefig/191007162304_UCSDped2_reform_motion_reformed_pyramid_80_56_4region_segment_32_c3d_npy/191007162304.ckpt-55_eval_json/Normal006',
    '/home/zbh/PycharmProjects/anoma_v0.8/temp/test_savefig/191007162304_UCSDped2_reform_motion_reformed_pyramid_80_56_4region_segment_32_c3d_npy/191007162304.ckpt-55_eval_json/Anomaly010',
    '/home/zbh/PycharmProjects/anoma_v0.8/temp/test_savefig/191007162304_UCSDped2_reform_motion_reformed_pyramid_80_56_4region_segment_32_c3d_npy/191007162304.ckpt-55_eval_json/Anomaly002',
    '/home/zbh/PycharmProjects/anoma_v0.8/temp/test_savefig/191007162304_UCSDped2_reform_motion_reformed_pyramid_80_56_4region_segment_32_c3d_npy/191007162304.ckpt-1_eval_json/Anomaly010',
    '/home/zbh/PycharmProjects/anoma_v0.8/temp/test_savefig/191007162304_UCSDped2_reform_motion_reformed_pyramid_80_56_4region_segment_32_c3d_npy/191007162304.ckpt-1_eval_json/Anomaly002'
                ][0]
GT_FOLDER_PATH = '/absolute/datasets/UCSDped2_spatial_annotation'
# GT_FOLDER_PATH = '/absolute/datasets/anoma_spatial_annotations'
SAVE_DIR_PATH = '/home/zbh/PycharmProjects/SupplementaryMaterials'


def main(results_draw=RESULTS_DRAW, gt_folder_path=GT_FOLDER_PATH, save_dir_path=SAVE_DIR_PATH):
    to_draw_list = basepy.get_1tier_file_path_list(results_draw, suffix=get_charactor(results_draw)[1])
    to_draw_list = sorted(to_draw_list, key=lambda x: int(osp.basename(x).split('.')[0]))
    video_gt_path = osp.join(gt_folder_path, osp.basename(results_draw))
    save_img_path = basepy.check_or_create_path(osp.join(save_dir_path, osp.basename(results_draw)))
    image_size = get_charactor(results_draw)[2]
    for frame in to_draw_list:
        img = cv2.imread(frame)
        img_gt_file = osp.join(video_gt_path, osp.basename(frame).split('.')[0]+'.txt')
        if osp.exists(img_gt_file):
            spatial_annotations = [yoloLine2Shape(image_size, k[1], k[2], k[3], k[4]) for k in
                                   basepy.read_txt_lines2list(img_gt_file, ' ')]
            for y0, x0, y1, x1 in spatial_annotations:
                cv2.rectangle(img, (y0, x0), (y1, x1), (53, 134, 238), 1)

        save_frame = osp.join(save_img_path, osp.basename(frame))
        cv2.imwrite(save_frame, img)


if __name__ == '__main__':
    main()