import oz_flags
import data_io.tfr_high_motion_v1 as test
from pprint import pprint
import os.path as osp
import cv2
import numpy as np


def main():
    example = ['sf_UCF101frames_task_siam.3.cpr.2_snum_6_cstr_0.0_csec_1.0_ecod_dynamic_spsi_randomcrop_spty_backwards',
               'sf_UCF101frames_task_siam.6.rec_snum_6_cstr_0.0_csec_1.0_ecod_dynamic_spsi_randomcrop_spty_backwards',
               'sf_UCF-101.hm.20_task_siam.6.rec_snum_6_cstr_0.0_csec_1.0_ecod_dynamic_spsi_randomcrop_spty_backwards',
               'sf_UCF101frames_task_orig']

    abc = oz_flags.StrAna(inputdata=example[2])
    sample_list = [['ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01', '1']]

    tt = test.HighMotionTFR(data_root='/absolute/datasets', inputdata=example[2])
    pprint(tt.__dict__)

    tfrecords_list = tt.make_tfrecords_list(sample_list)
    print(tfrecords_list)

    source_cor_dpath = osp.join(tt.source_dpath, osp.basename(tfrecords_list[0]).replace('@', '/').split('.')[0])
    video_name = osp.basename(source_cor_dpath)
    class_name = osp.basename(osp.dirname(source_cor_dpath))

    video_pname = osp.join(tt.source_dpath, class_name, video_name + '.avi')

    cap, index, of_state = cv2.VideoCapture(video_pname), 0, []
    ret, frame = cap.read()
    prvsgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    # ret, frame = cap.read()
    # cv2.imshow('frame', frame)
    # cv2.waitKey(1000)

    while ret:
        ret, frame = cap.read()
        if ret is True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvsgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # (prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # cv2.imshow('ofrgb', rgb)
            # cv2.waitKey(int(400))
            #
            # print('frame.shape:', frame.shape)
            # print(index)

            jun_zhi = np.mean(flow)
            of_state.append(jun_zhi)

            prvsgray = gray
            index = index + 1

    cap.release()
    # cv2.destroyAllWindows()

    bb = [sum(of_state[i:i + 19]) for i in range(len(of_state) - 18)]
    win_start = bb.index(max(bb))
    # [ win_start+i for i in range(20)]

    print('wow')

if __name__ == '__main__':
    main()
