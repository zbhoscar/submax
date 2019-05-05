from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import os.path as osp

from nets import nets_map
import data_io.basepy as baseio
import oz_flags
import data_io.feed_semi_lr_v1 as semialphaio
import data_io.tfr_frames_v1 as tfrecordsio

F = oz_flags.tags.FLAGS


class SampleList(object):
    def __init__(self,
                 source_folder='UCF101frames',
                 data_root='/abs',
                 split='orig.1',
                 is_training=True):
        self.source_folder = source_folder
        self.data_root = data_root
        self.split = split
        self.is_training = is_training

    def get_sample_list(self):
        if 'orig' in self.split:
            index = int(self.split.split('.')[1])
            if 'UCF101' in self.source_folder:
                split_dpath = osp.join(self.data_root, 'ucfTrainTestlist')
                class_list = [k.strip().split() for k in baseio.read_txt_lines2list(osp.join(split_dpath, 'classInd.txt'))]
                split_fname = 'trainlist0%d.txt' % index if self.is_training else 'testlist0%d.txt' % index
                sample_list = baseio.read_txt_lines2list(osp.join(split_dpath, split_fname))
                sample_list = [k.strip().replace('.avi', '').split() for k in sample_list]
                # class_list MUST be in the 1 -> 101 order
                if [int(i[0]) for i in class_list] != [i + 1 for i in range(len(class_list))]:
                    raise ValueError('class_list not in a appropriate order')
                # for test.split in UCF101 do not have label
                if not self.is_training:
                    _ = [j.append(class_list[[k[1] for k in class_list].index(osp.dirname(j[0]))][0]) for j in
                         sample_list]
            else:
                raise NameError('Unknown DataSet Name in %s' % self.source_folder)
        else:
            raise NameError('Unknown Split Type in %s' % self.split)
        # class_list[i]  = ['1', ApplyEyeMakeup]
        # sample_list[i] = ['ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01', '1']
        return class_list, sample_list


class BatchInput(object):
    def __init__(self,
                 data_root='/abs',
                 epoch_num=200,
                 inputdata='sf_UCF101frames_task_orig',
                 inputform='feed_dict' or 'tfrecords',
                 is_training=True,
                 class_and_sample_list=(['1', 'apple'], [['sub/sample_1', '1']]),
                 batch_size=64):
        self.data_root = data_root
        self.epoch_num = epoch_num
        self.inputdata = inputdata
        self.inputform = inputform
        self.is_training = is_training
        self.class_and_sample_list = class_and_sample_list
        self.batch_size = batch_size

    def get_batch_data(self):
        sa = oz_flags.StrAna(self.inputdata)
        if 'tfrecord' in self.inputform.lower():
            if sa.task() == 'orig':
                tfr_factory = tfrecordsio.OrigFramesTFR(data_root=self.data_root, inputdata=self.inputdata,
                                                        epoch_num=self.epoch_num, is_training=self.is_training,
                                                        batch_size=self.batch_size)
                tfrecords_list = tfr_factory.make_tfrecords_list(self.class_and_sample_list[1])
                baseio.check_or_create_tfrecords_multiprocessing(tfr_factory.check_and_create_tfrecords, tfrecords_list,
                                                                 self.class_and_sample_list[0])
                # tfr_factory.check_and_create_tfrecords(tfrecords_list, self.class_and_sample_list[0])
                data = tfr_factory.get_tfrecords_input(tfrecords_list, nets_map[F.net_name]['crop_size'])
            else:
                raise ValueError('Wrong string in FLAGS.inputdata: %s' % sa.task())

        elif 'feed' in self.inputform.lower():
            sample_queue = baseio.sample_list_for_epochs(self.class_and_sample_list[1], epoch_num=self.epoch_num,
                                                         shuffle=self.is_training)

            if sa.task() is not None and sa.source_folder() is not None and sa.sample_type() is not None and \
                    sa.sample_num() is not None and sa.consecutive() is not None and sa.constrain() is not None and \
                    sa.encoder() is not None and sa.specific_size() is not None:
                func = semialphaio.ones_multiprocessing
                args = (osp.join(self.data_root, sa.source_folder()), sa.task(), sa.sample_num(),
                        sa.constrain(), sa.consecutive(), sa.sample_type(), sa.encoder(),
                        nets_map[F.net_name]['crop_size'], sa.specific_size())
                sample_queue = [i[0] for i in sample_queue]
            else:
                raise ValueError('Wrong string in FLAGS.inputdata')

            @functools.wraps(func)
            def get_input(step):
                sub_sample_queue = sample_queue[self.batch_size * step: self.batch_size * step + self.batch_size]
                try:
                    return func(sub_sample_queue, *args)
                except ValueError:
                    return False

            data = get_input
        else:
            raise ValueError('Wrong form for input, feed_dict or TFRecords? for F.inputform')
        return data
