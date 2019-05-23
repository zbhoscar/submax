import data_io.basepy as basepy

# Default keys
EXPERIMENT_KEYS = {'batch_size': 64,
                   'epoch_num': 1,
                   'learning_rate_base': 0.001,
                   'moving_average_decay': 0.99,
                   'regularization_scale': 0.0005,
                   'set_gpu': '0',
                   'feature_len': 4096,
                   'segment_num': 1001,
                   'lambda1': 0.00008,
                   'lambda2': 0.00008,
                   'fusion': 'standard',
                   'attention_l': 4096,
                   'embedding': '_avg'
                   }

def check(json_file_path):
    return basepy.DictCtrl(EXPERIMENT_KEYS).read4path(json_file_path)

