'''
    CAUTION
    MAKE SURE PATH IS CORRECT
    '''
import os
import shutil

log_path = '/home/zbh/Dropbox/droplogaaaaa'
log_list = [i[:12] for i in os.listdir(log_path) if '.txt' in i]    # 12 for 201809281102
model_path = '/home/zbh/Desktop/alpha_1_zbh/tf_modelsaaaaa'
model_list = [i for i in os.listdir(model_path)]
_ = [shutil.rmtree(os.path.join(model_path, j)) for j in model_list if j not in log_list]