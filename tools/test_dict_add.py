import numpy as np
import time

feature_dict = {}


for i in range(190190):
    if i % 1001 == 1 :
        t1 = time.time()
    class_video_name = np.random.randint(190)
    np_as_line = np.random.rand(4014)

    if class_video_name in feature_dict.keys():
        feature_dict[class_video_name] = np.concatenate(
            (feature_dict[class_video_name], np.expand_dims(np_as_line, axis=0)))
    else:
        feature_dict[class_video_name] = np.expand_dims(np_as_line, axis=0)

    if i % 1001 ==0 and i !=0:
        print((time.time()-t1)/1001)

print('wow')