import numpy as np
import data_io.basepy as basepy

sample_txt = './guinea/test.txt'

array_list = np.random.rand(2, 3).tolist()

sep = ',,'

make_line = (str(array_list), ',,', '1', ',,', '32', '\n')

basepy.write_txt_add_lines(sample_txt, 'a', '1', 'name')
basepy.read_txt_lines2list(sample_txt)
