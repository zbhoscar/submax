import matplotlib.pyplot as plt
import numpy as np

single_region_single_scale=[65.82, 67.73, 78.49, 76.90]
multi_region_single_scale=[67.22, 76.81, 81.09, 80.43]
single_region_multi_scale=[]
multi_region_multi_scale=[70.53, 77.62, 83.16, 82.87]

plt.grid(axis='y')
plt.xticks(np.arange(4), ('40x40', '60x60', '80x80', '120x120') )
# plt.xlim((0,1))
plt.ylim((65,85))
plt.plot(np.arange(4), single_region_single_scale, marker='o',color= 'b', label='Original') # '#3C75AF', label='Original')
plt.plot(np.arange(4), multi_region_single_scale, marker='v', color= 'g', label='Single region')  # '#EE8635', label='Single region')
plt.plot(np.arange(4), multi_region_multi_scale, marker='s', color= 'r', label='Multi region')    # '#C53932', label='Multi region')
plt.xlabel('IOU', fontsize=12)
plt.ylabel('Recall', fontsize=12)
plt.legend(fontsize=12)
plt.show()