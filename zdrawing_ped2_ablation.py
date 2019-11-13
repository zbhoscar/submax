import matplotlib.pyplot as plt
import numpy as np

single_region_single_scale= [84.69, 87.63, 87.78, 84.02]
multi_region_single_scale=[86.64, 91.77, 90.54, 88.98]
single_region_multi_scale=[87.60, 89.09, 89.44, 85.65]
multi_region_multi_scale=[87.94, 94.10, 93.27, 89.87]

plt.figure(figsize=(6.4, 4.8))
plt.grid(axis='y')
plt.xticks(np.arange(4), ('40x40', '60x60', '80x80', '120x120') )
# plt.xlim((0,1))
plt.ylim((81,95))
plt.plot(np.arange(4), single_region_single_scale, marker='s',color= 'b', label='single region + non-pyramid feature') # '#3C75AF', label='Original')
plt.plot(np.arange(4), single_region_multi_scale, marker='v', color= 'm', label='single region + pyramid feature')  # '#EE8635', label='Single region')
plt.plot(np.arange(4), multi_region_single_scale, marker='o', color= 'g', label='multi region + non-pyramid feature')  # '#EE8635', label='Single region')
plt.plot(np.arange(4), multi_region_multi_scale, marker='*', color= 'r', label='multi region + pyramid feature')    # '#C53932', label='Multi region')
plt.xlabel('SEP window size', fontsize=14)
plt.ylabel('AUC (%)', fontsize=14)
plt.legend(loc=8, fontsize=12)
plt.show()