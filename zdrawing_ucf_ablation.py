import matplotlib.pyplot as plt
import numpy as np

single_region_single_scale= [74.19, 76.61, 78.43, 77.36]
multi_region_single_scale=[76.83, 79.20, 81.01, 80.32]
single_region_multi_scale=[74.50, 77.71, 79.23, 78.52]
multi_region_multi_scale=[77.22, 80.94, 83.16, 81.56]

linewidth = 2
plt.figure(figsize=(3.7, 4.8))
plt.grid(axis='y')
plt.xticks(np.arange(4), ('40x40', '60x60', '80x80', '120x120'), fontsize=16)
# plt.xlim((0,1))
plt.ylim((72,84))
plt.plot(np.arange(4), single_region_single_scale, marker='s',color= 'b', label='s + n', linewidth = linewidth) # '#3C75AF', label='Original')
plt.plot(np.arange(4), single_region_multi_scale, marker='v', color= 'm', label='s + p', linewidth = linewidth)  # '#EE8635', label='Single region')
plt.plot(np.arange(4), multi_region_single_scale, marker='o', color= 'g', label='m + n', linewidth = linewidth)  # '#EE8635', label='Single region')
plt.plot(np.arange(4), multi_region_multi_scale, marker='*', color= 'r', label='m + p', linewidth = linewidth)    # '#C53932', label='Multi region')
plt.xlabel('SEP window size', fontsize=20)
plt.ylabel('AUC (%)', fontsize=20)
plt.legend(fontsize=14)
plt.show()

# plt.plot(np.arange(4), single_region_single_scale, marker='s',color= 'b', label='sSEP + nonpyramid') # '#3C75AF', label='Original')
# plt.plot(np.arange(4), single_region_multi_scale, marker='v', color= 'm', label='sSEP + pyramid')  # '#EE8635', label='Single region')
# plt.plot(np.arange(4), multi_region_single_scale, marker='o', color= 'g', label='mSEP + nonpyramid')  # '#EE8635', label='Single region')
# plt.plot(np.arange(4), multi_region_multi_scale, marker='*', color= 'r', label='mSEP + pyramid')    # '#C539
# plt.xlabel('SEP window size', fontsize=14)
# plt.ylabel('AUC (%)', fontsize=14)
# plt.legend(loc=4, fontsize=12)