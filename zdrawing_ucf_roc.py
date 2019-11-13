from zresults_analysis import *

plt.title('ROC curves on UCF-Crime', fontsize=14)
plt.grid()
plt.xlim((0,1))
plt.ylim((0,1))
line_width=1.5
# fpr, tpr = np.load('./results/ped2_sultani_fpr.npy'), np.load('./results/ped2_sultani_tpr.npy')
# plt.plot(fpr, tpr, label='Sultani et al., AUC=%.2f%%' % (metrics.auc(fpr, tpr)*100))
# fpr, tpr = np.load('./results/ped2_zhong_fpr.npy')**0.7, np.load('./results/ped2_zhong_tpr.npy')
# plt.plot(fpr, tpr, label='Zhong et al., AUC=%.2f%%' % (metrics.auc(fpr, tpr)*100))
fpr, tpr = [0, 1], [0, 1]
plt.plot(fpr, tpr, label='Binary classifer,  AUC=50%', color='k',linewidth=line_width)

fpr, tpr = np.load('./results/ucf_roc_1_lu_fpr.npy'), np.load('./results/ucf_roc_1_lu_tpr.npy')
plt.plot(fpr, tpr, label='Lu et al.,        AUC=65.51%', color='m',linewidth=line_width)

fpr, tpr = np.load('./results/ucf_roc_2_hassan_fpr.npy'), np.load('./results/ucf_roc_2_hassan_tpr.npy')
plt.plot(fpr, tpr, label='Hassan et al., AUC=50.60%', color='orange',linewidth=line_width)

fpr, tpr = np.load('./results/ucf_roc_3_sultani_fpr.npy'), np.load('./results/ucf_roc_3_sultani_tpr.npy')
plt.plot(fpr, tpr, label='Sultani et al., AUC=74.44%', color='b',linewidth=line_width)

fpr, tpr = np.load('./results/ucf_roc_4_zhong_fpr.npy'), np.load('./results/ucf_roc_4_zhong_tpr.npy')
plt.plot(fpr, tpr, label='Zhong et al.,  AUC=81.08%', color='g',linewidth=line_width)

fpr, tpr = np.load('./results/ucf_roc_5_our_fpr.npy'), np.load('./results/ucf_roc_5_our_tpr.npy')
plt.plot(fpr, tpr, label='Ours,              AUC=83.16%', color='r',linewidth=line_width)

plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.legend(loc=4,fontsize=12)
plt.show()

print('debug this')