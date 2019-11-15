from zresults_analysis import *

line_width=2
plt.figure(figsize=(3.7, 4.8))
plt.title('ROC curves on UCF-Crime', fontsize=14)
plt.grid()
plt.xlim((0,1))
plt.ylim((0,1))

fpr, tpr = np.load('./results/ped2_roc_sultani_fpr.npy'), np.load('./results/ped2_roc_sultani_tpr.npy')
plt.plot(fpr, tpr, label='[34],   AUC=82.03%', color='b',linewidth=line_width)
# plt.plot(fpr, tpr, label='Sultani et al., AUC=74.44%', color='b',linewidth=line_width)

fpr, tpr = np.load('./results/ped2_roc_zhong_fpr.npy'), np.load('./results/ped2_roc_zhong_tpr.npy')
plt.plot(fpr, tpr, label='[42],   AUC=88.12%', color='g',linewidth=line_width)
# plt.plot(fpr, tpr, label='Zhong et al.,  AUC=81.08%', color='g',linewidth=line_width)

fpr, tpr = np.load('./results/ped2_roc_sSEP_fpr.npy'), np.load('./results/ped2_roc_sSEP_tpr.npy')
plt.plot(fpr, tpr, label='sSEP,  AUC=89.12%', color='m',linewidth=line_width)

fpr, tpr = np.load('./results/ped2_roc_mSEP_fpr.npy'), np.load('./results/ped2_roc_mSEP_tpr.npy')
plt.plot(fpr, tpr, label='mSEP, AUC=94.10%', color='r',linewidth=line_width)

plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.legend(fontsize=14)
plt.show()

print('debug this')

# fpr, tpr = np.load('./results/ped2_sultani_fpr.npy'), np.load('./results/ped2_sultani_tpr.npy')
# plt.plot(fpr, tpr, label='Sultani et al., AUC=%.2f%%' % (metrics.auc(fpr, tpr)*100))
# fpr, tpr = np.load('./results/ped2_zhong_fpr.npy')**0.7, np.load('./results/ped2_zhong_tpr.npy')
# plt.plot(fpr, tpr, label='Zhong et al., AUC=%.2f%%' % (metrics.auc(fpr, tpr)*100))
# fpr, tpr = [0, 1], [0, 1]
# plt.plot(fpr, tpr, label='Binary classifer,  AUC=50%', color='k',linewidth=line_width)
# fpr, tpr = np.load('./results/ucf_roc_2_hassan_fpr.npy'), np.load('./results/ucf_roc_2_hassan_tpr.npy')
# plt.plot(fpr, tpr, label='[13],  50.60%', color='orange',linewidth=line_width)
# # plt.plot(fpr, tpr, label='Hasan et al.,  AUC=50.60%', color='orange',linewidth=line_width)
#
# fpr, tpr = np.load('./results/ucf_roc_1_lu_fpr.npy'), np.load('./results/ucf_roc_1_lu_tpr.npy')
# plt.plot(fpr, tpr, label='[24],  65.51%', color='m',linewidth=line_width)
# plt.plot(fpr, tpr, label='Lu et al.,        AUC=65.51%', color='m',linewidth=line_width)