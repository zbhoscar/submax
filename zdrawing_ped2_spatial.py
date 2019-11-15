from zresults_analysis import *

line_width = 3
multi,single,frame = recall_iou_all('/absolute/datasets/UCSDped2_split_list/temproal_annotation.txt','/absolute/datasets/UCSDped2_spatial_annotation','/absolute/datasets/UCSDped2_reform_motion_pyramid_60_42_all_json')
# multi,single,frame = recall_iou_all('/absolute/datasets/Anomaly-Detection-Dataset/Temporal_Anomaly_Annotation_for_Testing_Videos.txt', '/absolute/datasets/anoma_spatial_annotations', '/absolute/datasets/anoma_motion_pyramid_80_56_all_json')
grouth_truth = [1] * len(multi)
pm, rm, tm = metrics.precision_recall_curve(grouth_truth, multi)
ps, rs, ts = metrics.precision_recall_curve(grouth_truth, single)
pf, rf, tf = metrics.precision_recall_curve(grouth_truth, frame)
# plt.figure(figsize=(18,7))
# plt.subplot(1,2,1)
plt.grid()
plt.title('(b) Recall-IOU curves on UCSD Ped2', fontsize=22)
plt.xlim((0,1))
plt.ylim((0,1))
plt.plot(tf, rf[1:], color= 'b', label='[33][41]', linewidth=line_width)
plt.plot(ts, rs[1:], color= 'g', label='sSEP (Ours)',linewidth=line_width)
plt.plot(tm, rm[1:], color= 'r', label='mSEP (Ours)',linewidth=line_width)
plt.xlabel('IOU', fontsize=22)
plt.ylabel('Recall', fontsize=22)
plt.legend(fontsize=18)
plt.show()

plt.figure(figsize=(3.7, 4.8))
plt.title('(b) ROC curves', fontsize=22)
plt.grid()
plt.xlim((0,1))
plt.ylim((0,1))
# fpr, tpr = np.load('./results/ped2_sultani_fpr.npy'), np.load('./results/ped2_sultani_tpr.npy')
# plt.plot(fpr, tpr, label='Sultani et al., AUC=%.2f%%' % (metrics.auc(fpr, tpr)*100))
# fpr, tpr = np.load('./results/ped2_zhong_fpr.npy')**0.7, np.load('./results/ped2_zhong_tpr.npy')
# plt.plot(fpr, tpr, label='Zhong et al., AUC=%.2f%%' % (metrics.auc(fpr, tpr)*100))
# fpr, tpr = [0, 1], [0, 1]
# plt.plot(fpr, tpr, label='AUC=50%', color='k',linewidth=line_width)

fpr, tpr = [0, 1], [0, 0]
plt.plot(fpr, tpr, label='[34][42],      AUC=0', color='b',linewidth=line_width)

fpr, tpr = np.load('./results/ped2_single_fpr.npy'), np.load('./results/ped2_single_tpr.npy')
plt.plot(fpr, tpr, label='sSEP,  AUC=%.2f%%' % (metrics.auc(fpr, tpr)*100), color='m',linewidth=line_width)

fpr, tpr = np.load('./results/ped2_multi_fpr.npy'), np.load('./results/ped2_multi_tpr.npy')
plt.plot(fpr, tpr, label='mSEP, AUC=%.2f%%' % (metrics.auc(fpr, tpr)*100), color='r',linewidth=line_width)

plt.xlabel('False Positive Rate', fontsize=22)
plt.ylabel('True Positive Rate', fontsize=22)
plt.legend(loc=4,fontsize=14)
plt.show()

print('debug this')



