from zresults_analysis import *


a,b,c,d=collect_features('/absolute/datasets/UCSDped2_split_list/temproal_annotation.txt', '/absolute/datasets/UCSDped2_reform_motion_original_c3d_npy','/absolute/datasets/UCSDped2_reform_motion_pyramid_80_56_c3d_npy','/absolute/datasets/UCSDped2_spatial_annotation',max_num=32)
# a,b,c,d=collect_features('/absolute/datasets/Anomaly-Detection-Dataset/Temporal_Anomaly_Annotation_for_Testing_Videos_test_only.txt', '/absolute/datasets/anoma_motion_original_test_only_c3d_npy', '/absolute/datasets/anoma_motion_pyramid_120_85_c3d_npy', '/absolute/datasets/anoma_spatial_annotations',max_num=16)
tsne(c,d, title='t-SNE of anomalous and normal features using EVENT PROPOSAL')
tsne(a,b, title='t-SNE of anomalous and normal features using WHOLE FRAME')

step = 2
tsne(np.array(c[::step]),np.array(d[::step]), title='t-SNE of anomalous and normal features using EVENT PROPOSAL')
tsne(np.array(a[::step]),np.array(b[::step]), title='t-SNE of anomalous and normal features using WHOLE FRAME')

from zresults_analysis import *
multi,single,frame = recall_iou_all('/absolute/datasets/UCSDped2_split_list/temproal_annotation.txt','/absolute/datasets/UCSDped2_spatial_annotation','/absolute/datasets/UCSDped2_reform_motion_pyramid_60_42_all_json')
# multi,single,frame = recall_iou_all('/absolute/datasets/Anomaly-Detection-Dataset/Temporal_Anomaly_Annotation_for_Testing_Videos_anomaly_only.txt', '/absolute/datasets/anoma_spatial_annotations', '/absolute/datasets/anoma_motion_pyramid_180_127_all_json')

grouth_truth = [1] * len(multi)
pm, rm, tm = metrics.precision_recall_curve(grouth_truth, multi)
ps, rs, ts = metrics.precision_recall_curve(grouth_truth, single)
pf, rf, tf = metrics.precision_recall_curve(grouth_truth, frame)

plt.grid()
plt.xlim((0,1))
plt.ylim((0,1))
plt.plot(tf, rf[1:], color= '#3C75AF', label='Original')
plt.plot(ts, rs[1:], color= '#EE8635', label='Single region')
plt.plot(tm, rm[1:], color= '#C53932', label='Multi region')
plt.xlabel('IOU', fontsize=12)
plt.ylabel('Recall', fontsize=12)
plt.legend(fontsize=12)


plt.plot(tm, rm[:-1])
plt.plot(ts, rs[:-1])
plt.plot(tf, rf[:-1])



