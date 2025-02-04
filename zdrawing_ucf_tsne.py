from zresults_analysis import *
a,b,c,d=collect_features('/absolute/datasets/UCSDped2_split_list/temproal_annotation.txt', '/absolute/datasets/UCSDped2_reform_motion_original_c3d_npy','/absolute/datasets/UCSDped2_reform_motion_pyramid_80_56_c3d_npy','/absolute/datasets/UCSDped2_spatial_annotation',max_num=32)
# a,b,c,d=collect_features('/absolute/datasets/Anomaly-Detection-Dataset/Temporal_Anomaly_Annotation_for_Testing_Videos_test_only.txt', '/absolute/datasets/anoma_motion_original_test_only_c3d_npy', '/absolute/datasets/anoma_motion_pyramid_120_85_c3d_npy', '/absolute/datasets/anoma_spatial_annotations',max_num=16)
tsne(a,b, title='Video clip features')
tsne(c,d, title='SEP features')

step = 2
tsne(np.array(c[::step]),np.array(d[::step]), title='t-SNE of anomalous and normal features using EVENT PROPOSAL')
tsne(np.array(a[::step]),np.array(b[::step]), title='t-SNE of anomalous and normal features using VIDEO CLIP')