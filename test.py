from src.BlobDetector import BlobDetector
from src import util
from src.BlobMetrics import BlobMetrics
import matplotlib.pyplot as plt
import numpy as np

# aic 5,6,7,8,9
# aic

# aic_2 = [808172.65,946932.28,918745.32,1055535.54,481649.52,1227079.82,1249242.79,699104.6,1492844.94,1194254.93]
# aic_3 = [689779.31,815726.04,860704.09,991744.4,492800.38,1152074.14,1178049.7,653409.68,1420399.59,1122719.17]
#
# bic_2 = [808262.5,947022.93,918836.68,1055626.84,481733.22,1227173.47,1249337.42,699192.27,1492940.27,1194347.93]
# bic_3 = [756407.07,815874.17,860853.12,991888.03,458448.92,1152227.57,1178187.82,653545.21,1420547.2,1122857.04]
#
# bar_width = 0.35
#
# x_values = np.arange(10)
# best_clusters = [2,2,2,2,3,3,3,3,3,3]
#
# aic_2_bars = plt.bar(x_values, aic_2, bar_width, alpha=0.4, color='b', label='k=2')
# aic_3_bars = plt.bar(x_values + bar_width, aic_3, bar_width, alpha=0.4, color='r', label='k=3')
#
# scatter_x = []
# scatter_y = []
#
# for i in x_values:
#     if best_clusters[i] == 2:
#         scatter_x.append(i)
#         scatter_y.append(aic_2[i])
#     else:
#         scatter_x.append(i + bar_width)
#         scatter_y.append(aic_3[i])
#
# plt.scatter(scatter_x, scatter_y, color='b', label='Manually chosen k-value')
#
# plt.xlabel('Subvolume #')
# plt.ylabel('AIC')
# plt.title('GMM Akaike Information Criterion for k=2 and k=3')
#
# plt.xticks(x_values + bar_width / 2, x_values)
# plt.legend()
# plt.tight_layout()
# plt.show()

exp_name = 'cell_detection_'
best_clusters = [2,2,2,2,3,3,3,3,3,3]

for i in range(10):
    print('{}{}'.format(exp_name, i))
    detector = BlobDetector('data/s3617/tifs/{}{}.tiff'.format(exp_name, i), n_components=best_clusters[i])
    centroids, diameters = detector.get_blob_centroids()

    f, p = plt.subplots()
    _, _, patches = p.hist(diameters, bins=int(len(diameters)/2), rwidth=1)
    for patch in patches:
        patch.set_edgecolor('k')

    p.set_title('Distribution of detected cells\' diameter')
    p.set_xlabel('Diameter')
    p.set_ylabel('Count')
    f.savefig('plots/diameter/{}{}.png'.format(exp_name, i))


    # detector2 = BlobDetector('data/s3617/tifs/{}{}.tiff'.format(exp_name, i), n_components=3)
    # centroids2, sizes2 = detector2.get_blob_centroids()
    # print('...')
    #
    # ground_truth_path = 'data/s3617/annotations/{}{}.csv'.format(exp_name, i)
    # ground_truth = util.get_list_from_csv(ground_truth_path)
    #
    # metrics1 = BlobMetrics(ground_truth, centroids1, euclidean_distance_threshold=12)
    # print('Precision: {}\nRecall: {}'.format(metrics1.precision(), metrics1.recall()))
    # print('F1 Measure (k=2):', metrics1.f_measure())
    #
    # metrics2 = BlobMetrics(ground_truth, centroids2, euclidean_distance_threshold=12)
    # print('Precision: {}\nRecall: {}'.format(metrics2.precision(), metrics2.recall()))
    # print('F1 Measure (k=3):', metrics2.f_measure())
    # print('.....')
