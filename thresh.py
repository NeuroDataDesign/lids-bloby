
# coding: utf-8

# # Threshold & GMM Analysis

# In[143]:

from tifffile import imread, imsave
from src.BlobDetector import BlobDetector
from src.BlobMetrics import BlobMetrics
from src import util
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from datetime import datetime

exp_name = 'cell_detection_9'
data_source = 'COLM' if int(exp_name.split('_')[2]) <= 3 else 'laVision'

input_tif_path = './data/s3617/tifs/{}.tiff'.format(exp_name)
img = imread(input_tif_path)
detector = BlobDetector(input_tif_path, data_source='COLM')
s1 = datetime.now()
centroids = detector.get_blob_centroids()
s2 = datetime.now()
print('time taken', (s2-s1))
print(detector.threshold)

gmm = detector.gmm
x = np.arange(0, img.max())
std_devs = np.sqrt(np.linalg.eigvals(gmm.covariances_).flatten())

gaussians = np.array([p * norm.pdf(x, mu, std_dev) for mu, std_dev, p in zip(gmm.means_.flatten(), std_devs, gmm.weights_)])

img = imread(input_tif_path)

plt.hist(img.flatten(), bins=256, normed=True)

for gauss in gaussians:
    plt.plot(x, gauss, linewidth=3.0)

#plt.show()

ground_truth_path = 'data/s3617/annotations/{}.csv'.format(exp_name)
ground_truth = util.get_list_from_csv(ground_truth_path)

metrics = BlobMetrics(ground_truth, centroids, euclidean_distance_threshold=12)
print('Precision: {}\nRecall: {}'.format(metrics.precision(), metrics.recall()))
imsave('threshold_analysis.tiff', detector.thresholded_img.astype(np.uint8))


# In[ ]:
