"""This class is the core detector for this package"""

from tifffile import imread, imsave
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from skimage import measure, transform, morphology
import scipy.stats
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter

__docformat__ = 'reStructuredText'

CHUNK_VOL_CUTOFF = 5000
AXIS_LENGTH_CUTOFF = 15

class BlobDetector(object):
    """
    BlobDetector class can be instantiated with the following args

    - **parameters**, **types**, **return** and **return types**::
    :param tif_img_path: full path of the input TIF stack
    :param data_source: either 'laVision' or 'COLM' - the imaging source of the input image
    :type tif_img_path: string
    :type data_source: string
    """

    def __init__(self, tif_img_path, data_source='COLM', verbose=False):
        self.img = imread(tif_img_path)
        self.n_components = 4 if data_source == 'COLM' else 4
        self.data_source = data_source
        self.verbose = verbose

    def _gmm_cluster(self, img, data_points, n_components):
        if self.verbose:
            print('Starting GMM Cluster')

        num_pixels = len(img.flatten())
        scale_factor = 8 if num_pixels <= 3.6e7 else 64
        data_points = img.reshape(-1, 1)[::scale_factor]

        if self.verbose:
            print('Image reshape')

        v = 2 if self.verbose else 0
        gmm = GaussianMixture(n_components=n_components, covariance_type='spherical', verbose=v).fit(data_points)
        self.gmm = gmm

        if self.verbose:
            print('Fitting GMM')

        means = gmm.means_.flatten()
        stdev = np.sqrt(gmm.covariances_.flatten())
        max_index = np.argmax(means)

        start_th = int(means[max_index] - (stdev[max_index]/2))
        end_th = int(means[max_index] + (stdev[max_index]/2))

        max_prob = 0.0
        max_prob_th = 0.0

        max_comparison = 1.0 if img.max() >= 63000 else 0.92

        for th in range(start_th, end_th, 25):
            prob = max(gmm.predict_proba([[th]]).flatten())
            if prob >= max_comparison:
                max_prob = prob
                max_prob_th = th
                break

        self.threshold = max_prob_th if not max_prob_th == 0.0 else (start_th + end_th)/2.0
        if self.verbose:
            print('Threshold chosen', self.threshold)

        shape_z, shape_y, shape_x = img.shape

        new_img = np.ndarray((shape_z, shape_y, shape_x))
        new_img[img > self.threshold] = 255
        new_img[img < self.threshold] = 0

        self.thresholded_img = new_img

        if self.verbose:
            print('Thresholding done successfully')

        return new_img

    def _get_physical_length(self, rprop):
        z, y, x = rprop.image.shape

        if z <= 2 or y <= 2 or x <= 2:
            return 0.0

        rescaled_roi = transform.rescale(rprop.image, scale=[0.5, 0.5, 5], multichannel=False)
        rescaled_roi[rescaled_roi != 0] = 255
        label_img = measure.label(rescaled_roi, background=0)

        rescaled_rprops = measure.regionprops(label_img)
        return rescaled_rprops[0].major_axis_length

    def _span_z_max(self, z, y, x, z_range=4):
        ints = []
        r_start = max(0, z-z_range)
        r_end = min(z+z_range+1, self.img.shape[0] - 1)
        r = range(r_start, r_end)
        for z_index in r:
            ints.append(self.img[z_index, y, x])
        return max(ints)

    def _get_extended_region_props(self, region_props):
        extended_region_props = []
        for rprop in region_props:
            z, y, x = [int(round(rprop.centroid[0])), int(round(rprop.centroid[1])), int(round(rprop.centroid[2]))]
            props = {
                'centroids': [z, y, x],
                'mean_intensity': rprop.mean_intensity,
                'volume_in_vox': rprop.area,
                'label': rprop.label,
                'span_z_max': self._span_z_max(z, y, x),
                'bbox': rprop.bbox,
                'coords': rprop.coords
            }
            extended_region_props.append(props)
        return extended_region_props

    def get_blob_centroids(self):
        """
        Gets the blob centroids based on GMM thresholding, erosion and connected components
        """
        uniq = np.unique(self.img, return_counts=True)

        data_points = [p for p in zip(*uniq)]
        gm_img = self._gmm_cluster(self.img, data_points, self.n_components)

        eroded_img = gm_img
        
        if self.data_source == 'COLM':
            eroded_img = morphology.erosion(gm_img)
        else:
            if self.verbose:
                print('Opening operation of image')

            eroded_img = morphology.opening(gm_img)

            labeled_img = measure.label(gm_img, background=0)
            extended_region_props = self._get_extended_region_props(measure.regionprops(labeled_img, self.img))

            if self.verbose:
                print('Removing larger chunks')

            large_chunks = [prop for prop in extended_region_props if prop['volume_in_vox'] >= CHUNK_VOL_CUTOFF]

            for region in large_chunks:
                z_min, y_min, x_min, z_max, y_max, x_max = region['bbox']
                chunk = self.img[z_min:z_max, y_min:y_max, x_min:x_max]
                chunk[chunk <= self.threshold * 1.75] = 0
                chunk[chunk >= self.threshold * 1.75] = 255
                chunk = morphology.opening(chunk)
                eroded_img[z_min:z_max, y_min:y_max, x_min:x_max] = chunk

            if self.verbose:
                print('Getting region props')

        labeled_img = measure.label(eroded_img, background=0)
        extended_region_props = self._get_extended_region_props(measure.regionprops(labeled_img, self.img))

        self.processed_img = eroded_img
        self.extended_region_props = extended_region_props

        if self.data_source == 'COLM':
            centroids = [rprop['centroids'] for rprop in extended_region_props if rprop['volume_in_vox'] >= 15]
        else:
            centroids = [rprop['centroids'] for rprop in extended_region_props if rprop['volume_in_vox'] >= 3]
        return centroids

    def get_avg_intensity_by_region(self, reg_atlas_path):
        """
        Given registered atlas image path, gives the average intensity of the regions
        """

        reg_img = imread(reg_atlas_path).astype(np.uint16)
        raw_img = self.img.astype(np.uint16)

        region_numbers = np.unique(reg_img, return_counts=True)[0]

        region_intensities = {}

        rgn_pbar = tqdm(region_numbers)

        for rgn in rgn_pbar:
            rgn_pbar.set_description('Summing intensities of region {}'.format(rgn))

            voxels = np.where(reg_img == rgn)
            voxels = map(list, zip(*voxels))
            region_intensities[str(rgn)] = float(np.sum([raw_img[v[0], v[1], v[2]] for v in voxels]))

        return region_intensities
