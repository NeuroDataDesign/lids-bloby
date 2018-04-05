import numpy as np
import math
import os
import progressbar
from scipy import ndimage
from sklearn.mixture import (
    DPGMM,
    GaussianMixture
)
from collections import namedtuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tifffile import imsave
import csv

class ImageException(Exception):
    pass

def raster_3d_generator(img_shape):
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            for k in range(img_shape[2]):
                yield (i, j, k)


def DoG(img, gamma=2, sigma=2, print_level=0, scales=5):
    a = 0.5
    DoG_stack = []
    sigma_range = np.linspace(sigma, sigma+10, scales)
    bar = progressbar.ProgressBar()
    for sigma in bar(sigma_range):
        scale_constant = np.power(sigma, gamma - 1)
        G_1 = ndimage.filters.gaussian_filter(img, sigma+a)
        G_2 = ndimage.filters.gaussian_filter(img, sigma)
        DoG = scale_constant * (G_1 - G_2)/(a*sigma)
        DoG_stack.append((sigma, DoG))
    return DoG_stack


def img_3d_hessian(x):
    x_grad = np.gradient(x)
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = tol_check(np.nan_to_num(grad_kl))
    return hessian

def tol_check(x, tol = 1e-16):
    x[np.abs(x) < tol] = 0.0
    return x

def find_concave_points(H):
    img_iter = raster_3d_generator(H.shape[2:])
    concave_points = set()
    bar = progressbar.ProgressBar()
    for i,j,k in bar(img_iter):
        if is_negative_definite(H[:,:,i,j,k]):
            concave_points.add((i,j,k))
    return concave_points


def is_negative_definite(m):
    d1 = m[0, 0]
    d2 = np.linalg.det(m[:1, :1])
    d3 = np.linalg.det(m)
    return d1 > 0.0 and d2 > 0.0 and d3 < 0.0


def neighboring_pixels(z, y, x):
    # returns an iterator for 26 voxels around the center
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                if i != 0 or j != 0 or k != 0:
                    yield (i+z,j+y,k+x)


def get_neighbours(point):
    cz, cy, cx = point[0], point[1], point[2]
    neighbor_iter = neighboring_pixels(cz, cy, cx)
    return [(i, j, k) for i,j,k in neighbor_iter]


def find_connected_component(center, U, img):
    cz, cy, cx = center[0], center[1], center[2]
    shape_z, shape_y, shape_x = img.shape

    neighbor_iter = neighboring_pixels(cz, cy, cx)
    connected_component = []
    for i,j,k in neighbor_iter:
        if i >= 0 and j >= 0 and k >= 0 and i < shape_z and j < shape_y and k < shape_x and img[i,j,k] > 0:
            if (i,j,k) in U:
                connected_component.append((i,j,k))
    return connected_component


def draw_connected_components(img, connected_components, fname):
    draw_points = []
    for ccenter,cc in connected_components:
        for c in cc:
            draw_points.append(c)
    ImageDrawer.draw_centers(original_image, draw_points, (255, 0, 0), fname=fname)


def format_H(H):
    # If H was computed from a Z,Y,X image then the derivatives are inverted so
    # this method rectifies the mixup
    fxx = H[2, 2]
    fyy = H[1, 1]
    fzz = H[0, 0]
    fxy = H[1, 2]
    fxz = H[0, 2]
    fyz = H[0, 1]
    return np.array([
        [fxx, fxy, fxz],
        [fxy, fyy, fyz],
        [fxz, fyz, fzz]
    ])


def block_principal_minors(H):
    # This method assumes the hessian is set up canonically
    D_1 = H[0,0]
    D_2 = np.linalg.det(H[::2, ::2])
    D_3 = np.linalg.det(H)
    return D_1 + D_2 + D_3


def regional_blobness(H):
#     # This method assumes the hessian is set up canonically
#     det = np.linalg.det(H)
#     # Note these are the 2x2 principal minors
#     pm = block_principal_minors(H)
#     return 3*np.abs(det)**(2.0/3)/pm

    [lp3, lp2, lp1],_ = np.linalg.eig(H)
    return abs(lp1 * lp2 * lp3)/(max(abs(lp1*lp2), abs(lp2*lp3), abs(lp1*lp3))**1.5)


def regional_flatness(H):
    # This method assumes the hessian is set up canonically
#     tr = np.trace(H)
#     pm = block_principal_minors(H)
#     return np.sqrt(tr**2 - 2*pm)
    # import pdb; pdb.set_trace()
    [lp3, lp2, lp1],_ = np.linalg.eig(H)
    return math.sqrt(lp1**2 + lp2**2 + lp3**2)


class BlobCandidate(namedtuple('BlobCandidate', ['center', 'blobness', 'flatness', 'avg_int'])):
    pass


def normalize_image(img):
    img_max = np.amax(img)
    img_min = np.amin(img)
    img = np.subtract(img, img_min)
    img = np.divide(img, img_max - img_min)
    return img

def post_prune(blob_candidates):
    candidates_features = []
    candidate_coords = []
    PIXELS_PER_BLOB = 35
    for c in blob_candidates:
        data_point = []
        data_point.extend([c[1], c[2], c[3]])
        candidates_features.append(data_point)
        candidate_coords.append(list(c[0]))

    if len(candidates_features) <= 2:
        return candidate_coords

    #print('Running GMM on blob candidates')

    model = GaussianMixture(n_components=2, covariance_type='full')

    model.fit(candidates_features)
    class_labels = model.predict(candidates_features)
    scores = model.score_samples(np.array(candidates_features))

    avg_scores = np.zeros(len(np.unique(class_labels)))

    for i, x in enumerate(candidates_features):
        avg_scores[class_labels[i]] += scores[i]

    for i,avg_score in enumerate(avg_scores):
        n = len([x for x in class_labels if x == i])
        avg_scores[i] = avg_score/float(n)

    blob_class_index = list(avg_scores).index(max(avg_scores))
    blobs = [b for i,b in enumerate(candidate_coords) if class_labels[i] == blob_class_index]

    blobs = [(b[0], b[1], b[2]) for b in blobs]
    #return blobs

    clusters = int(len(blobs)/PIXELS_PER_BLOB)

    max_score = 0
    max_kmeans = None

    min_k = max(2, clusters-10)
    max_k = clusters + 10

    print('running KMeans from {} to {}'.format(min_k, max_k))
    for k in range(min_k, max_k):
        #print('K={}'.format(k))
        try:
            kmeans = KMeans(n_clusters=k, init='k-means++')
            cluster_labels = kmeans.fit_predict(blobs)
            s_score = silhouette_score(blobs, cluster_labels)
        except ValueError:
            #print('K={} not possible'.format(k))
            continue

        #print('silhouette_score={}'.format(s_score))

        if s_score > max_score:
            max_score = s_score
            max_kmeans = kmeans
    if max_kmeans == None:
        return blobs
    # max_kmeans = KMeans(n_clusters=14, init='k-means++')
    # cluster_labels = max_kmeans.fit_predict(blobs)
    return [[math.ceil(b[0]), math.ceil(b[1]), math.ceil(b[2])] for b in max_kmeans.cluster_centers_]

def draw_centers(orig_img, points, copy=True):
    drawn_img = None
    for z,y,x in points:
        if copy:
            drawn_img = draw_square(orig_img, z, y, x, 2, [255, 0, 0])
        else:
            draw_square(orig_img, z, y, x, 2, [255, 0, 0])
        # original_image[b[0], b[1], b[2], :] = [255, 0, 0]
    if copy:
        return drawn_img

def min_max(x, minimum, maximum):
    return np.maximum(np.minimum(x, minimum), maximum)

def set_rgb(img, x, y, z, r, g, b):
    img[z, y, x, 0] = r
    img[z, y, x, 1] = g
    img[z, y, x, 2] = b

def draw_square(img, x, y, z, radi, rgb, copy=True, overwrite=True):
    z_range, y_range, x_range, _ = img.shape
    drawn_img = np.copy(img) if copy else img
    r, g, b = rgb
    for i in range(radi):
        for j in range(radi):
            for k in range(radi):
                if overwrite:
                    set_rgb(drawn_img,
                            min_max(x+i, x_range-1, 0),
                            min_max(y+j, y_range-1, 0),
                            min_max(z+k, z_range-1, 0),
                            r,
                            g,
                            b)
                else:
                    add_rgb(drawn_img,
                            min_max(x+i, x_range-1, 0),
                            min_max(y+j, y_range-1, 0),
                            min_max(z+k, z_range-1, 0),
                            r,
                            g,
                            b)
    return drawn_img

def save_tif(img, fname):
    if ".tif" not in fname:
        fname = fname + ".tif"
    save_path = "../img/"+fname if os.path.isdir("../img/") else fname
    save_path = "./img/"+fname if os.path.isdir("./img/") else fname
    imsave(save_path, img.astype(np.uint8))
    print("Saved tif as: ", fname, " at ", save_path)

def write_csv(rows, fname):
    if ".csv" not in fname:
        fname = fname + ".csv"
    save_path = "../centers/"+fname if os.path.isdir("../centers/") else fname
    save_path = "./centers/"+fname if os.path.isdir("./centers/") else fname
    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    print("Saved csv as: ", fname, " at ", save_path)
