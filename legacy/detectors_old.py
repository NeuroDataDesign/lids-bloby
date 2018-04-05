from scipy import (
    ndimage,
    signal
)
from image_processing import (
    gradient_x,
    gradient_y,
    gradient_z,
    BlobCandidate,
    voxel_region_iter,
    raster_3d_generator,
    block_principal_minors,
    img_hessian_3d,
    format_H,
    regional_blobness,
    regional_flatness,

)

from util import bound_check
import numpy as np
import progressbar
import math
import csv
from sklearn.mixture import DPGMM, GaussianMixture

DOG_STACK_SIZE = 5

def DoG(img, gamma = 2, dark = 1, sigma = 2, print_level = 0):
    # if the image has white blobs then invert them to being dark blobs
    if not dark:
        img = 1 - img

    # Differential difference
    a = 0.01
    DoG_stack = []
    sigma_range = np.linspace(sigma, sigma+10, DOG_STACK_SIZE)
    if print_level:
        bar = progressbar.ProgressBar()
        sigma_range = bar(sigma_range)
    for sigma in sigma_range:
        scale_constant = np.power(sigma, gamma - 1)
        # TODO: Do we need a inhouse gaussian filter to control filter size?
        G_1 = ndimage.filters.gaussian_filter(img, sigma+a)
        G_2 = ndimage.filters.gaussian_filter(img, sigma)
        DoG = scale_constant * (G_1 - G_2)/a
        DoG_stack.append((sigma,DoG))
    return DoG_stack

def blob_descriptors(DoG_img, intensity_img, sigma, U):
    # TODO: Add functionality to do a truncated blob-descriptor.
    #       iterations are taking too long.
    blob_candidates_T = []
    H = img_hessian_3d(DoG_img)
    H_filter = np.zeros(DoG_img.shape)
    z_range, y_range, x_range = intensity_img.shape
    for i_c, j_c, k_c in U:
        # For each blob candidate, calculate the regional hession
        region_iter = voxel_region_iter(i_c, j_c, k_c)
        regional_H = np.copy(H[i_c,j_c,k_c,:])
        # TODO: Check is it average intenity of original or DoG?
        intensity = intensity_img[i_c, j_c, k_c]
        for i_r, j_r, k_r in region_iter:
            if bound_check(i_r, 0, z_range-1) and bound_check(j_r, 0, y_range-1) and bound_check(k_r, 0, x_range-1):
                intensity += intensity_img[i_r, j_r, k_r]
                regional_H += H[i_r, j_r, k_r,:]
        regional_H = format_H(regional_H)
        blobness = regional_blobness(regional_H)
        flatness = regional_flatness(regional_H)
        avg_int = intensity/7
        blob_candidates_T.append(
            BlobCandidate(
                (i_c, j_c, k_c ),
                sigma,
                blobness,
                flatness,
                avg_int
            )
        )
    return blob_candidates_T

def find_negative_curvative_points(img):
    img_fx, img_fy, img_fz = gradient_x(img), gradient_y(img), gradient_z(img)
    img_fxy, img_fxz, img_fyz = gradient_y(img_fx), gradient_z(img_fx), gradient_z(img_fy)
    img_fxx, img_fyy, img_fzz = gradient_x(img_fx), gradient_y(img_fy), gradient_z(img_fz)
    D_1, D_2, D_3, H_N = np.zeros(img.shape), np.zeros(img.shape), np.zeros(img.shape), np.zeros(img.shape)
    D_1[img_fxx < 0] = 1

    D_2[(img_fxx * img_fyy - img_fxy * img_fxy) > 0] = 1
    D_3[(
            img_fxx * (img_fyy*img_fzz - img_fyz*img_fyz) \
            - img_fxy * (img_fxy*img_fzz - img_fyz*img_fxz) \
            + img_fxz * (img_fxy*img_fyz - img_fyy * img_fxz)
        ) < 0
    ] = 1

    H_N[np.logical_and(D_1 == D_2, D_2 == D_3)] = 1
    return np.argwhere(H_N == 1)

considered_candidates = []

def get_neighbours(point):
  cx = point[0]
  cy = point[1]
  cz = point[2]

  neighbors = [[cx, cy, cz + 1],
               [cx, cy, cz - 1],
               [cx, cy + 1, cz],
               [cx, cy - 1, cz],
               [cx, cy + 1, cz + 1],
               [cx, cy + 1, cz - 1],
               [cx, cy - 1, cz + 1],
               [cx, cy - 1, cz - 1],
               [cx + 1, cy, cz],
               [cx - 1, cy, cz],
               [cx + 1, cy, cz + 1],
               [cx + 1, cy, cz - 1],
               [cx - 1, cy, cz + 1],
               [cx - 1, cy, cz - 1],
               [cx + 1, cy + 1, cz],
               [cx + 1, cy - 1, cz],
               [cx - 1, cy + 1, cz],
               [cx - 1, cy - 1, cz],
               [cx + 1, cy + 1, cz + 1],
               [cx + 1, cy + 1, cz - 1],
               [cx + 1, cy - 1, cz + 1],
               [cx + 1, cy - 1, cz - 1],
               [cx - 1, cy + 1, cz + 1],
               [cx - 1, cy + 1, cz - 1],
               [cx - 1, cy - 1, cz + 1],
               [cx - 1, cy - 1, cz - 1]]

  return [n for n in neighbors if n[0] >= 0 and n[1] >= 0 and n[2] >= 0]

def is_well_connected(candidate, blob_candidates, use_considered_candidates=False):
  global considered_candidates
  neighbors = get_neighbours(candidate)

  l = len([n for n in neighbors if (n[0], n[1], n[2]) in blob_candidates and str(n) not in considered_candidates])
  considered_candidates = considered_candidates + [str(n) for n in neighbors]
  considered_candidates.append(str(candidate))

  return l >= 6

def post_prune(blob_candidates):
  # blob_candidates are separated according to the scale T. So put all the candidates into one list
  all_candidates = []
  for sigma in blob_candidates.keys():
    all_candidates += blob_candidates[sigma]

  all_candidates = [c for c in all_candidates if not math.isnan(c[2]) and c[2] > 0.0 and c[3] > 0.0 and c[4] > 0.0]
  candidates_as_list = []

  for c in all_candidates:
    data_point = []
    data_point.extend(list(c[0]))
    data_point.extend([c[2], c[3], c[4]])
    candidates_as_list.append(data_point)

  if len(candidates_as_list) == 0:
    return []
  model = GaussianMixture(n_components=2, covariance_type='full')
  model.fit(candidates_as_list)
  class_labels = model.predict(candidates_as_list)
  scores = model.score_samples(np.array(candidates_as_list))

  avg_scores = np.zeros(len(np.unique(class_labels)))

  for i, x in enumerate(candidates_as_list):
    avg_scores[class_labels[i]] += scores[i]

  for i,avg_score in enumerate(avg_scores):
    n = len([x for x in class_labels if x == i])
    avg_scores[i] = avg_score/float(n)

  blob_class_index = list(avg_scores).index(max(avg_scores))
  blobs = [b for i,b in enumerate(candidates_as_list) if class_labels[i] == blob_class_index]

  return [[b[2], b[1], b[0]] for b in blobs]
