import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import csv
import progressbar
from tifffile import imread

### TODO: SOO WEIRD THIS NEEDS TO BE IMPORTED TO AVOID NaN
from Grapher import Grapher

from image_processing import (
    ImageException,
    raster_3d_generator,
    DoG,
    img_3d_hessian,
    find_concave_points,
    find_connected_component,
    post_prune,
    save_tif,
    regional_blobness,
    regional_flatness,
    BlobCandidate,
    normalize_image,
    tol_check,
    draw_centers,
    write_csv,
    save_tif,
    draw_square
)

INTENSITY_THRESHOLD = 0.5
DOG_SCALES = 1


class BlobDetector():

    @classmethod
    def detect_3d_blobs(cls, orig_img, scale, dark_blobs=1):
        # IMPORTANT: This method assumes the image comes in Z,Y,X format
        if len(orig_img.shape) != 3:
            raise ImageException("Invalid image shape")

        img = normalize_image(orig_img)

        if not dark_blobs:
            img = 1 - img

        print("Computing DoG at scales {}".format(np.linspace(scale, scale+10, DOG_SCALES)))
        dog_stack = DoG(img, sigma=scale, scales=DOG_SCALES)

        for sigma, img_dog in dog_stack:
            print("Processing for scale: {}\n".format(sigma))
            img_hessian = img_3d_hessian(img_dog)
            print("Computing hessian at each pixel")
            concave_points = find_concave_points(img_hessian)
            print("{} concave points found\n".format(len(concave_points)))

            connected_components = []
            concave_points_cpy = set(concave_points)

            bar = (progressbar.ProgressBar())(iter(concave_points))
            print("Finding connected components")
            for c in bar:
                cc = find_connected_component(c, concave_points_cpy, img_dog)
                if len(cc) >= 6:
                    concave_points_cpy -= set(cc+[c])
                    connected_components.append((c, cc))
            print("{} connected components found\n".format(len(connected_components)))

            print("Finding blob desciptors for connected components")
            blob_candidates = []
            bar = (progressbar.ProgressBar())(iter(connected_components))
            for center, cc in bar:
                # import pdb; pdb.set_trace()
                regional_hessian = np.empty(img_hessian.shape[:2])
                average_intensity = 0
                for i, j, k in cc:
                    regional_hessian += img_hessian[:, :, i, j, k]
                    regional_hessian = tol_check(np.nan_to_num(regional_hessian))
                    average_intensity += img[i, j, k]

                blobness = regional_blobness(regional_hessian)
                flatness = regional_flatness(regional_hessian)
                average_intensity /= len(cc)
                blob_candidates.append(
                    BlobCandidate(
                        center,
                        blobness,
                        flatness,
                        average_intensity
                    )
                )
            print("Done computing blob descriptors\n")

            z_max, y_max, x_max = img.shape

            print("Running post pruning")
            blob_candidates = [b for b in blob_candidates if b[0][0] < z_max and b[0][1] < y_max and b[0][2] < x_max]
            detected_blobs = post_prune(blob_candidates)
            print("{} blobs are detected".format(len(detected_blobs)))
            return detected_blobs

    @classmethod
    def draw_blobs(cls, orig_img, blobs, output_path):
        for b in detected_blobs:
            draw_square(orig_img, b[2], b[1], b[0], 2, [255, 0, 0], copy=False)
        #   orig_img[b[0], b[1], b[2], :] = [255, 0, 0]
        # drawn_img = draw_centers(orig_img, blobs)
        save_tif(orig_img, output_path)

    @classmethod
    def save_centers(cls, centers, output_path):
        write_csv(centers, output_path)

if __name__ == "__main__":
    IMG_DIR = './img/'
    CENTERS_DIR = './centers/'

    orig_img = imread('./img/subvolumes-normalized/reg.tiff')
    #orig_img = imread('./img/s3617_cutout_normalized.tif')
    shape_z, shape_y, shape_x, _ = orig_img.shape

    overlay_img = imread('./img/subvolumes/reg.tiff')
    #overlay_img = imread('./img/s3617_cutout.tif')

    detected_blobs = []
    # image_slices = []
    #
    # index = 0
    # batch_length_y = int(shape_y/100)
    # batch_length_x = int(shape_x/100)
    #
    # for i in range(0, batch_length_x):
    #     cutout_start_x = (i*100)
    #     cutout_end_x = cutout_start_x + 100
    #     for j in range(0,batch_length_y):
    #         print("Processing slice {}".format(index))
    #         cutout_start_y = (j*100)
    #         cutout_end_y = cutout_start_y + 100
    #
    #         print('X={}-{} and Y={}-{}'.format(cutout_start_x, cutout_end_x, cutout_start_y, cutout_end_y))

    #         grey_img = orig_img[:, cutout_start_y:cutout_end_y, cutout_start_x:cutout_end_x, 0]
    #         blobs = BlobDetector.detect_3d_blobs(grey_img, 0.5, dark_blobs=1)
    #         blobs = [[b[0], b[1] + cutout_start_y, b[2] + cutout_start_x] for b in blobs]
    #         detected_blobs.extend(blobs)
    #         print('{} blobs detected so far'.format(len(detected_blobs)))
    #         print("======================================")
    #         index += 1
    #
    #         if (index + 1) % 10 == 0:
    #             BlobDetector.save_centers(detected_blobs, "./img/subvolumes-predicted/cell_detection_9")
    #             BlobDetector.draw_blobs(overlay_img, detected_blobs, "subvolumes-predicted/cell_detection_9.tiff")
    #             print("NOTE: Centers are written in (z,y,x)")

    cutout_start_x = 0
    cutout_end_x = 540
    cutout_start_y = 0
    cutout_end_y = 640

    # cutout_start_x = 650
    # cutout_end_x = 800
    # cutout_start_y = 350
    # cutout_end_y = 500

    grey_img = orig_img[:, cutout_start_y:cutout_end_y, cutout_start_x:cutout_end_x, 0]
    detected_blobs = BlobDetector.detect_3d_blobs(grey_img, 0.5, dark_blobs=1)

    # with open('output_centers.csv') as csv_file:
    #     reader = csv.reader(csv_file)
    #     centroids = list(reader)
    #     detected_blobs = [[int(c[0]), int(c[1]), int(c[2])] for c in centroids]
    #
    # BlobDetector.save_centers(detected_blobs, "output_centers")
    # BlobDetector.draw_blobs(overlay_img[:, cutout_start_y:cutout_end_y, cutout_start_x:cutout_end_x], detected_blobs, "drawn_output.tif")

    BlobDetector.save_centers(detected_blobs, "./img/subvolumes-predicted/reg")
    BlobDetector.draw_blobs(overlay_img, detected_blobs, "subvolumes-predicted/reg.tiff")

    print("NOTE: Centers are written in (z,y,x)")
