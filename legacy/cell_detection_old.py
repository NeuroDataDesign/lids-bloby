import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import progressbar
from image_processing import (
    ImageStack,
    read_tif
)
from detectors import (
    DoG,
    find_negative_curvative_points,
    blob_descriptors,
    post_prune,
    is_well_connected
)

#REMOVE THIS CODE LATER
import sys
sys.path.append('../clarity-f17s18/src/util/')
from ImageDrawer import ImageDrawer
import tifffile as tiff

class BlobDetector():
    print_level = 1

    @classmethod
    def detect_3d_blobs(cls, fname, batch_process=False, inverted=0, output_dir='./output/'):
        # Read in images.
        # If batch is true then image is broken up for faster processing
        print_level = cls.print_level
        img_stack = read_tif(fname, batch=batch_process, print_level=print_level)

        # Compute SIFT features
        DoG_stack = []
        detected_blobs = []

        for i in range(img_stack.stack_size):
            if print_level:
                if img_stack.stack_size == 1:
                    print("Computing DoG for image")
                else:
                    print("Computing DoG for image {}".format(i+1))
            DoG_stack = DoG(img_stack.images[i], dark=inverted, print_level=print_level)

            # Find concave points
            U = set()
            bar = progressbar.ProgressBar()
            concave_point_bar = bar(DoG_stack)
            if not print_level:
                concave_point_bar = DoG_stack
            else:
                print("Computing concave points")
            for sigma, DoG_img in concave_point_bar:
                indices = find_negative_curvative_points(DoG_img)
                for idx in range(indices.shape[0]):
                     U.add(tuple(indices[idx,:].astype(int)))
            if print_level:
                print("{} concave points found".format(len(U)))

            # Compute blob descriptors
            # TODO: calculating the blob descriptors is taking way to long. We need to trunate U

            stack_iter = zip(DoG_stack, img_stack.images)
            if print_level:
                bar = progressbar.ProgressBar()
                stack_iter = bar([x for x in stack_iter])
                print("Computing blob descriptors")
            blob_candidates_T = {}
            for (sigma, DoG_img), intensity_img in stack_iter:
                blob_candidates_T[sigma] = blob_descriptors(DoG_img, intensity_img, sigma, U)

            # Auto post-pruning using GMM
            detected_blobs = post_prune(blob_candidates_T)
            outfile_path = output_dir + 'detected_blob_centers_stack_{}.csv'.format(i+1)
            print("Writing detected blobs to {} ...".format(outfile_path))

            outfile = open(outfile_path, 'w')
            for blob in detected_blobs:
              outfile.write(','.join(str(x) for x in blob) + '\n')

            outfile.close()

        print("Done")


if __name__ == "__main__":
    file_path = './img/blurred_147_cells.tif'
    BlobDetector.detect_3d_blobs(file_path, batch_process=True)
