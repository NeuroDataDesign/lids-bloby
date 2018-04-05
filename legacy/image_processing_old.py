import numpy as np
from collections import namedtuple
from tifffile import imsave, imread
from collections import namedtuple

class ImageException(Exception):
    pass

class ImageStack(namedtuple('ImageStack', ['images', 'x_range', 'y_range', 'z_range', 'stack_size'])):
    pass

class BlobCandidate(namedtuple('BlobCandidate', ['center', 'sigma', 'blobness', 'flatness', 'avg_int'])):
    pass

def normalize_image(img):
    img_max = np.amax(img)
    img_min = np.amin(img)
    img = np.subtract(img, img_min)
    img = np.divide(img, img_max - img_min)
    return img

def read_tif(fname, normalize=True, batch=False, batch_size=10, print_level=0):
    img = imread(fname)
    z_range, y_range, x_range, chans = img.shape
    if print_level:
        print("Reading tif with shape: {}".format(img.shape))

    # TODO: more robust rgb to grey conversion
    if chans == 3:
        img = img[:, :, :, 0]
    if normalize:
        img = normalize_image(img)
        if np.amax(img) != 1.0:
            raise ImageException("Normalized image value not between 0 and 1")

    if batch:
        image_stack = []
        for i in range(batch_size):
            image_stack.append(img[:, (i*100):((i+1)*100), (i*100):((i+1)*100)])
        if print_level:
            print("Splitting image into batch of {} images of size {}".format(len(image_stack), image_stack[0].shape))
        return ImageStack(image_stack, 100, 100, 100, batch_size)
    else:
        return ImageStack([img], z_range, y_range, x_range, 1)

def gradient_x(arr):
    return np.gradient(arr, axis=2)

def gradient_y(arr):
    return np.gradient(arr, axis=1)

def gradient_z(arr):
    return np.gradient(arr, axis=0)

def img_hessian_3d(img):
    '''
    Note that this method computes the image hessians.
    The output H will have shape same as img with an additional
    dimension that will hold the hessian computations.
    the last dimension will be 6 elements long and contain
    each of the upper triangular hessians.
    '''
    img_fx, img_fy, img_fz = gradient_x(img), gradient_y(img), gradient_z(img)
    img_fxy, img_fxz, img_fyz = gradient_y(img_fx), gradient_z(img_fx), gradient_z(img_fy)
    img_fxx, img_fyy, img_fzz = gradient_x(img_fx), gradient_y(img_fy), gradient_z(img_fz)
    H = np.zeros(tuple(list(img.shape)+[6]))
    H[:,:,:,0] = img_fxx
    H[:,:,:,1] = img_fyy
    H[:,:,:,2] = img_fzz
    H[:,:,:,3] = img_fxy
    H[:,:,:,4] = img_fxz
    H[:,:,:,5] = img_fyz
    # TODO: Very memory inefficient. Is it better to compute the hessians on the spot?
    return H

def format_H(H):
    fxx = H[0]
    fyy = H[1]
    fzz = H[2]
    fxy = H[3]
    fxz = H[4]
    fyz = H[5]
    return np.array([
        [fxx, fxy, fxz],
        [fxy, fyy, fyz],
        [fxz, fyz, fzz]
    ])

def principal_minors_3d(M):
    n,m = M.shape
    # We're going to assume M is square for now
    D_1 = M[0,0]
    D_2 = np.linalg.det(M[::2, ::2])
    D_3 = np.linalg.det(M)
    return D_1, D_2, D_3

def curvature(H):
    '''
    Returns
        0 if indefinite
        1 if positive definite
        2 if positive semi-definite
        3 if negative definite
        4 if negative semi-definite
    '''
    D_1, D_2, D_3 = principal_minors_3d(H)
    if D_1 < 0 and D_2 > 0 and D_3 < 0:
        # Negative definite
        return 3
    # TODO: Implement other cases
    return 0

def raster_3d_generator(img_shape):
    z_range, y_range, x_range = img_shape
    for i in range(z_range):
        for j in range(y_range):
            for k in range(x_range):
                yield (i, j, k)

def voxel_region_iter(z, y, x):
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                yield (i+z,j+y,k+x)


def block_principal_minors(H):
    D_1 = np.linalg.det(H[:2, :2])
    D_2 = np.linalg.det(H[1:3, 1:3])
    D_3 = np.linalg.det(H[0:3:2, 0:3:2])
    return D_1 + D_2 + D_3


def regional_blobness(H):
    det = np.linalg.det(H)
    # Note these are the 2x2 principal minors
    pm = block_principal_minors(H)
    return 3*np.abs(det)**(2.0/3)/pm


def regional_flatness(H):
    tr = np.trace(H)
    pm = block_principal_minors(H)
    return np.sqrt(tr**2 - 2*pm)


def grey_img_to_rgb(img):
    z_range, y_range, x_range = img.shape
    rgb_img = np.zeros((z_range, y_range, x_range, 3))
    img_iter = raster_3d_generator(img.shape)
    for i, j, k in img_iter:
        grey = img[i,j,k]*255
        set_rgb(rgb_img, k, j, i, grey, grey, grey)
    return rgb_img
