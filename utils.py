"""
Copyright notice:

This code should be used for the MR-Recon-GAN project only.
If you want to use this code for other purposes, please contact the author for permission first.

Author: Jin Zhu jin.zhu@cl.cam.ac.uk zhujin1121@gmail.com
Date: October 09 2020

"""

import cv2


# ## RandomCrop
class RandomCrop(object):

    def __init__(self, size, margin=0):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, (list, tuple)):
            if all(isinstance(_, int) for _ in size):
                self.size = size
            else:
                raise TypeError('Crop size should be int, list(int) or tuple(int)')
        else:
            raise TypeError('Crop size should be int, list(int), or tuple(int)')

        if isinstance(margin, int):
            self.margin = (margin, margin)
        elif isinstance(margin, (list, tuple)):
            if all(isinstance(_, int) for _ in margin):
                self.margin = margin
            else:
                raise TypeError('Crop margin should be int, list(int) or tuple(int)')
        else:
            raise TypeError('Crop margin should be int, list(int), or tuple(int)')

    def __call__(self, in_img):
        ori_H, ori_W = in_img.shape[:2]
        x_top_left = np.random.randint(
            self.margin[0], ori_H - self.size[0] - self.margin[0]
        )
        y_top_left = np.random.randint(
            self.margin[1], ori_W - self.size[1] - self.margin[1]
        )

        return in_img[x_top_left:x_top_left + self.size[0], y_top_left:y_top_left + self.size[1]]


def resize(data):
    """
    data:
      [img, size, interpolation_method, blur_method, blur_kernel, blur_sigma]
    cv2 coordinates:
      [horizontal, vertical], which is different as numpy array image.shape
      'cubic': cv2.INTER_LINEAR
      'linear': cv2.INTER_CUBIC
      'nearest' or None(default): cv2.INTER_NEAREST
    Caution: cubic interpolation may generate values out of original data range (e.g. negative values)

    """
    data += [None, ] * (6 - len(data))

    img, size, interpolation_method, blur_method, blur_kernel, blur_sigma = data

    #
    if interpolation_method == 'nearest':
        interpolation_method = cv2.INTER_NEAREST
    elif interpolation_method is None or interpolation_method == 'cubic':
        interpolation_method = cv2.INTER_CUBIC
    elif interpolation_method == 'linear':
        interpolation_method = cv2.INTER_LINEAR
    else:
        raise ValueError('cv2 Interpolation methods: None, nearest, cubic, linear')

    if blur_kernel is None:
        blur_kernel = 5
    if blur_sigma is None:
        blur_sigma = 0

    # calculate the output size
    if isinstance(size, (float, int)):
        size = [size, size]
    if not isinstance(size, (list, tuple)):
        raise TypeError('The input Size of LR image should be (float, int, list or tuple)')
    if isinstance(size[0], float):
        size = int(img.shape[0] * size[0]), int(img.shape[1] * size[1])
    if size[0] <= 0 or size[1] <= 0:
        raise ValueError('Size of output image should be positive')

    # resize the image
    if size[0] == img.shape[0] and size[1] == img.shape[1]:
        output_img = img
    else:
        # opencv2 is [horizontal, vertical], so the output_size should be reversed
        size = size[1], size[0]
        output_img = cv2.resize(img, dsize=size, interpolation=interpolation_method)

    # blur the image if necessary
    if blur_method == 'gaussian':
        output_img = cv2.GaussianBlur(output_img, (blur_kernel, blur_kernel), blur_sigma)
    else:
        # todo: add more blur methods
        pass
    if img.ndim != output_img.ndim:
        output_img = output_img[:, :, np.newaxis]
    return output_img

