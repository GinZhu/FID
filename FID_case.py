"""
Copyright notice:

This code should be used for the MR-Recon-GAN project only. 
If you want to use this code for other purposes, please contact the author for permission first.

Author: Jin Zhu jin.zhu@cl.cam.ac.uk zhujin1121@gmail.com
Date: October 09 2020

"""

from glob import glob
from os.path import join
from skimage import io
import numpy as np

from fid import FID
from utils import RandomCrop


gpu_id = 0
# ## batch size could be bigger if you have enough GPU memory.
batch_size = 4

root_folder = '/Users/Jin/Insync/0_ResearchWork/recon_MR_GAN/Knee'
gt_folder = glob(join(root_folder, 'GT*'))[0]
recon_folder = glob(join(root_folder, 'Recon*'))[0]
refine_folder = glob(join(root_folder, 'Refine*'))[0]

gt_paths = glob(join(gt_folder, '*.tif'))
recon_paths = glob(join(recon_folder, '*.tif'))
refine_paths = glob(join(refine_folder, '*.tif'))

rc = RandomCrop(256, 0)

fid = FID(gpu_id=gpu_id, batch_size=batch_size)

gt_imgs = [io.imread(_) for _ in gt_paths]
gt_imgs = [_/255 for _ in gt_imgs]
gt_imgs = np.array(gt_imgs)
gt_imgs = gt_imgs[:, :, :, np.newaxis]

recon_imgs = [io.imread(_) for _ in recon_paths]
recon_imgs = [_/255 for _ in recon_imgs]
recon_imgs = np.array(recon_imgs)
recon_imgs = recon_imgs[:, :, :, np.newaxis]

refine_imgs = [io.imread(_) for _ in refine_paths]
refine_imgs = [_/255 for _ in refine_imgs]
refine_imgs = np.array(refine_imgs)
refine_imgs = refine_imgs[:, :, :, np.newaxis]

# ## recon
print('Folder {} Recon GAN FID: {:.4}'.format(
    root_folder, fid(recon_imgs, gt_imgs)
))

# ## refine
print('Folder {} Refine GAN FID: {:.4}'.format(
    root_folder, fid(refine_imgs, gt_imgs)
))
