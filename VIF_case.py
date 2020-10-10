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
from sewar.full_ref import vifp

import argparse


parser = argparse.ArgumentParser(description='VIF Parameters & Data folder')
parser.add_argument('--case-folder', type=str, required=True, metavar='Folder',
                    help='In the case folder there should be three folders: GT*, Recon*, and Refine*')

args = parser.parse_args()

# ## folder
root_folder = args.case_folder
gt_folder = glob(join(root_folder, 'GT*'))[0]
recon_folder = glob(join(root_folder, 'Recon*'))[0]
refine_folder = glob(join(root_folder, 'Refine*'))[0]

gt_paths = glob(join(gt_folder, '*.tif'))

# ## VIF for recon images and refine images
vif_recon = []
vif_refine = []

for gtp in gt_paths:
    recon_p = gtp.replace(gt_folder, recon_folder).replace('gt', 'rec')
    refine_p = gtp.replace(gt_folder, refine_folder).replace('gt', 'ref')

    gt_img = io.imread(gtp)
    gt_img = gt_img[:, :, np.newaxis]/255
    recon_img = io.imread(recon_p)
    recon_img = recon_img[:, :, np.newaxis]/255
    refine_img = io.imread(refine_p)
    refine_img = refine_img[:, :, np.newaxis]/255

    vif_recon.append(vifp(gt_img, recon_img))
    vif_refine.append(vifp(gt_img, refine_img))

# ## recon
print('Folder {} Recon GAN VIF: {:.4}({:.2})'.format(
    root_folder, np.mean(vif_recon), np.std(vif_recon)
))

# ## refine
print('Folder {} Refine GAN VIF: {:.4}({:.2})'.format(
    root_folder, np.mean(vif_refine), np.std(vif_refine)
))

np.savez_compressed(
    join(root_folder, 'vif.npz'),
    recon=vif_recon,
    refine=vif_refine
)
