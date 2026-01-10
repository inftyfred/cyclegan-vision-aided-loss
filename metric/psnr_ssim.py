"""
# > Script for measuring quantitative performances in terms of
#    - Structural Similarity Metric (SSIM)
#    - Peak Signal to Noise Ratio (PSNR)
# > Maintainer: https://github.com/xahidbuffon
"""
## python libs
import numpy as np
from PIL import Image
from glob import glob
from os.path import join
from ntpath import basename
## local libs
from imqual_utils import getSSIM, getPSNR
import cv2
import os

## compares avg ssim and psnr
def SSIMs_PSNRs(gtr_dir, gen_dir, im_res=(256, 256)):
    """
        - gtr_dir contain ground-truths
        - gen_dir contain generated images
    """
    gtr_paths = sorted(glob(join(gtr_dir, "*.*")))
    gen_paths = sorted(glob(join(gen_dir, "*.*")))
    ssims, psnrs = [], []
    for gtr_path, gen_path in zip(gtr_paths, gen_paths):
        gtr_f = basename(gtr_path).split('.')[0]
        gen_f = basename(gen_path).split('.')[0]
        
        # read images from two datasets
        r_im = Image.open(gtr_path).resize(im_res)
        g_im = Image.open(gen_path).resize(im_res)

        # get ssim on RGB channels
        ssim = getSSIM(np.array(r_im), np.array(g_im))
        ssims.append(ssim)
        # get psnt on L channel (SOTA norm)
        r_im = r_im.convert("L"); g_im = g_im.convert("L")
        psnr = getPSNR(np.array(r_im), np.array(g_im))
        psnrs.append(psnr)
    return np.array(ssims), np.array(psnrs)

def get_dirImage(real_dir, fake_dir):
    real_image_paths = sorted(glob(os.path.join(real_dir, "*.*")))
    fake_image_paths = sorted(glob(os.path.join(fake_dir, "*.*")))
    return real_image_paths, fake_image_paths

"""
Get datasets from
 - http://irvlab.cs.umn.edu/resources/euvp-dataset
 - http://irvlab.cs.umn.edu/resources/ufo-120-dataset
"""
gtr_dir = r".\results\seprated\fakeA"

## generated im paths
gen_dir = r".\results\seprated\realA"



real_paths, fake_paths = get_dirImage(gtr_dir, gen_dir)
print(len(real_paths))


print("Real dir:", gtr_dir)
print("Fake dir:", gen_dir)
print("Real found:", len(real_paths))
print("Fake found:", len(fake_paths))
### compute SSIM and PSNR
SSIM_measures, PSNR_measures = SSIMs_PSNRs(gtr_dir, gen_dir)
print ("SSIM on {0} samples".format(len(SSIM_measures)))
print ("Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures)))

print ("PSNR on {0} samples".format(len(PSNR_measures)))
print ("Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures)))


