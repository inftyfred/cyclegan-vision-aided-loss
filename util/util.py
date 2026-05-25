"""This module contains simple helper functions"""

from __future__ import print_function
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import torch.distributed as dist
import os


def tensor2im(input_image, imtype=np.uint8, min_val=None, max_val=None):
    """Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
        min_val (float)      --  for 16-bit: minimum pixel value (original range)
        max_val (float)      --  for 16-bit: maximum pixel value (original range)

    For standard 8-bit (min/max None): [-1, 1] → [0, 255] uint8 (3-channel RGB)
    For 16-bit (min/max provided): [-1, 1] → [min_val, max_val] uint16 (1-channel grayscale)
    """
    if isinstance(input_image, np.ndarray):
        return input_image

    image_tensor = input_image.data if isinstance(input_image, torch.Tensor) else input_image
    image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array

    if min_val is not None and max_val is not None:
        # 16-bit: [-1, 1] → [min_val, max_val], single channel
        image_numpy = (image_numpy + 1.0) / 2.0  # [0, 1]
        image_numpy = image_numpy * (max_val - min_val) + min_val
        # (C, H, W) → (H, W) for single channel
        if image_numpy.shape[0] == 1:
            image_numpy = image_numpy[0]
        else:
            image_numpy = np.transpose(image_numpy, (1, 2, 0))
        return image_numpy.astype(np.uint16)
    else:
        # Standard 8-bit: [-1, 1] → [0, 255]
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)


def diagnose_network(net, name="network"):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


# initialize ddp
def init_ddp():
    # Initialize DDP if LOCAL_RANK is set
    is_ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1

    if is_ddp:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")
    print(f"Initialized with device {device}")
    return device


# cleanup ddp
def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
        aspect_ratio (float)      -- aspect ratio (ignored for 16-bit single-channel)
    """

    if image_numpy.ndim == 2:
        # 16-bit single-channel image
        image_pil = Image.fromarray(image_numpy)
        image_pil.save(image_path)
    else:
        image_pil = Image.fromarray(image_numpy)
        h, w, _ = image_numpy.shape

        if aspect_ratio > 1.0:
            image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
        if aspect_ratio < 1.0:
            image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
        image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print("shape,", x.shape)
    if val:
        x = x.flatten()
        print("mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f" % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)
