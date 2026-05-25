"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""

import random
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == "resize_and_crop":
        new_h = new_w = opt.load_size
    elif opt.preprocess == "scale_width_and_crop":
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {"crop_pos": (x, y), "flip": flip}


class ToTensor16Bit:
    """Convert PIL I;16 image to float32 tensor, preserving original pixel values (0-65535)."""

    def __call__(self, pic):
        arr = np.array(pic, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        tensor = torch.from_numpy(arr.transpose((2, 0, 1)))
        return tensor


class MinMaxNormalize:
    """Normalize a float32 tensor from [min_val, max_val] to [-1, 1], with optional hard clipping."""

    def __init__(self, min_val, max_val, clamp=True):
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.range = self.max_val - self.min_val
        self.clamp = clamp

    def __call__(self, tensor):
        if self.clamp:
            tensor = tensor.clamp(self.min_val, self.max_val)
        if self.range < 1e-8:
            return torch.zeros_like(tensor)
        return -1.0 + 2.0 * (tensor - self.min_val) / self.range


def get_transform(opt, params=None, grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True, bit_depth=8, min_val=None, max_val=None):
    transform_list = []
    if bit_depth == 16:
        # For 16-bit I;16 images, use BILINEAR (BICUBIC can be unreliable on I;16 in some PIL versions)
        if method == transforms.InterpolationMode.BICUBIC:
            method = transforms.InterpolationMode.BILINEAR
    else:
        if grayscale:
            transform_list.append(transforms.Grayscale(1))

    if "resize" in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif "scale_width" in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))

    if "crop" in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params["crop_pos"], opt.crop_size)))

    if opt.preprocess == "none":
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params["flip"]:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params["flip"])))

    if convert:
        if bit_depth == 16:
            transform_list.append(ToTensor16Bit())
            if min_val is None or max_val is None:
                raise ValueError("min_val and max_val are required when bit_depth=16")
            transform_list.append(MinMaxNormalize(min_val, max_val, clamp=False))
        else:
            transform_list += [transforms.ToTensor()]
            if grayscale:
                transform_list += [transforms.Normalize((0.5,), (0.5,))]
            else:
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __transforms2pil_resize(method):
    mapper = {
        transforms.InterpolationMode.BILINEAR: Image.BILINEAR,
        transforms.InterpolationMode.BICUBIC: Image.BICUBIC,
        transforms.InterpolationMode.NEAREST: Image.NEAREST,
        transforms.InterpolationMode.LANCZOS: Image.LANCZOS,
    }
    return mapper[method]


def __make_power_2(img, base, method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, "has_printed"):
        print("The image size needs to be a multiple of 4. " "The loaded image size was (%d, %d), so it was adjusted to " "(%d, %d). This adjustment will be done to all images " "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
