import os
import numpy as np
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.

    Alternatively, you can directly specify the paths to domain A and B images using
    --dataroot_A and --dataroot_B flags. If both are provided, they override the default
    dataroot/phaseA and dataroot/phaseB behavior.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--dataroot_A', type=str, default=None,
                           help='direct path to domain A images (e.g., /path/to/trainA). If specified, overrides the default dataroot/phaseA behavior.')
        parser.add_argument('--dataroot_B', type=str, default=None,
                           help='direct path to domain B images (e.g., /path/to/trainB). If specified, overrides the default dataroot/phaseB behavior.')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # Determine directories for domain A and B
        # If dataroot_A and dataroot_B are provided, use them directly
        # Otherwise, fall back to the original behavior (dataroot/phaseA, dataroot/phaseB)
        has_dataroot_A = opt.dataroot_A is not None
        has_dataroot_B = opt.dataroot_B is not None

        if has_dataroot_A and has_dataroot_B:
            self.dir_A = opt.dataroot_A
            self.dir_B = opt.dataroot_B
        elif not has_dataroot_A and not has_dataroot_B:
            self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")  # create a path '/path/to/data/trainB'
        else:
            raise ValueError('Both --dataroot_A and --dataroot_B must be specified together, or neither.')

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == "BtoA"
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image

        # 16-bit: pre-scan datasets for robust percentile-based normalization
        if opt.bit_depth == 16:
            if getattr(opt, "min_A", None) is not None and getattr(opt, "min_B", None) is not None \
               and getattr(opt, "max_A", None) is not None and getattr(opt, "max_B", None) is not None:
                global_min = min(opt.min_A, opt.min_B)
                global_max = max(opt.max_A, opt.max_B)
                print(f"Using pre-set min/max from config: global range=[{global_min:.0f}, {global_max:.0f}]")
            else:
                print("Scanning domain A (16-bit, 1st/99th percentile)...")
                p1_A, p99_A = self._scan_dataset(self.A_paths)
                print("Scanning domain B (16-bit, 1st/99th percentile)...")
                p1_B, p99_B = self._scan_dataset(self.B_paths)
                # Shared global range across both domains
                global_min = min(p1_A, p1_B)
                global_max = max(p99_A, p99_B)
                opt.min_A = opt.min_B = global_min
                opt.max_A = opt.max_B = global_max
                print(f"  Domain A P1/P99: {p1_A:.0f}/{p99_A:.0f}")
                print(f"  Domain B P1/P99: {p1_B:.0f}/{p99_B:.0f}")
                print(f"  Shared global range: [{global_min:.0f}, {global_max:.0f}]")

            self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1),
                                             bit_depth=16, min_val=global_min, max_val=global_max)
            self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1),
                                             bit_depth=16, min_val=global_min, max_val=global_max)
        else:
            self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
            self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def _scan_dataset(self, paths):
        """Scan all images to find 1st and 99th percentile pixel values.

        Uses adaptive stride subsampling for memory efficiency.
        The 1%/99% range is robust against outlier pixels that would
        stretch global min/max normalization.

        Parameters:
            paths (list of str) -- paths to images

        Returns:
            tuple[float, float]: (p1, p99) across all images
        """
        all_samples = []
        total = len(paths)
        for i, path in enumerate(paths):
            if i % 500 == 0:
                print(f"    Scanning 16-bit images: {i}/{total}")
            img = Image.open(path)
            arr = np.array(img, dtype=np.float32).ravel()
            # Adaptive stride: target ~2000 evenly-spaced samples per image
            stride = max(1, len(arr) // 2000)
            all_samples.append(arr[::stride])

        all_samples = np.concatenate(all_samples)
        p1, p99 = np.percentile(all_samples, [1, 99])
        print(f"    Scanned {total} images, P1={p1:.0f}, P99={p99:.0f} "
              f"(from {len(all_samples):,} samples)")
        return float(p1), float(p99)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        if self.opt.bit_depth == 16:
            A_img = Image.open(A_path)  # keep I;16 mode, no .convert("RGB")
            B_img = Image.open(B_path)
        else:
            A_img = Image.open(A_path).convert("RGB")
            B_img = Image.open(B_path).convert("RGB")
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
