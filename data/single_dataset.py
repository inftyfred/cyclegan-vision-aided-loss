import numpy as np
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image


class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc

        if opt.bit_depth == 16:
            if getattr(opt, "min_A", None) is not None and getattr(opt, "max_A", None) is not None:
                print(f"Using pre-set min/max from config: min_A={opt.min_A}, max_A={opt.max_A}")
            else:
                print("Scanning dataset for min/max values (16-bit mode)...")
                opt.min_A, opt.max_A = self._scan_dataset(self.A_paths)
                print(f"  min={opt.min_A}, max={opt.max_A}")
            self.transform = get_transform(opt, grayscale=(input_nc == 1),
                                           bit_depth=16, min_val=opt.min_A, max_val=opt.max_A)
        else:
            self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def _scan_dataset(self, paths):
        """Scan all images to find 1st and 99th percentile pixel values.

        Uses adaptive stride subsampling for memory efficiency.
        Robust against outlier pixels.

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
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        if self.opt.bit_depth == 16:
            A_img = Image.open(A_path)  # keep I;16 mode
        else:
            A_img = Image.open(A_path).convert("RGB")
        A = self.transform(A_img)
        return {"A": A, "A_paths": A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
