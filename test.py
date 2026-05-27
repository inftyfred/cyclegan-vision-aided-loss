"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import os
import json
import shutil
import time
from pathlib import Path
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
from datetime import datetime

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.

    # Load training config.json — override critical params from training
    config_path = Path(opt.checkpoints_dir) / opt.name / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        for key in ("input_nc", "output_nc", "bit_depth",
                     "min_A", "max_A", "min_B", "max_B"):
            if key in config and config[key] is not None:
                setattr(opt, key, config[key])
        print(f"Loaded training config from {config_path}: "
              f"input_nc={opt.input_nc}, output_nc={opt.output_nc}, "
              f"bit_depth={opt.bit_depth}"
              + (f", min_A={opt.min_A}, max_A={opt.max_A}"
                 f", min_B={opt.min_B}, max_B={opt.max_B}"
                 if getattr(opt, "bit_depth", 8) == 16 else ""))
    else:
        print(f"Note: {config_path} not found, using command-line defaults.")

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    # create a website
    if opt.direction == "AtoB":
        dir_suffix = f"{opt.domainA}2{opt.domainB}"
    else:
        dir_suffix = f"{opt.domainB}2{opt.domainA}"
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    #base_name = f"{opt.phase}_{opt.epoch}" + f"_{time_str}"
    web_dir = Path(opt.results_dir) / opt.name / f"{opt.phase}_{opt.epoch}"  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = Path(f"{web_dir}_iter{opt.load_iter}")
    web_dir = Path(f"{web_dir}"f"_{dir_suffix}"f"_{time_str}")
    print(f"creating web directory {web_dir}")
    webpage = html.HTML(web_dir, f"Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}", opt=opt)
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    # Initialize timing variables
    inference_times = []
    use_cuda = torch.cuda.is_available()

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader

        # Measure inference time
        if use_cuda:
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        model.test()  # run inference

        if use_cuda:
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        # Calculate inference time in milliseconds
        inference_time_ms = (end_time - start_time) * 1000.0
        inference_times.append(inference_time_ms)

        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths

        # Print inference time for each image
        print(f"Image {i:04d}: {img_path} - Inference time: {inference_time_ms:.2f} ms")

        if i % 5 == 0:  # save images to an HTML file (original behavior)
            print(f"processing ({i:04d})-th image... {img_path}")
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    # Print timing statistics
    if inference_times:
        avg_time = sum(inference_times) / len(inference_times)
        min_time = min(inference_times)
        max_time = max(inference_times)
        print("\n" + "="*60)
        print("Inference Time Statistics:")
        print(f"  Number of images processed: {len(inference_times)}")
        print(f"  Average inference time: {avg_time:.2f} ms")
        print(f"  Minimum inference time: {min_time:.2f} ms")
        print(f"  Maximum inference time: {max_time:.2f} ms")
        print(f"  Total inference time: {sum(inference_times):.2f} ms")
        print("="*60)

    webpage.save()  # save the HTML

    # Also save to a fixed latest_* directory for easy frontend access
    latest_dir = Path(opt.results_dir) / opt.name / f"test_latest" 
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(web_dir, latest_dir)
    print(f"Latest results also saved to {latest_dir}")
