"""Calculate LPIPS between CycleGAN input and output images.

Usage:
    # Single-direction test (A → B)
    python metric/calculate_lpips_cyclegan.py --dataroot ./datasets/my_data/testA \\
        --name my_experiment --model test --direction AtoB

    # Two-direction test (A ↔ B)
    python metric/calculate_lpips_cyclegan.py --dataroot ./datasets/my_data \\
        --name my_experiment --model cycle_gan
"""

import json
import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from options.test_options import TestOptions
from data import create_dataset
from models import create_model


def to_3ch(tensor):
    """Expand (1, H, W) grayscale to (3, H, W) RGB by repeating channels."""
    if tensor.shape[1] == 1:
        return tensor.repeat(1, 3, 1, 1)
    return tensor


if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True

    # Load training config
    config_path = Path(opt.checkpoints_dir) / opt.name / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        for key in ("input_nc", "output_nc", "bit_depth",
                     "min_A", "max_A", "min_B", "max_B"):
            if key in config and config[key] is not None:
                setattr(opt, key, config[key])
        print(f"Loaded training config: input_nc={opt.input_nc}, "
              f"output_nc={opt.output_nc}, bit_depth={opt.bit_depth}")
    else:
        print(f"Note: {config_path} not found.")

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()

    # Import lpips here (optional dependency)
    try:
        import lpips
    except ImportError:
        print("Please install lpips: pip install lpips")
        sys.exit(1)

    lpips_model = lpips.LPIPS(net='vgg').to(opt.device)

    scores_A2B = []  # real_A → fake_B
    scores_B2A = []  # real_B → fake_A
    scores_cycle_A = []  # real_A → rec_A
    scores_cycle_B = []  # real_B → rec_B

    with torch.no_grad():
        for i, data in enumerate(dataset):
            if i >= opt.num_test:
                break
            model.set_input(data)
            model.test()

            visuals = model.get_current_visuals()

            if "real" in visuals and "fake" in visuals:
                # Single-direction test model
                real = to_3ch(visuals["real"])
                fake = to_3ch(visuals["fake"])
                scores_A2B.append(lpips_model(real, fake).item())
            else:
                # Two-direction cycle_gan model
                if "real_A" in visuals and "fake_B" in visuals:
                    real_A = to_3ch(visuals["real_A"])
                    fake_B = to_3ch(visuals["fake_B"])
                    scores_A2B.append(lpips_model(real_A, fake_B).item())
                if "real_B" in visuals and "fake_A" in visuals:
                    real_B = to_3ch(visuals["real_B"])
                    fake_A = to_3ch(visuals["fake_A"])
                    scores_B2A.append(lpips_model(real_B, fake_A).item())
                if "real_A" in visuals and "rec_A" in visuals:
                    real_A = to_3ch(visuals["real_A"])
                    rec_A = to_3ch(visuals["rec_A"])
                    scores_cycle_A.append(lpips_model(real_A, rec_A).item())
                if "real_B" in visuals and "rec_B" in visuals:
                    real_B = to_3ch(visuals["real_B"])
                    rec_B = to_3ch(visuals["rec_B"])
                    scores_cycle_B.append(lpips_model(real_B, rec_B).item())

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{opt.num_test} images")

    print("\n" + "=" * 60)
    print("LPIPS Results (lower = more similar)")
    print("=" * 60)

    for name, scores in [("A→B (real vs fake_B)", scores_A2B),
                          ("B→A (real vs fake_A)", scores_B2A),
                          ("Cycle A (real vs rec_A)", scores_cycle_A),
                          ("Cycle B (real vs rec_B)", scores_cycle_B)]:
        if scores:
            print(f"\n  {name}:")
            print(f"    Mean:   {np.mean(scores):.4f}")
            print(f"    Std:    {np.std(scores):.4f}")
            print(f"    Min:    {np.min(scores):.4f}")
            print(f"    Max:    {np.max(scores):.4f}")
            print(f"    Count:  {len(scores)}")
