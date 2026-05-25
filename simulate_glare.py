"""Simulate strong light/glare effect on images via histogram operations."""

import os
import numpy as np
from PIL import Image
import argparse


def histogram_strong_light(img, brightness_factor=1.8, gamma=0.6, highlight_boost=0.3):
    """Apply strong light simulation using histogram-based operations.

    Args:
        img: PIL Image (RGB).
        brightness_factor: Linear brightness multiplier.
        gamma: Gamma correction value (<1 brightens midtones/shadows more).
        highlight_boost: Amount of histogram clipping to boost highlights.
    """
    arr = np.array(img, dtype=np.float32) / 255.0

    # 1) Gamma correction — compress dynamic range toward bright end
    arr = np.power(arr, gamma)

    # 2) Linear brightness scaling
    arr = arr * brightness_factor

    # 3) Highlight boost via histogram clipping —
    #    shift upper histogram bins to push more pixels toward saturation
    #    this simulates the "blown-out highlights" of strong light
    highlight_threshold = 1.0 - highlight_boost
    mask = arr > highlight_threshold
    arr[mask] = 1.0 - (1.0 - arr[mask]) * (1.0 - highlight_boost * 2)

    # 4) Slight warm color shift (strong sunlight tends toward warm tones)
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.05, 0, 1)
    arr[:, :, 1] = np.clip(arr[:, :, 1] * 1.02, 0, 1)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.95, 0, 1)

    arr = np.clip(arr, 0, 1)
    result = (arr * 255).astype(np.uint8)
    return Image.fromarray(result, mode='RGB')


def histogram_equalize_light(img, brightness_factor=1.5, gamma=0.7):
    """Simulate strong light via histogram equalization + gamma."""
    arr = np.array(img, dtype=np.float32) / 255.0

    result = np.zeros_like(arr)
    for c in range(3):
        channel = arr[:, :, c]
        hist, bins = np.histogram(channel.flatten(), bins=256, range=(0, 1))
        cdf = hist.cumsum()
        cdf_min = cdf[cdf > 0].min()
        cdf_max = cdf.max()
        if cdf_max > cdf_min:
            cdf_norm = (cdf - cdf_min) / (cdf_max - cdf_min)
        else:
            cdf_norm = np.zeros_like(cdf)
        bin_idx = np.clip((channel * 255).astype(int), 0, 255)
        result[:, :, c] = cdf_norm[bin_idx]

    result = np.power(result, gamma)
    result = result * brightness_factor
    result[:, :, 0] = np.clip(result[:, :, 0] * 1.05, 0, 1)
    result[:, :, 2] = np.clip(result[:, :, 2] * 0.93, 0, 1)

    result = np.clip(result, 0, 1)
    return Image.fromarray((result * 255).astype(np.uint8), mode='RGB')


def random_local_light(img, num_spots=3, max_intensity=0.9, min_radius=0.4, max_radius=0.6, seed=None):
    """Simulate random local strong light spots with gaussian falloff.

    Each spot is a 2D gaussian centered at a random location, simulating
    localized glare or light reflections.

    Args:
        img: PIL Image (RGB).
        num_spots: Number of random light spots.
        max_intensity: Peak brightness added at each spot center (0~1).
        min_radius: Minimum gaussian radius in pixels.
        max_radius: Maximum gaussian radius in pixels.
        seed: Random seed for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)

    arr = np.array(img, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]

    # Create coordinate grids
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)

    light_map = np.zeros((h, w), dtype=np.float32)
    
    for _ in range(num_spots):
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        radius = np.random.uniform(min_radius*min(h,w)/2, max_radius*min(h,w)/2)
        intensity = np.random.uniform(max_intensity * 0.8, max_intensity)

        # 2D gaussian: intensity decays from center outward
        gaussian = intensity * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * radius ** 2))
        light_map += gaussian

    light_map = np.clip(light_map, 0, 1)

    # Add light to image with warm tint (light spots are warm)
    warm_light = np.stack([
        light_map * 1.08,   # red boosted
        light_map * 1.0,    # green neutral
        light_map * 0.85,   # blue reduced → warm tint
    ], axis=-1)

    arr = arr + warm_light
    arr = np.clip(arr, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8), mode='RGB')


def center_radial_light(img, center=None, intensity=0.85, radius_ratio=0.4, falloff='gaussian'):
    """Simulate strong light radiating from a center point, decreasing outward.

    Args:
        img: PIL Image (RGB).
        center: (x, y) center point; if None, uses image center.
        intensity: Peak brightness added at center (0~1).
        radius_ratio: Radius as ratio of image diagonal size.
        falloff: 'gaussian' or 'linear' intensity decay.
    """
    arr = np.array(img, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]

    if center is None:
        cx, cy = w // 2, h // 2
    else:
        cx, cy = center
        cx = cx * w
        cy = cy * h

    diag = np.sqrt(h ** 2 + w ** 2)
    radius = radius_ratio * diag

    y, x = np.mgrid[0:h, 0:w].astype(np.float32)

    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    if falloff == 'gaussian':
        light_map = intensity * np.exp(-(dist ** 2) / (2 * radius ** 2))
    else:
        light_map = intensity * np.clip(1.0 - dist / radius, 0, 1)

    warm_light = np.stack([
        light_map * 1.08,
        light_map * 1.0,
        light_map * 0.85,
    ], axis=-1)

    arr = arr + warm_light
    arr = np.clip(arr, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8), mode='RGB')


def main():
    parser = argparse.ArgumentParser(description='Strong light simulation via histogram operations')
    parser.add_argument('--input', type=str, default='datasets/test',
                        help='Input image path or directory')
    parser.add_argument('--output', type=str, default='results/test_strong_light',
                        help='Output image directory')
    parser.add_argument('--method', type=str, default='random_local',
                        choices=['gamma_highlight', 'histogram_equalize', 'random_local', 'center_radial'],
                        help='Light simulation method')
    # gamma_highlight / histogram_equalize params
    parser.add_argument('--brightness', type=float, default=1.8,
                        help='Brightness factor (for gamma_highlight / histogram_equalize)')
    parser.add_argument('--gamma', type=float, default=0.6,
                        help='Gamma value, <1 brightens (for gamma_highlight / histogram_equalize)')
    parser.add_argument('--highlight_boost', type=float, default=0.3,
                        help='Highlight boost intensity (for gamma_highlight)')
    # random_local params
    parser.add_argument('--num_spots', type=int, default=3,
                        help='Number of random light spots (for random_local)')
    parser.add_argument('--max_intensity', type=float, default=0.9,
                        help='Peak intensity of each spot (for random_local / center_radial)')
    parser.add_argument('--min_radius', type=float, default=0.5,
                        help='Min gaussian radius in pixels (for random_local)')
    parser.add_argument('--max_radius', type=float, default=0.8,
                        help='Max gaussian radius in pixels (for random_local)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (for random_local)')
    # center_radial params
    parser.add_argument('--center_x', type=float, default=None,
                        help='Light center x coordinate (for center_radial, default=image center)')
    parser.add_argument('--center_y', type=float, default=None,
                        help='Light center y coordinate (for center_radial, default=image center)')
    parser.add_argument('--radius_ratio', type=float, default=0.4,
                        help='Radius as ratio of image diagonal (for center_radial)')
    parser.add_argument('--falloff', type=str, default='gaussian',
                        choices=['gaussian', 'linear'],
                        help='Intensity falloff type (for center_radial)')
    args = parser.parse_args()

    # Determine if input is a single file or a directory
    if os.path.isfile(args.input):
        # Single image mode
        single_file = os.path.basename(args.input)
        input_dir = os.path.dirname(args.input) or '.'
        filenames = [single_file]
    elif os.path.isdir(args.input):
        single_file = None
        input_dir = args.input
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        filenames = [f for f in os.listdir(input_dir)
                     if f.lower().endswith(exts)]
    else:
        print(f'Input path not found: {args.input}')
        return

    if not filenames:
        print(f'No images found in {input_dir}')
        return

    # Avoid overwriting existing output folders by adding numeric suffix
    out_dir = args.output
    if os.path.exists(out_dir):
        i = 1
        while os.path.exists(f"{out_dir}_{i}"):
            i += 1
        out_dir = f"{out_dir}_{i}"
    os.makedirs(out_dir)

    print(f'Processing {len(filenames)} images from {input_dir} -> {out_dir} (method={args.method})')

    for fname in filenames:
        path = os.path.join(input_dir, fname)
        img = Image.open(path).convert('RGB')

        if args.method == 'gamma_highlight':
            result = histogram_strong_light(img, args.brightness, args.gamma, args.highlight_boost)
        elif args.method == 'histogram_equalize':
            result = histogram_equalize_light(img, args.brightness, args.gamma)
        elif args.method == 'random_local':
            result = random_local_light(img, args.num_spots, args.max_intensity,
                                        args.min_radius, args.max_radius, args.seed)
        elif args.method == 'center_radial':
            center = None
            if args.center_x is not None and args.center_y is not None:
                center = (args.center_x, args.center_y)
            result = center_radial_light(img, center=center, intensity=args.max_intensity,
                                         radius_ratio=args.radius_ratio, falloff=args.falloff)

        out_path = os.path.join(out_dir, fname)
        result.save(out_path)
        print(f'  Saved: {out_path}')

    print('Done.')


if __name__ == '__main__':
    main()