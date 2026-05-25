# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Environment setup
conda env create -f environment.yml
pip install vision-aided-loss  # optional, for vision-aided discriminator
conda activate pytorch-img2img

# Dataset download
bash ./datasets/download_cyclegan_dataset.sh <dataset>  # maps, horse2zebra, mini, etc.
bash ./datasets/download_pix2pix_dataset.sh <dataset>    # facades, etc.

# Pretrained model download
bash ./scripts/download_cyclegan_model.sh <model>   # horse2zebra, maps, etc.
bash ./scripts/download_pix2pix_model.sh <model>    # facades_label2photo, etc.

# Train (CycleGAN)
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

# Train (pix2pix)
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

# Train with vision-aided loss (CycleGAN)
python train.py --dataroot ./datasets/maps --name maps_cv_cyclegan --model cycle_gan \
    --use_vision_aided_loss --cv_type clip --cv_output_type conv_multi_level \
    --cv_loss multilevel_sigmoid_s --cv_lambda 0.5 --cv_diffaug --cv_lr 0.0001 --cv_warmup_iter 2000

# Test (full CycleGAN, both directions)
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

# Test (one side only, using pretrained model)
python test.py --dataroot ./datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

# Multi-GPU DDP training (must use --norm syncbatch)
torchrun --nproc_per_node=4 train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --norm syncbatch

# Logging options (add any of these to train.py)
--use_wandb --wandb_project_name CycleGAN-and-pix2pix
--use_tensorboard

# Run pre-commit test suite
pytest scripts/test_before_push.py -v

# Lint
flake8 --ignore E501 .
```

## Documentation

Reference docs available in `docs/`:
- **tips.md** — Best practices for training/testing
- **qa.md** — Frequently asked questions
- **overview.md** — Code structure overview
- **datasets.md** — Dataset preparation guide
- **docker.md** — Docker usage

## Project Architecture

### Entry Points
- **train.py** — General-purpose training script for all models. Iterates epochs, calls `model.optimize_parameters()`, logs via `Visualizer`, and saves checkpoints. Supports `--continue_train` for resuming.
- **test.py** — General-purpose test script. Hard-codes `batch_size=1`, `num_threads=0`, `serial_batches=True`, `no_flip=True`. Reports per-image inference timing statistics.

### Options (`options/`)
- **base_options.py** — Shared CLI args (dataroot, model type, dataset_mode, network architecture, normalization, image dimensions, wandb/tensorboard flags, domain naming via `--domainA`/`--domainB`). Uses argparse with auto-gathering from model/dataset `modify_commandline_options` static methods.
- **train_options.py** — Training options (epochs, learning rate, GAN mode `[vanilla|lsgan|wgangp]`, `pool_size`, `lr_policy [linear|step|plateau|cosine]`, frequencies). Vision-aided loss args defined in CycleGANModel instead.
- **test_options.py** — Test options (`num_test`, `aspect_ratio`, `results_dir`, `phase`, `model_suffix` for one-sided generation, `--eval` mode).

### Models (`models/`)
- **base_model.py** — Abstract base class. Subclasses define four lists: `loss_names`, `model_names`, `visual_names`, `optimizers`. Key methods: `setup()` (weight init, optional checkpoint loading, DDP wrapping, auto-creates LR schedulers from `optimizers`), `save_networks()`/`load_networks()` (with `_get_model_filename()` substituting domainA/domainB in filenames — e.g., `net_G_A` → `net_G_A2B.pth`), `test()`, `update_learning_rate()`.
- **cycle_gan_model.py** — CycleGAN: two generators (G_A: A→B, G_B: B→A), two discriminators (D_A, D_B), cycle consistency loss (L1, weighted by `--lambda_A`/`--lambda_B`), identity loss (weighted by `--lambda_identity`). Integrates vision-aided discriminators (`cvD_A`, `cvD_B`) with frozen pretrained backbones and trainable decoder heads, separate optimizers (`optimizer_cvD_A`, `optimizer_cvD_B`), and warmup period (`--cv_warmup_iter`). Uses image pools (`fake_A_pool`, `fake_B_pool`) for discriminator training stability.
- **pix2pix_model.py** — pix2pix: one generator (U-Net), one conditional discriminator (concatenates input+output). L1 + GAN loss.
- **test_model.py** — Single-direction inference wrapper for CycleGAN (sets `dataset_mode=single`, uses `--model_suffix` to pick which generator to load, e.g., `--model_suffix _A` loads only G_A).
- **colorization_model.py** — RGB→Lab colorization model.
- **template_model.py** — Starting point for custom model implementations.
- **networks.py** — Network zoo: generators (ResNet 6/9-block, U-Net 128/256), discriminators (PatchGAN `basic`=70×70, `n_layers`, `pixel`), GAN loss functions (vanilla, LSGAN, WGAN-GP), norm layers (batch, instance, syncbatch, none), weight init methods (normal, xavier, kaiming, orthogonal), LR schedulers (linear, step, plateau, cosine).

### Data (`data/`)
- **base_dataset.py** — Abstract base dataset with transform pipelines (resize, crop, flip, normalization), `get_transform()` helper, and `__getitem__` contract.
- **unaligned_dataset.py** — Unpaired two-domain dataset (trainA/trainB dirs). Supports `--dataroot_A`/`--dataroot_B` overrides. Randomly samples domain B images per epoch.
- **aligned_dataset.py** — Paired image dataset (A,B pairs in same directory).
- **single_dataset.py** — Single-domain dataset for one-sided test mode.
- **colorization_dataset.py** — RGB→Lab (L, ab) conversion for colorization model.
- **template_dataset.py** — Starting point for custom dataset implementations.

### Model/Dataset Auto-Discovery
Models and datasets are loaded by convention:
- `--model cycle_gan` imports `models/cycle_gan_model.py` and instantiates `CycleGANModel` (case-insensitive match on ABC subclass in the module).
- `--dataset_mode unaligned` imports `data/unaligned_dataset.py` and instantiates `UnalignedDataset`.
- Both use `modify_commandline_options()` static methods to register their own CLI flags.

### Vision-Aided Loss (`vision_aided_loss/`)
Submodule from [vision-aided-gan](https://github.com/nupurkmr9/vision-aided-gan). Provides a `Discriminator` class wrapping frozen pretrained vision encoders (CLIP, DINO, Swin, VGG) with trainable decoder heads. Supports multi-level features, DiffAugment, and multiple loss types. Two independent instances (`cvD_A`, `cvD_B`) are created for each domain with separate optimizers. Activated by `--use_vision_aided_loss`. Key options:
- `--cv_type` — Model type: clip, dino, swin, vgg (combine with `+`)
- `--cv_output_type` — conv, conv_multi_level
- `--cv_loss` — sigmoid, multilevel_sigmoid_s, hinge
- `--cv_lambda` — Loss weight (default: 0.5)
- `--cv_diffaug` / `--no_cv_diffaug` — DiffAugment (default: True)
- `--cv_lr` — Decoder learning rate (default: 0.0002)
- `--cv_warmup_iter` — Iterations before applying cv loss (default: 2000)

### Utilities (`util/`)
- **visualizer.py** — Logging to wandb, tensorboard, HTML (via `dominate`), and console. Writes loss plots and saves current results visuals.
- **html.py** — HTML results page generation.
- **image_pool.py** — Image buffer of `--pool_size` for discriminator history replay (stabilizes GAN training).
- **util.py** — Tensor-to-image conversion, directory creation, diagnostics, `init_ddp()`/`cleanup_ddp()`.

### Metrics (`metric/`)
- **fid_kid.py** — FID and KID computation using InceptionV3.
- **psnr_ssim.py** — PSNR/SSIM quality assessment.
- **inception.py** — InceptionV3 feature extractor.

### Scripts (`scripts/`)
- **test_before_push.py** — Pytest integration test: downloads mini datasets, runs train+test for all model types (CycleGAN, pix2pix, template, colorization).
- Shell scripts for dataset/model download and train/test commands.

## Key Design Patterns

- **Domain naming**: Internal names `G_A`, `D_B` etc. map to filenames via `_get_model_filename()` using `--domainA`/`--domainB` (defaults: A/B). E.g., with `--domainA photo --domainB map`, `net_G_A` saves as `latest_net_G_photo2map.pth`.
- **Checkpoint saving**: Files named `{epoch}_net_{name}.pth` in `checkpoints/{experiment_name}/`. Iteration-based saves when `--save_by_iter` is set (e.g., `iter_5000_net_G_A.pth`).
- **Checkpoint loading priority**: First tries `weights_only=True`, falls back without it if the PyTorch version doesn't support the parameter.
- **DDP support**: Uses `DistributedSampler`, `torch.nn.parallel.DistributedDataParallel`, and `init_ddp()`/`cleanup_ddp()` from `util.util`. Only rank 0 saves checkpoints. Must use `--norm syncbatch` (or `sync_instance`) for DDP compatibility — `--norm batch` is not compatible.
- **Multi-GPU launch**: Use `torchrun --nproc_per_node=N` instead of `python train.py`. The `LOCAL_RANK` env var is detected automatically.
- **Vision-aided loss**: Optional. Creates two independent discriminators with frozen backbone, separate optimizers appended to `self.optimizers`, warmup period, and the `should_apply_cv_loss()` guard for warmup logic.
- **Optimizers & schedulers**: Defined as a list (`self.optimizers`). `model.setup()` auto-creates LR schedulers for each optimizer in the list. Supported policies: linear, step, plateau, cosine.
- **Model auto-discovery via import convention**: Model file must be named `{name}_model.py` with a class whose lowercased name matches `{name}model`.
- **Dataset auto-discovery via import convention**: Dataset file must be named `{name}_dataset.py` with a class whose lowercased name matches `{name}dataset`.
