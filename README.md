# ZS-DeconvNet (Custom Development)

A customized development version of **ZS-DeconvNet** — a zero-shot learning deep neural network for instant denoising and super-resolution in optical fluorescence microscopy.

> Based on the original work by Zeng et al.: [TristaZeng/ZS-DeconvNet](https://github.com/TristaZeng/ZS-DeconvNet)
>
> Paper: *"Zero-shot learning enables instant denoising and super-resolution in optical fluorescence microscopy"*
>
> Original tutorial: [ZS-DeconvNet Tutorial](https://tristazeng.github.io/ZS-DeconvNet-page/Tutorial/)

## What's Different from the Original

This fork extends the original ZS-DeconvNet with:

- **Python-based PSF generation** (`custom/psf/`) — optical and Gaussian PSF generation with Born & Wolf theory, replacing MATLAB dependency
- **Python data augmentation** (`custom/data_augmentation/`) — 2D and 3D augmentation scripts that replace the original MATLAB-based pipeline
- **Custom training/inference scripts** (`custom/scripts/`) — tailored for specific microscopy datasets (AISR, iUExM)
- **Pre-configured Conda environments** (`custom/envs/`) — ready-to-use environment files for both TensorFlow and PyTorch setups
- **Minor modifications to upstream code** — marked with `# MODIFIED:` comments; see `custom/docs/notes.md` for details

## Directory Structure

The repository separates upstream code (under `Python_MATLAB_Codes/`) from this fork's additions (under `custom/`). Training outputs land in `outputs/`. Raw input data lives under `Raw_Data/`.

```
ZS-DeconvNet/
│
├── Python_MATLAB_Codes/              # Upstream code (preserved unchanged)
│   ├── train_inference_python/       # Training and inference entry points
│   │   ├── Train_*.py / Infer_*.py   # Original training and inference scripts
│   │   ├── models/                   # U-Net 2D/3D, RCAN 3D architectures
│   │   ├── utils/                    # Loss functions, data loader, utilities
│   │   ├── train_demo_*.sh           # Original demo training scripts
│   │   ├── infer_demo_*.sh           # Original demo inference scripts
│   │   ├── requirements.txt          # Original pip dependencies
│   │   ├── trained_models/           # Historical training outputs (gitignored)
│   │   └── augmented_datasets/       # Historical augmented data (gitignored)
│   └── data_augment_recorrupt_matlab/ # Upstream MATLAB augmentation code
│
├── custom/                           # Fork-specific additions
│   ├── psf/                          # PSF generation and testing
│   │   ├── generate_psf.py           # Optical and Gaussian PSF generation
│   │   ├── psf_test_2d.py            # 2D PSF convolution testing
│   │   └── psf_test_3d.py            # 3D PSF convolution testing
│   ├── data_augmentation/            # Python data augmentation
│   │   ├── augment_2d.py             # 2D augmentation
│   │   └── augment_3d.py             # 3D augmentation (used in this fork)
│   ├── scripts/                      # Shell scripts for training and inference
│   ├── envs/                         # Conda env files and pip requirements
│   └── docs/                         # Notes and parameter reference
│       └── notes.md
│
├── outputs/                          # Future training outputs (gitignored)
│   ├── trained_models/3d/...         # New training runs land here
│   └── augmented_datasets/3d/...     # Augmented data produced from custom/data_augmentation/
│
├── Raw_Data/                         # Raw inputs (most subtrees gitignored)
│   ├── README.md                     # Data overview and source links
│   ├── upstream_demo/                # Sample data from upstream Zenodo (gitignored)
│   ├── lab_data/                     # Lab-internal data (untracked)
│   └── examples/                     # Public-facing example data (placeholder)
│
└── README.md / .gitignore
```

`Fiji_Plugin/` (the upstream ImageJ plugin) is kept locally but not tracked in git, since this fork does not exercise the Fiji integration path.

## Quick Start

### 1. Environment Setup

```bash
# Recommended: pre-configured Conda environment
conda env create -f custom/envs/tensorflow.yml
conda activate zs-deconvnet
```

The original author's `Python_MATLAB_Codes/train_inference_python/requirements.txt` is also available unchanged for reference.

### 2. PSF Generation

```bash
cd custom/psf/
python generate_psf.py  # Edit parameters in the __main__ block for your optical setup
```

### 3. Data Augmentation (Zero-Shot)

```bash
# 3D data augmentation (used by this fork)
python custom/data_augmentation/augment_3d.py
```

The 2D version (`augment_2d.py`) is present but not exercised by the current workflow.

### 4. Training

```bash
# Using the custom shell wrapper (handles cd, conda, args)
./custom/scripts/train_custom_3d.sh

# Or call upstream training directly
cd Python_MATLAB_Codes/train_inference_python
python Train_ZSDeconvNet_3D.py --psf_path [PSF] --data_dir [DATA_DIR] --folder [FOLDER]
```

### 5. Inference

```bash
# Using the custom shell wrapper
./custom/scripts/infer_custom_3d.sh

# Or call upstream inference directly
cd Python_MATLAB_Codes/train_inference_python
python Infer_3D.py --input_dir [INPUT] --load_weights_path [WEIGHTS]
```

## Architecture

The model uses a two-stage architecture:

1. **Stage 1 (Denoising)**: U-Net or RCAN that removes noise from the input image
2. **Stage 2 (Deconvolution)**: Network that performs deconvolution using a PSF-based frequency domain loss, with optional 2x super-resolution

Supported model architectures:
- `twostage_Unet` — 2D U-Net (wide-field microscopy)
- `twostage_Unet3D` — 3D U-Net (confocal, LLSM)
- `twostage_RCAN3D` — 3D Residual Channel Attention Network (confocal, LLSM, SIM)

For detailed parameter explanations and architecture notes, see [`custom/docs/notes.md`](custom/docs/notes.md).

## Acknowledgments

This project is based on [ZS-DeconvNet](https://github.com/TristaZeng/ZS-DeconvNet) by Zeng et al. All credit for the original architecture and methodology goes to the original authors.
