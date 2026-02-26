# ZS-DeconvNet (Custom Development)

A customized development version of **ZS-DeconvNet** — a zero-shot learning deep neural network for instant denoising and super-resolution in optical fluorescence microscopy.

> Based on the original work by Zeng et al.: [TristaZeng/ZS-DeconvNet](https://github.com/TristaZeng/ZS-DeconvNet)
>
> Paper: *"Zero-shot learning enables instant denoising and super-resolution in optical fluorescence microscopy"*
>
> Original tutorial: [ZS-DeconvNet Tutorial](https://tristazeng.github.io/ZS-DeconvNet-page/Tutorial/)

## What's Different from the Original

This fork extends the original ZS-DeconvNet with:

- **Python-based PSF generation** (`psf/`) — optical and Gaussian PSF generation with Born & Wolf theory, replacing MATLAB dependency
- **Python data augmentation** (`data_augmentation/`) — 2D and 3D data augmentation scripts that replace the original MATLAB-based pipeline
- **Custom training/inference scripts** (`scripts/`) — tailored for specific microscopy datasets (AISR, iUExM)
- **Pre-configured Conda environments** (`envs/`) — ready-to-use environment files for both TensorFlow and PyTorch setups
- **Minor modifications to upstream code** — marked with `# MODIFIED:` comments; see `docs/notes.md` for details

## Directory Structure

**Naming convention**: Uppercase directories = original author's code (unchanged), lowercase directories = custom additions.

```
ZS-DeconvNet/
│
│ ── Original author's code (preserved from upstream) ──
│
├── Python_MATLAB_Codes/              # Original TensorFlow 2.5.0 implementation
│   ├── train_inference_python/       # Training and inference scripts
│   │   ├── models/                   # U-Net 2D/3D, RCAN 3D architectures
│   │   ├── utils/                    # Loss functions, data loader, utilities
│   │   ├── train_demo_*.sh           # Demo training scripts
│   │   ├── infer_demo_*.sh           # Demo inference scripts
│   │   ├── requirements.txt          # Original pip dependencies
│   │   ├── trained_models/           # Trained model weights (not in git)
│   │   └── augmented_datasets/       # Generated training data (not in git)
│   ├── data_augment_recorrupt_matlab/ # Original MATLAB augmentation code
│   └── saved_models/                  # Original pre-trained models
├── Fiji_Plugin/                       # Original Fiji/ImageJ plugin
├── Raw_Data/                          # Experimental raw data (mostly gitignored)
│
│ ── Custom additions (this fork) ──
│
├── psf/                              # PSF generation and testing (replaces MATLAB)
│   ├── generate_psf.py               # Optical & Gaussian PSF generation
│   ├── psf_test_2d.py                # 2D PSF convolution testing (PyTorch)
│   └── psf_test_3d.py                # 3D PSF convolution testing (SciPy)
├── data_augmentation/                # Python data augmentation (replaces MATLAB)
│   ├── augment_2d.py                 # 2D augmentation with noise generation
│   └── augment_3d.py                 # 3D augmentation with noise generation
├── scripts/                          # Custom shell scripts for training/inference
├── envs/                             # Environment configurations
│   ├── tensorflow.yml                # Conda env: TF 2.5 + CUDA 11.3
│   ├── pytorch.yml                   # Conda env: PyTorch 1.13
│   └── requirements.txt              # pip dependencies
├── docs/                             # Documentation and notes
│   └── notes.md                      # Parameter reference, architecture notes, upstream diffs
└── screenshots/                      # Personal screenshots (gitignored)
```

## Quick Start

### 1. Environment Setup

```bash
# Option A: Use pre-configured Conda environment (recommended)
conda env create -f envs/tensorflow.yml
conda activate zs-deconvnet

# Option B: Manual setup
conda create -n zs-deconvnet python=3.9.7
conda activate zs-deconvnet
pip install -r envs/requirements.txt
conda install cudatoolkit==11.3.1 cudnn==8.2.1

# Note: The original author's requirements.txt is also available at
# Python_MATLAB_Codes/train_inference_python/requirements.txt (unchanged)
```

### 2. PSF Generation

```bash
cd psf/
python generate_psf.py  # Edit parameters in script for your optical setup
```

### 3. Data Augmentation (Zero-shot)

```bash
# 2D data augmentation
python data_augmentation/augment_2d.py --input_dir [RAW_DATA_DIR]

# 3D data augmentation
python data_augmentation/augment_3d.py --data_path [RAW_DATA_DIR]
```

### 4. Training

```bash
cd Python_MATLAB_Codes/train_inference_python

# 2D training
python Train_ZSDeconvNet_2D.py --otf_or_psf_path [PSF_PATH] --data_dir [DATA_DIR] --folder [FOLDER_NAME]

# 3D training
python Train_ZSDeconvNet_3D.py --psf_path [PSF_PATH] --data_dir [DATA_DIR] --folder [FOLDER_NAME]

# Or use demo/custom scripts:
./train_demo_2D.sh           # Original demo
../../scripts/train_custom_3d.sh  # Custom 3D training
```

### 5. Inference

```bash
cd Python_MATLAB_Codes/train_inference_python

# 2D inference
python Infer_2D.py --input_dir [INPUT_PATH] --load_weights_path [WEIGHTS_PATH]

# 3D inference
python Infer_3D.py --input_dir [INPUT_PATH] --load_weights_path [WEIGHTS_PATH]
```

## Architecture

The model uses a **two-stage architecture**:

1. **Stage 1 (Denoising)**: U-Net that removes noise from the input image
2. **Stage 2 (Deconvolution)**: Network that performs deconvolution using PSF-based frequency domain loss, with optional 2x super-resolution

Supported model architectures:
- `twostage_Unet` — 2D U-Net (for wide-field microscopy)
- `twostage_Unet3D` — 3D U-Net (for confocal, LLSM)
- `twostage_RCAN3D` — 3D Residual Channel Attention Network (for confocal, LLSM, SIM)

For detailed parameter explanations and architecture notes, see [`docs/notes.md`](docs/notes.md).

## Acknowledgments

This project is based on [ZS-DeconvNet](https://github.com/TristaZeng/ZS-DeconvNet) by Zeng et al. All credit for the original architecture and methodology goes to the original authors.
