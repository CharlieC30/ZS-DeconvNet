# ZS-DeconvNet (Custom Development)

A customized development version of **ZS-DeconvNet** — a zero-shot learning deep neural network for instant denoising and super-resolution in optical fluorescence microscopy.

> Based on the original work by Zeng et al.: [TristaZeng/ZS-DeconvNet](https://github.com/TristaZeng/ZS-DeconvNet)
>
> Paper: *"Zero-shot learning enables instant denoising and super-resolution in optical fluorescence microscopy"*
>
> Original tutorial: [ZS-DeconvNet Tutorial](https://tristazeng.github.io/ZS-DeconvNet-page/Tutorial/)

## What's Different from the Original

This fork extends the original ZS-DeconvNet with:

- **Python-based PSF generation** (`Python_PSF/`) — optical and Gaussian PSF generation with Born & Wolf theory, replacing the need for external PSF tools
- **Python data augmentation** — 2D and 3D data augmentation scripts (`DataAugmFor2d_python.py`, `DataAugmFor3d_python.py`) that replace the original MATLAB dependency
- **Custom training/inference scripts** (`custom_script/`) — tailored for specific microscopy datasets (AISR, iUExM)
- **Pre-configured Conda environments** (`Conda_env/`) — ready-to-use environment files for both TensorFlow and PyTorch setups
- **Reorganized project structure** — cleaner directory layout with `.gitkeep` for structure preservation

## Directory Structure

```
ZS-DeconvNet/
├── Python_MATLAB_Codes/          # Main TensorFlow 2.5.0 implementation
│   ├── train_inference_python/   # Training and inference scripts
│   │   ├── models/               # U-Net 2D/3D, RCAN 3D architectures
│   │   ├── utils/                # Loss functions, data loader, utilities
│   │   ├── custom_script/        # Custom augmentation and training scripts
│   │   ├── trained_models/       # Trained model weights (not in git)
│   │   └── augmented_datasets/   # Generated training data (not in git)
│   └── data_augment_recorrupt_matlab/  # Original MATLAB augmentation code
├── Python_PSF/                   # PSF generation and testing tools
│   ├── generate_psf.py           # Optical & Gaussian PSF generation
│   ├── psf_test_2d.py            # 2D PSF convolution testing
│   └── psf_test_3d.py            # 3D PSF convolution testing
├── Conda_env/                    # Conda environment configuration files
├── Raw_Data/                     # Experimental raw data
├── Fiji_Plugin/                  # Original Fiji/ImageJ plugin (unchanged)
├── Pytorch_2d/                   # PyTorch 2D implementation (in development)
├── Vibe_coding_prompt/           # AI coding task specifications
└── Screenshots/                  # Screenshots
```

## Quick Start

### 1. Environment Setup

```bash
# Option A: Use pre-configured environment (recommended)
cd Conda_env/
conda env create -f zs-deconvnet_environment.yml
conda activate zs-deconvnet

# Option B: Manual setup
conda create -n zs-deconvnet python=3.9.7
conda activate zs-deconvnet
cd Python_MATLAB_Codes/train_inference_python
pip install -r requirements.txt
conda install cudatoolkit==11.3.1 cudnn==8.2.1
```

### 2. Data Augmentation (Zero-shot)

```bash
cd Python_MATLAB_Codes/train_inference_python

# 2D data augmentation
python DataAugmFor2d_python.py --input_dir [RAW_DATA_DIR] --output_dir [OUTPUT_DIR]

# 3D data augmentation
python DataAugmFor3d_python.py --input_dir [RAW_DATA_DIR] --output_dir [OUTPUT_DIR]
```

### 3. Training

```bash
cd Python_MATLAB_Codes/train_inference_python

# 2D training
python Train_ZSDeconvNet_2D.py --otf_or_psf_path [PSF_PATH] --data_dir [DATA_DIR] --folder [FOLDER_NAME]

# 3D training
python Train_ZSDeconvNet_3D.py --psf_path [PSF_PATH] --data_dir [DATA_DIR] --folder [FOLDER_NAME]

# Or use demo scripts (edit paths first):
./train_demo_2D.sh
./train_demo_3D.sh
```

### 4. Inference

```bash
cd Python_MATLAB_Codes/train_inference_python

# 2D inference
python Infer_2D.py --input_dir [INPUT_PATH] --load_weights_path [WEIGHTS_PATH]

# 3D inference
python Infer_3D.py --input_dir [INPUT_PATH] --load_weights_path [WEIGHTS_PATH]
```

### 5. PSF Generation

```bash
cd Python_PSF/
python generate_psf.py  # Edit parameters in script for your optical setup
```

## Architecture

The model uses a **two-stage architecture**:

1. **Stage 1 (Denoising)**: U-Net that removes noise from the input image
2. **Stage 2 (Deconvolution)**: Network that performs deconvolution using PSF-based frequency domain loss, with optional 2x super-resolution

Supported model architectures:
- `twostage_Unet` — 2D U-Net (for wide-field microscopy)
- `twostage_Unet3D` — 3D U-Net (for confocal, LLSM)
- `twostage_RCAN3D` — 3D Residual Channel Attention Network (for confocal, LLSM, SIM)

## Acknowledgments

This project is based on [ZS-DeconvNet](https://github.com/TristaZeng/ZS-DeconvNet) by Zeng et al. All credit for the original architecture and methodology goes to the original authors.
