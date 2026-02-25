# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

ZS-DeconvNet is a zero-shot learning deep neural network for instant denoising and super-resolution in optical fluorescence microscopy. This is a customized fork of [TristaZeng/ZS-DeconvNet](https://github.com/TristaZeng/ZS-DeconvNet) with Python-based PSF generation, data augmentation (replacing MATLAB), and custom training scripts.

### Directory Structure (from repo root)
- `Python_MATLAB_Codes/` — Main TensorFlow 2.5.0 implementation
  - `train_inference_python/` — Training and inference scripts
    - `models/` — Network architectures (U-Net 2D/3D, RCAN 3D)
    - `utils/` — Loss functions, data loader, utilities
    - `custom_script/` — Custom data augmentation and training/inference scripts
    - `trained_models/` — Trained model weights (ignored by git)
    - `augmented_datasets/` — Generated training data (ignored by git)
  - `data_augment_recorrupt_matlab/` — Original MATLAB codes for data generation and PSF simulation
- `Python_PSF/` — PSF generation and testing tools
  - `generate_psf.py` — Optical (Born & Wolf) and Gaussian PSF generation
  - `psf_test_2d.py` — 2D PSF convolution testing (PyTorch-based)
  - `psf_test_3d.py` — 3D PSF convolution testing (SciPy-based)
  - `PSFoutput/` — Generated PSF files
  - `PSFtest/` — PSF test input/output
- `Conda_env/` — Pre-configured Conda environment files (TF and PyTorch)
- `Raw_Data/` — Experimental raw data (mostly ignored by git)
- `Fiji_Plugin/` — Original Fiji/ImageJ plugin and model conversion tools
- `Pytorch_2d/` — PyTorch 2D implementation (in development)
- `Vibe_coding_prompt/` — AI coding task specifications
- `Screenshots/` — Screenshots

## Development Commands

### Python Environment Setup
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

### Training Commands
All training scripts must be run from `train_inference_python/` directory:

```bash
cd train_inference_python

# 2D model training
python Train_ZSDeconvNet_2D.py --otf_or_psf_path [PSF_PATH] --data_dir [DATA_DIR] --folder [FOLDER_NAME] --test_images_path [TEST_PATH]

# 3D model training
python Train_ZSDeconvNet_3D.py --psf_path [PSF_PATH] --data_dir [DATA_DIR] --folder [FOLDER_NAME] --test_images_path [TEST_PATH]

# Use demo scripts (edit paths first):
./train_demo_2D.sh        # 2D wide-field data
./train_demo_3D.sh        # 3D wide-field, confocal, LLSM data
./train_demo_2DSIM.sh     # 2D reconstructed SIM data
./train_demo_3DSIM.sh     # 3D reconstructed SIM data
```

### Inference Commands
```bash
cd train_inference_python

# 2D inference
python Infer_2D.py --input_dir [INPUT_PATH] --load_weights_path [WEIGHTS_PATH]

# 3D inference
python Infer_3D.py --input_dir [INPUT_PATH] --load_weights_path [WEIGHTS_PATH]

# Use demo scripts (edit paths first):
./infer_demo_2D.sh
./infer_demo_3D.sh
```

### Data Augmentation (Python)
```bash
cd Python_MATLAB_Codes/train_inference_python

# Generate augmented 2D training data
python DataAugmFor2d_python.py --input_dir [RAW_DATA_DIR] --output_dir [OUTPUT_DIR]

# Generate augmented 3D training data
python DataAugmFor3d_python.py --input_dir [RAW_DATA_DIR] --output_dir [OUTPUT_DIR]

# Custom scripts are also available in custom_script/:
# custom_script/DataAugmFor2d_python.py  — 2D augmentation (same as above, organized copy)
# custom_script/DataAugmFor3d_python.py  — 3D augmentation (same as above, organized copy)
```

### PSF Generation
```bash
cd Python_PSF/

# Edit parameters in generate_psf.py for your optical setup, then run:
python generate_psf.py

# Test PSF with sample images:
python psf_test_2d.py   # 2D convolution test
python psf_test_3d.py   # 3D convolution test
```

## Architecture Overview

### Core Components

1. **Two-Stage Architecture**: All models use a two-stage approach:
   - Stage 1: Denoising network
   - Stage 2: Deconvolution network with PSF-based loss

2. **Model Types**:
   - `twostage_Unet`: 2D U-Net based model
   - `twostage_Unet3D`: 3D U-Net based model
   - `twostage_RCAN3D`: 3D Residual Channel Attention Network

3. **Key Files** (in `train_inference_python/`):
   - `models/twostage_Unet.py`: 2D U-Net architecture
   - `models/twostage_Unet3D.py`: 3D U-Net architecture
   - `models/twostage_RCAN3D.py`: 3D Residual Channel Attention Network
   - `utils/data_loader.py`: Data loading and preprocessing functions
   - `utils/loss.py`: Loss functions including frequency domain losses
   - `utils/utils.py`: General utilities for PSF handling, normalization
   - `utils/augment_sim_img.py`: Data augmentation for SIM images
   - Training scripts: `Train_ZSDeconvNet_2D.py`, `Train_ZSDeconvNet_3D.py`, `Train_ZSDeconvNet_2DSIM.py`, `Train_ZSDeconvNet_3DSIM.py`
   - Inference scripts: `Infer_2D.py`, `Infer_3D.py`
   - `DataAugmFor2d_python.py`: Python-based data augmentation for 2D datasets (replaces MATLAB)
   - `DataAugmFor3d_python.py`: Python-based data augmentation for 3D datasets (replaces MATLAB)
   - `custom_script/`: Custom training/inference shell scripts and data augmentation copies

### Data Structure
Training data should be organized as:
```
data_dir/
├── folder_name/
│   ├── input/          # Noisy input images
│   └── gt/             # Ground truth images (for supervised training)
```

For zero-shot training, only input images are needed with data augmentation.

### Loss Functions
- **Denoising Loss**: MSE or MAE between denoised and target
- **Deconvolution Loss**: Frequency domain loss using PSF/OTF
- **Regularization**: Hessian, TV, L1 regularization terms

### PSF/OTF Handling
- Supports both PSF (.tif) and OTF (.mrc) formats
- Automatic interpolation for pixel size matching
- PSF normalization before loss calculation

## Key Parameters

### Training Parameters
- `--gpu_id`: GPU device ID
- `--iterations`: Total training iterations
- `--start_lr`: Initial learning rate (5e-5 for 2D, 1e-4 for 3D)
- `--batch_size`: Batch size (4 for 2D, 2-3 for 3D)
- `--input_x/y/z`: Patch dimensions (128 for 2D, 64 for 3D)
- `--dx/dy/dz`: Pixel spacing in micrometers
- `--upsample_flag`: Whether to perform super-resolution

### Loss Weights
- `--denoise_loss_weight`: Weight for denoising loss
- `--Hess_rate`: Hessian regularization (0.02 for 2D, 0.1 for 3D)
- `--TV_rate`: Total variation regularization
- `--l1_rate`: L1 regularization

### Data Augmentation (Zero-shot)
- `--alpha`: Noise magnification factor [1-2]
- `--beta1`: Poisson noise factor [0.5-1.5]
- `--beta2`: Gaussian noise variance (estimated from data)

## Development Notes

### GPU Memory Management
- Set `--gpu_memory_fraction` to limit GPU usage
- Use `--mixed_precision_training` for memory efficiency
- Adjust `--batch_size` based on available memory

### Model Saving
- Models saved in `train_inference_python/trained_models/` (2D and 3D subdirectories)
- Weights saved as `.h5` files at specified intervals
- Configuration saved as `config.txt`
- Logs saved to `graph/` subdirectory for TensorBoard monitoring

### Testing and Monitoring
- Use `--test_interval` to specify validation frequency during training
- Test images specified via `--test_images_path`
- Inference results saved to `Inference/` subdirectory within model folder
- Monitor training with: `tensorboard --logdir [model_dir]/graph`

### Dependencies and Requirements
- TensorFlow 2.5.0 with GPU support
- Required packages: `imageio`, `tifffile`, `scipy==1.7.1`, `opencv-python`
- CUDA 11.3/11.4 and cuDNN 8.2 for GPU acceleration

## Common Issues

1. **CUDA compatibility**: Ensure TensorFlow 2.5.0 matches CUDA 11.3-11.4/cuDNN 8.2
2. **Memory errors**: Reduce `--batch_size` or patch dimensions (`--input_x/y/z`)
3. **PSF format**: Verify PSF dimensions are odd numbers and properly normalized
4. **Working directory**: Always run training/inference from `train_inference_python/` directory
5. **Demo scripts**: Edit absolute paths in `.sh` files before running
6. **Data organization**: Ensure training data follows `data_dir/folder/input/` and `data_dir/folder/gt/` structure

## Git Workflow

- This repo is forked from [TristaZeng/ZS-DeconvNet](https://github.com/TristaZeng/ZS-DeconvNet)
- `origin` → `CharlieC30/ZS-DeconvNet` (your fork)
- `upstream` → `TristaZeng/ZS-DeconvNet` (original author)
- Use `git fetch upstream` to pull original author's updates
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`

## Quick Start Workflow

1. **Setup environment**: Use `Conda_env/zs-deconvnet_environment.yml` or install manually
2. **Prepare data**: Run data augmentation scripts to generate training pairs
3. **Generate PSF**: Use `Python_PSF/generate_psf.py` for your optical setup
4. **Train model**: Edit and run demo training scripts from `train_inference_python/`
5. **Run inference**: Edit and run demo inference scripts
6. **Monitor training**: Use TensorBoard: `tensorboard --logdir [model_dir]/graph`