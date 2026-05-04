# ZS-DeconvNet

ZS-DeconvNet is a zero-shot deep learning method for fluorescence microscopy denoising and super-resolution.
This repository provides Python tooling for PSF generation, data augmentation, and custom training scripts.

## Installation

The environment uses TensorFlow 2.5 with CUDA 11.3 and cuDNN 8.2.

```bash
conda create -n zs-deconvnet python=3.9.7
conda activate zs-deconvnet
conda install cudatoolkit==11.3.1 cudnn==8.2.1
pip install -r custom/envs/tensorflow_requirement.txt
```

## Quick Start

PSF generation:

```bash
python custom/psf/generate_psf.py
```

3D data augmentation:

```bash
python custom/data_augmentation/augment_3d.py
```

Training:

```bash
./custom/scripts/train_custom_3d.sh
```

Inference:

```bash
./custom/scripts/infer_custom_3d.sh
```

Additional training and inference scripts are under `Python_MATLAB_Codes/train_inference_python/`.

## Repository Structure

```
ZS-DeconvNet/
├── README.md
├── .gitignore
│
├── custom/                      # PSF, augmentation, scripts, env files
│   ├── psf/
│   ├── data_augmentation/
│   ├── scripts/
│   ├── envs/
│   └── docs/
│
├── Python_MATLAB_Codes/         # Python and MATLAB implementation
│   ├── train_inference_python/
│   └── data_augment_recorrupt_matlab/
│
├── Raw_Data/                    # raw inputs
│   ├── upstream_demo/
│   ├── lab_data/
│   └── examples/
│
└── outputs/                     # training outputs
```

## Citation

Qiao, C., Zeng, Y., Meng, Q. et al. Zero-shot learning enables instant denoising and super-resolution in optical fluorescence microscopy. *Nat Commun* 15, 4180 (2024). https://doi.org/10.1038/s41467-024-48575-9
