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

## Repository structure

```
ZS-DeconvNet/
├── README.md
├── .gitignore
├── custom/                      # PSF tools, augmentation, scripts, env files
├── Python_MATLAB_Codes/         # Python and MATLAB training/inference code
├── Raw_Data/                    # Raw input data
│   └── examples/                # Example data for end-to-end runs
└── outputs/                     # Augmentation, training, and inference outputs
```

## Pipeline

1. **PSF preparation**: provide a 3D PSF `.tif` under `custom/psf/output/other/`, generated via `custom/psf/generate_psf.py` or supplied directly.
2. **Data augmentation**: `custom/data_augmentation/augment_3d.py` produces `input/` and `gt/` patch pairs.
3. **Training**: `Train_ZSDeconvNet_3D.py` consumes the patches plus the PSF and produces weights `.h5` files.
4. **Inference**: `Infer_3D.py` loads the weights and a raw input image and produces denoised and deconvolved volumes.

## Example data

- Examples and how to run them are documented in [`Raw_Data/examples/README.md`](Raw_Data/examples/README.md).
- Download data from [Google Drive](https://drive.google.com/drive/folders/1akov-thP6K8xpBy16ApzMKyDzUMO_1vK).

## Citation

Qiao, C., Zeng, Y., Meng, Q. et al. Zero-shot learning enables instant denoising and super-resolution in optical fluorescence microscopy. *Nat Commun* 15, 4180 (2024). https://doi.org/10.1038/s41467-024-48575-9
