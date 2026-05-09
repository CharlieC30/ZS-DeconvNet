# Example data

This directory holds two example datasets, aisr and iUExM, for end-to-end reproduction of the ZS-DeconvNet pipeline.

## Download

Download all example files from [Google Drive](https://drive.google.com/drive/folders/1akov-thP6K8xpBy16ApzMKyDzUMO_1vK) and place them as follows:

```
Raw_Data/examples/
├── inputs/
│   ├── aisr.tif
│   └── iUExM.tif
├── test_images/
│   └── 082525-iUExM-roiC_cropcube128.tif
└── reference_outputs/
    ├── aisr_result_dec.tif
    ├── iUExM_result_dec.tif
    ├── Reslice of aisr.tif
    ├── Reslice of aisr_result_dec.tif
    ├── Reslice of iUExM.tif
    └── Reslice of iUExM_result_dec.tif
```

- `inputs/`: raw 3D stacks for the two datasets.
- `test_images/`: small volume used during training to monitor progress.
- `reference_outputs/`: expected inference outputs and Fiji reslice views for comparison.

## Running aisr

In `custom/data_augmentation/augment_3d.py`, ensure the aisr block in `__main__` is uncommented and the iUExM block is commented out. Then generate augmented patches:

```bash
python custom/data_augmentation/augment_3d.py
```

Patches are written to `outputs/augmented_datasets/3d/aisr/aisr_example_run/{input,gt}/`.

Train:

```bash
bash custom/scripts/train_custom_3d_aisr.sh
```

Outputs are in `outputs/trained_models/3d/aisr_example_run_twostage_RCAN3D__PSF_XY1.88um_Z15.04um_oddZ_118__upsample0__dx1dz2/`.

Run inference:

```bash
bash custom/scripts/infer_custom_3d_aisr.sh
```

Inference results are written to the same run directory under `Inference/00_dec.tif` and `Inference/00_den.tif`. Compare with `reference_outputs/aisr_result_dec.tif`.

## Running iUExM

In `augment_3d.py`, uncomment the iUExM block and comment out the aisr block. Run the three commands:

```bash
python custom/data_augmentation/augment_3d.py
bash custom/scripts/train_custom_3d_iUExM.sh
bash custom/scripts/infer_custom_3d_iUExM.sh
```

Outputs go to `outputs/augmented_datasets/3d/iUExM/iUExM_example_run/` and `outputs/trained_models/3d/iUExM_example_run_twostage_RCAN3D__PSF_XY1.88um_Z15.04um_oddZ_111__upsample0__dx0.5dz2/`. Compare with `reference_outputs/iUExM_result_dec.tif`.

## Path composition

Path templates used by the three scripts:

```
augment save_path        : <data_dir>/<folder>
augment patch dirs       : <save_path>/input/, <save_path>/gt/
train output dir         : <save_weights_dir>/<folder>_<model><save_weights_suffix>/
train weight files       : <train output dir>/weights_<iter>.h5
infer load_weights_path  : <train output dir>/weights_<iter>.h5
```

- Augment's `save_path` must equal `<data_dir>/<folder>` so the train script reads the right patches.
- Infer's `--load_weights_path` must point inside the train output directory.
