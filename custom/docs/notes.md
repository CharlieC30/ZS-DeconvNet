# Notes

## Code modifications vs upstream

All marked `# MODIFIED:` in source.

- `Python_MATLAB_Codes/train_inference_python/Train_ZSDeconvNet_3D.py:118`. Used `os.makedirs(exist_ok=True)` for recursive save-dir creation.
- `Python_MATLAB_Codes/train_inference_python/Train_ZSDeconvNet_3D.py:132`. Disabled background subtraction on test image read.
- `Python_MATLAB_Codes/train_inference_python/Train_ZSDeconvNet_3D.py:190`. Added `fill_value='extrapolate'` to PSF Z interpolation.
- `Python_MATLAB_Codes/train_inference_python/Infer_3D.py:102`. Disabled background subtraction on inference input read.
- `custom/data_augmentation/augment_3d.py:41`. Removed auto Zsize subfolder so save_path is used directly.

`utils/loss.py:63` has whitespace-only formatting differences with no functional impact.

## Augmentation

Only `augment_3d.py` is in pipeline.

**Deferred TODO:** Python `augment_3d.py` differs from upstream MATLAB `DataAugmFor3D.m` in two ways. Python produces 3 augmentation variants per crop, MATLAB has 4. Python adds an extra random Z-start offset within the crop window. Revisit when validating numerical correctness against MATLAB.

### `augment_3d.py` parameters

| Parameter | Value | MATLAB default | Match? | Note |
|---|---|---|---|---|
| `seg_x`, `seg_y` | 64, 64 | 64, 64 | yes | matches model input |
| `seg_z` | 13 | 13 | yes | matches model input |
| `seg_num` | **100** | 10000 | **no** | production default. Backup 46 runs, 27/29 iUExM and 17/17 aisr used 100. Adequate for ZS-DeconvNet's zero-shot self-supervised regime, not a leftover test value. |
| `rot_flag` | True | 1 | yes | |

## Experiment history

Two runs preserved at `Raw_Data/lab_data/examples/train_inference_python__trained_models__trained_models_3d/` are reproduce targets. The `_ZS-DeconvNet_result.tif` files in `Raw_Data/lab_data/examples/` are bit-identical to corresponding `Inference/Reslice of 00_dec*.tif` files in those backup-copy run dirs, verified by md5.

### iUExM, 29 runs

| Date | Batch tag | Runs | Setup | Notes |
|---|---|---|---|---|
| Sept 1 11:36 | `roiC_crop128_1128_0901_1136_100` | 4 | NA1.1, upsample × iter | convergence study, intermediate ckpts every 100 iter |
| Sept 1 11:39 | `roiC_crop256_1256` | 2 | NA1.1, upsample | patch scaling |
| Sept 1 11:42 | `roiC_0901_1142` | 10 | optical NA sweep × upsample | NA sensitivity, 8 at seg_num=100 plus 2 at seg_num=10000 |
| Sept 2 20:00 | `test_to_charlie_0902_2000` | 12 | NA × upsample, first custom PSFs | deliverable batch. `oddZ_111` run is **reproduce target B** |
| Sept 16 15:50 | `iUExM_roi_0916_1550_100` | 1 | `oddZ_118` + dx=1, dz=2 | physics-aligned conclusion, not preserved in `examples/` |

**Reproduce target B**

- Source backup run: `test_to_charlie_0902_2000_100_twostage_RCAN3D_PSF_XY1.88um_Z15.04um_oddZ_111_upsample0`
- Input: `Raw_Data/examples/iUExM.tif`. Originally `iUExM_roi.tif`, 512³ float32 already pre-normalized.
- PSF: `oddZ_111.tif`, isotropic with dxpsf=dzpsf=1. Does not match physical 8× anisotropy.
- Training voxel: dx=0.5, dz=2, upsample_flag=0, iterations=500.
- Reference output: `Raw_Data/lab_data/examples/iUExM_roi_ZS-DeconvNet_result.tif`
- Re-verified 2026-05-09.

**Deferred TODO:** The iUExM run uses isotropic PSF `oddZ_111` instead of physically correct `oddZ_118`. Training voxel dx=0.5 is half of dxpsf=1.0, which the code interprets as XY 2× upsample by interpolating PSF up. Decide later whether to keep this setup or switch to the aisr-style physics-aligned `oddZ_118 + dx=1, dz=2` configuration.

### aisr, 17 runs

Single batch, Sept 5 ~12:00 2025, all `aisr122424_roi_0905_1200_100`. PSF anisotropy × training voxel anisotropy sweep. All upsample=0, 500 iter, seg_num=100.

| PSF | dxpsf, dzpsf | dx/dz combos | runs |
|---|---|---|---|
| optical NA1.1 | 1.0, 1.0 | (0.5,1)(0.5,2)(1,1)(1,2) | 4 |
| oddZ_111 | 1.0, 1.0 | (0.5,1)(0.5,2)(1,1)(1,2) | 4 |
| oddZ_112 | 1.0, 2.0 | (0.5,1)(0.5,2)(1,2)×2, 1 dir-name typo | 4 |
| oddZ_114 | 1.0, 4.0 | (0.5,1)(0.5,2)(1,1)(1,2), 1 dir-name typo | 4 |
| oddZ_118 | 1.0, 8.0 | **(1, 2) only** | 1 |

`oddZ_118` had only 1 run, likely because it was already known to be the right combo.

**Reproduce target A** is the single oddZ_118 + dx=1, dz=2 run.

- Source backup run: `aisr122424_roi_0905_1200_100_twostage_RCAN3D_PSF_XY1.88um_Z15.04um_oddZ_118_upsample0__dx1dz2`
- Input: `Raw_Data/examples/aisr.tif`. Originally `aisr122424_roi.tif`, 512³ uint16.
- PSF: `custom/psf/output/other/PSF_XY1.88um_Z15.04um_oddZ_118.tif`. Matches physical 8× Z anisotropy.
- Training voxel: dx=1, dz=2, upsample_flag=0, iterations=500.
- Reference output: `Raw_Data/lab_data/examples/aisr122424_roi_ZS-DeconvNet_result.tif`
- Re-verified 2026-05-09.
