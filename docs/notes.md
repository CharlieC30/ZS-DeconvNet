# Development Notes

This document contains parameter explanations, architecture notes, and a list of modifications made to the upstream (original author) code.

## Modifications to Upstream Code

All modifications are marked with `# MODIFIED:` comments in the source code.

### Train_ZSDeconvNet_3D.py

1. **Background subtraction disabled** (line ~135)
   - Upstream: `image = image - background` (enabled)
   - Fork: commented out
   - Reason: experimental change, may need re-evaluation

2. **PSF interpolation extrapolation** (line ~193)
   - Upstream: `interp1d(z2, curCol, 'slinear')`
   - Fork: `interp1d(z2, curCol, 'slinear', fill_value='extrapolate')`
   - Reason: prevents boundary errors when interpolation points fall outside the original PSF range

### Infer_3D.py

1. **Background subtraction disabled** (line ~102)
   - Upstream: `image = image-background` (enabled)
   - Fork: commented out
   - Reason: same as Train_3D.py

### utils/loss.py

1. **Return statement formatting** (line ~66)
   - Upstream: `psf_loss+TV_weight*TV_loss+laplace_weight*laplace_loss+...`
   - Fork: `psf_loss + TV_weight*TV_loss + laplace_weight*laplace_loss + ...`
   - Reason: readability improvement only, no functional change

---

## Training Parameters Reference

### 2D Training (Train_ZSDeconvNet_2D.py)

**Model architecture:**
- `--conv_block_num` (default: 4): Number of encoder/decoder blocks in U-Net
- `--conv_num` (default: 3): Number of convolution layers per block
- `--upsample_flag` (default: 1): 1 = deconvolution + 2x super-resolution, 0 = deconvolution only

**Training settings:**
- `--gpu_memory_fraction` (default: 0.9): GPU memory usage fraction
- `--mixed_precision_training` (default: 1): Enable mixed precision for memory efficiency
- `--iterations` (default: 50000): Total training iterations
- `--test_interval` (default: 1000): Test every N iterations
- `--valid_interval` (default: 1000): Validate every N iterations

**Learning rate:**
- `--start_lr` (default: 5e-5): Initial learning rate
- `--lr_decay_factor` (default: 0.5): LR multiplied by this every 10000 iterations

**Pixel spacing:**
- `--dxypsf` (default: 0.0313): PSF pixel size in micrometers
- `--dx`, `--dy` (default: 0.0313): Input image pixel size in micrometers

**Patch dimensions:**
- `--input_x`, `--input_y` (default: 128): Training patch size
- `--insert_xy` (default: 16): Edge padding to avoid convolution boundary artifacts
- `--valid_num` (default: 3): Number of images used for validation

**Loss functions:**
- `--mse_flag` (default: 0): 0 = MAE loss, 1 = MSE loss
- `--denoise_loss_weight` (default: 0.5): Weight for denoising loss (0.5 = equal weight for denoising and deconvolution)
- `--l1_rate` (default: 0): L1 regularization weight — encourages sparsity, removes low-intensity background noise
- `--TV_rate` (default: 0): Total Variation regularization weight — smoothing, suppresses noise between adjacent pixels
- `--Hess_rate` (default: 0.02): Hessian regularization weight — preserves edges while maintaining structural continuity

### 3D Training (Train_ZSDeconvNet_3D.py)

Additional parameters compared to 2D:
- `--input_z` (default: 64): Z-axis patch dimension
- `--insert_z`: Z-axis edge padding
- `--dz`, `--dzpsf`: Z-axis pixel spacing
- `--background` (default: 100): Background intensity value for subtraction

---

## Architecture Overview

### Two-Stage Pipeline

1. **Stage 1 — Denoising**: U-Net removes noise → MSE/MAE loss between denoised output and input
2. **Stage 2 — Deconvolution**: Uses PSF-based frequency domain loss → optional 2x super-resolution

### Model Types

| Model | Use Case | Key Feature |
|-------|----------|-------------|
| `twostage_Unet` | 2D images | Classic U-Net, 4-level encoder-decoder with skip connections |
| `twostage_Unet3D` | 3D volumes | Asymmetric pooling (2,2,1) preserves Z-axis detail |
| `twostage_RCAN3D` | 3D volumes | Residual Channel Attention mechanism |

### U-Net 2D Architecture (twostage_Unet.py)

- `upsample_flag`: Controls whether Stage 2 performs 2x super-resolution upsampling
- `insert_x`, `insert_y`: Crop Stage 1 output to remove padding from the valid region
- `conv_block_num` (default: 4): Number of encoder/decoder downsampling/upsampling blocks
- `conv_num` (default: 3): Number of convolution layers within each block
- Bottleneck: Conv2D(channels*2) → Conv2D(channels) between encoder and decoder
- Skip connections: Concatenation along channel axis (axis=3) between encoder and decoder

### PSF/OTF Loss Function (utils/loss.py)

The PSF loss re-blurs the model prediction with the PSF and compares to the original input:
1. Convolve prediction with PSF kernel (`K.conv2d`)
2. If super-resolution is enabled, resize convolved output back to original dimensions
3. Crop to remove padding
4. Compare with input using MAE or MSE

Regularization terms:
- **TV (Total Variation)**: Penalizes differences between adjacent pixels → smoother results
- **Hessian**: Penalizes second-order derivatives (gradient of gradient) → preserves edges and structural continuity
- **Laplacian**: Alternative second-order regularization
- **L1**: Penalizes absolute pixel values → encourages sparse (dark background) results

### Data Loading (utils/data_loader.py)

- Random sampling: `np.random.choice(images_path, size=batch_size, replace=False)` ensures no duplicate images within a batch
- `imageio.mimread`: Multi-frame TIFF reader that handles both single and multi-frame files
- Normalization: Percentile-based normalization scales pixel values to [0, 1]

### Zero-Shot Learning via Re-corruption

No ground truth is needed. The approach:
1. Estimate noise parameters (β1 for Poisson, β2 for Gaussian) from data
2. Generate paired data: input (add more noise) / gt (reduce noise)
3. Train with noise magnification factor α (typically 1-2)
4. Noise model: `noise_variance = β1 * signal + β2`
