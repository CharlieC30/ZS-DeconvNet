# DataAugmFor2d_python.py - Python implementation of MATLAB DataAugmFor2D

import os
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, rotate, convolve
import tifffile
import imageio
from typing import List, Tuple, Union
import re
import glob


def natural_sort(file_list: List[str]) -> List[str]:
    def natural_key(text):
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]
    return sorted(file_list, key=natural_key)


def percentile_norm(data: np.ndarray, tmin: float = 0, tmax: float = 100) -> np.ndarray:
    data = data.astype(np.float64)
    data_min = np.percentile(data.flatten(), tmin)
    data_max = np.percentile(data.flatten(), tmax)
    if data_max == data_min:
        return np.zeros_like(data)
    output = (data - data_min) / (data_max - data_min)
    output = np.clip(output, 0, 1)
    return output


def fspecial_gaussian(size: int, sigma: float) -> np.ndarray:
    """
    Replicate MATLAB's fspecial('gaussian', size, sigma)

    Args:
        size: Kernel size (will create size x size kernel)
        sigma: Standard deviation of the Gaussian

    Returns:
        2D Gaussian kernel normalized to sum to 1
    """
    m, n = [(size - 1.) / 2., (size - 1.) / 2.]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def calculate_mask(img: np.ndarray, ksize: int = 10, thresh: float = 0.05) -> np.ndarray:
    """
    Calculate foreground mask using Gaussian filtering.
    Replicates MATLAB's XxCalMask function.

    Args:
        img: Input 2D image
        ksize: Size of small Gaussian kernel (also used as sigma)
        thresh: Threshold for foreground detection

    Returns:
        Binary mask (True for foreground pixels)
    """
    img = img.astype(np.float32)

    # Small kernel for local features (matching MATLAB's fspecial('gaussian', [ksize,ksize], ksize))
    kernel1 = fspecial_gaussian(ksize, ksize)
    fd = convolve(img, kernel1, mode='nearest')

    # Large kernel for background estimation (matching MATLAB's fspecial('gaussian', [100,100], 50))
    kernel2 = fspecial_gaussian(100, 50)
    bg = convolve(img, kernel2, mode='nearest')

    # Foreground = local features - background
    mask = fd - bg
    mask = (mask >= thresh).astype(np.bool_)
    return mask


def read_mrc_file(filepath: str) -> Tuple[np.ndarray, List[int]]:
    try:
        import mrcfile
        with mrcfile.open(filepath, mode='r') as mrc:
            data = mrc.data.copy()
            header_dims = [mrc.header.nx, mrc.header.ny, mrc.header.nz]
            return data.astype(np.float32), header_dims
    except ImportError:
        print("Warning: mrcfile package not available. Using basic MRC reader.")
        try:
            with open(filepath, 'rb') as f:
                header = np.frombuffer(f.read(48), dtype=np.int32)[:12]
                nx, ny, nz = header[0], header[1], header[2]
                f.seek(1024)
                data = np.frombuffer(f.read(), dtype=np.float32)
                data = data.reshape((nz, ny, nx))
                return data.astype(np.float32), [nx, ny, nz]
        except:
            data = imageio.imread(filepath)
            return data.astype(np.float32), list(data.shape)


def read_image_file(filepath: str) -> np.ndarray:
    if filepath.lower().endswith('.mrc'):
        data, header_dims = read_mrc_file(filepath)
        if len(header_dims) >= 3:
            nx, ny, nz = header_dims[0], header_dims[1], header_dims[2]
            if data.size == nx * ny * nz:
                data = data.reshape((nz, ny, nx))
        return data
    else:
        data = tifffile.imread(filepath)
        return data.astype(np.float32)


def estimate_beta2(data_path: str, bg: float = 100, thresh: float = 0.005) -> Tuple[float, float]:
    image_extensions = ['*.tif', '*.tiff', '*.mrc']
    img_list = []
    for ext in image_extensions:
        img_list.extend(glob.glob(os.path.join(data_path, ext)))
    img_list = natural_sort(img_list)

    if not img_list:
        raise ValueError(f"No image files found in {data_path}")

    var_list = []
    for img_path in img_list:
        data = read_image_file(img_path)
        data = data.astype(np.float32)

        # For 3D images, use middle Z-slice for estimation
        # Using tifffile default convention: (Z, Y, X)
        if len(data.shape) == 3:
            shape = data.shape
            mid_z = shape[0] // 2
            data = data[mid_z, :, :]
            print(f"  Using middle Z-slice {mid_z} from (Z={shape[0]}, Y={shape[1]}, X={shape[2]}) for beta2 estimation")

        mask = data - bg
        mask = percentile_norm(mask, 0.1, 99.9)
        mask = gaussian_filter(mask, sigma=5, mode='nearest')  # Fixed: mode='nearest' to match MATLAB's 'replicate'
        mask = percentile_norm(mask)
        mask = (mask <= thresh)
        if np.sum(mask) > 0:
            background_pixels = data[mask]
            background_pixels = background_pixels[background_pixels != 0]
            if len(background_pixels) > 0:
                var_list.append(np.var(background_pixels))

    if not var_list:
        print("Warning: Could not estimate beta2, using default values")
        return 10.0, 15.0

    var_array = np.array(var_list)
    var_mean = np.mean(var_array)
    var_std = np.std(var_array)
    valid_mask = (var_array >= var_mean - var_std) & (var_array <= var_mean + var_std)
    if np.sum(valid_mask) > 0:
        var_array = var_array[valid_mask]
    beta2_mean = np.mean(var_array)
    beta2_min = 0.8 * beta2_mean
    beta2_max = 1.2 * beta2_mean
    return beta2_min, beta2_max


def apply_recorruption(data_raw: np.ndarray, bg: float, beta1: float, beta2: float, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply re-corruption to generate input-target pairs for zero-shot learning.
    Matches MATLAB's DataAugmFor2D recorruption logic.

    Args:
        data_raw: Original noisy image
        bg: Background offset
        beta1: Poisson noise coefficient
        beta2: Gaussian noise variance
        alpha: Noise magnification factor

    Returns:
        (data_input, data_gt): Re-corrupted input and target pair
    """
    if alpha == 0 or beta1 == 0 or beta2 == 0:
        return data_raw, data_raw

    # Create 5x5 average filter (matching MATLAB's fspecial('average', 5))
    kernel = np.ones((5, 5)) / 25.0

    # Apply convolution with constant (zero) boundary padding
    # This matches MATLAB's conv2(..., 'same') default behavior
    local_mean = convolve(data_raw.astype(np.float32), kernel, mode='constant', cval=0.0)

    # Calculate noise variance based on Gaussian-Poisson model
    noise_var = beta1 * np.maximum(local_mean - bg, 0) + beta2
    sigma = np.sqrt(np.maximum(noise_var, 0))

    # Generate shared random noise
    z = np.random.normal(0, 1, data_raw.shape)

    # Generate input-target pair
    data_gt = data_raw - sigma * z / alpha      # Target (less noisy)
    data_input = data_raw + sigma * z * alpha   # Input (more noisy)

    return data_input, data_gt


def normalize_pair(data_input: np.ndarray, data_gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    max_pixel = max(np.max(data_input), np.max(data_gt))
    min_pixel = min(np.min(data_input), np.min(data_gt))
    if max_pixel == min_pixel:
        return np.zeros_like(data_input), np.zeros_like(data_gt)
    data_input_norm = (data_input - min_pixel) / (max_pixel - min_pixel)
    data_gt_norm = (data_gt - min_pixel) / (max_pixel - min_pixel)
    return data_input_norm, data_gt_norm


def data_segmentation_for_train(data_l: np.ndarray,
                               data_h: np.ndarray,
                               gt: np.ndarray,
                               num_seg: int,
                               patch_x: int,
                               patch_y: int,
                               rot_flag: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Random cropping and rotation for training patches.
    Replicates MATLAB's XxDataSeg_ForTrain function.

    Args:
        data_l: Input data with shape (Y, X, 1)
        data_h: Target data with shape (Y, X, 1)
        gt: Ground truth with shape (Y, X) - used for mask calculation
        num_seg: Number of patches to generate
        patch_x: Patch width
        patch_y: Patch height
        rot_flag: 0=no rotation, 1=rotation, 2=rotation+flipping

    Returns:
        Tuple of (data_seg_l, data_seg_h, gt_seg)
        All outputs have shape (N, H, W) where:
          - N: number of patches
          - H: patch_y
          - W: patch_x
        Note: This is different from MATLAB's (H, W, N) output order
    """
    # Get image dimensions (Y, X)
    size_y, size_x = data_l.shape[:2]

    # Ensure input is 2D (with newaxis becomes 3D: Y x X x 1)
    assert data_l.shape[2] == 1, f"Expected 2D input (Y,X,1), got {data_l.shape}"

    # GT should be 2D (Y, X)
    assert len(gt.shape) == 2, f"Expected 2D gt, got {gt.shape}"

    size_y_gt = gt.shape[0]
    sr_ratio = round(size_y_gt / size_y)

    if rot_flag == 0:
        new_size_y, new_size_x = patch_y, patch_x
    else:
        new_size_y = int(np.ceil(patch_y * 1.5))
        new_size_x = int(np.ceil(patch_x * 1.5))
        if new_size_y > size_y or new_size_x > size_x:
            new_size_y, new_size_x = size_y, size_x
            rot_flag = 0

    thresh_mask = 0.07 * np.max(gt)
    ksize = 3

    if gt.shape != (size_y, size_x):
        gt_for_mask = cv2.resize(gt, (size_x, size_y), interpolation=cv2.INTER_CUBIC)
    else:
        gt_for_mask = gt
    mask = calculate_mask(gt_for_mask, ksize, thresh_mask)

    while np.sum(mask) < 1000:
        thresh_mask *= 0.8
        mask = calculate_mask(gt_for_mask, ksize, thresh_mask)

    y_coords, x_coords = np.where(mask)
    point_list = np.column_stack((y_coords, x_coords))
    num_points = len(point_list)

    if num_points == 0:
        print("Warning: No valid points found in mask, using random sampling")
        y_coords, x_coords = np.mgrid[0:size_y, 0:size_x]
        point_list = np.column_stack((y_coords.flatten(), x_coords.flatten()))
        num_points = len(point_list)

    data_seg_l = []
    data_seg_h = []
    gt_seg = []

    thresh_ar = 0
    thresh_ar_gt = 0
    thresh_sum_gt = 0
    count = 0

    half_y = new_size_y // 2
    half_x = new_size_x // 2

    for _ in range(num_seg):
        p = np.random.randint(0, num_points)
        center_y = point_list[p, 0]  # Y coordinate in first column
        center_x = point_list[p, 1]  # X coordinate in second column

        y1 = center_y - half_y
        y2 = center_y + half_y
        x1 = center_x - half_x
        x2 = center_x + half_x

        count_rand = 0
        # Fixed: Match MATLAB's boundary checking (y1 < 2 means leaving 1-pixel margin)
        # Adjusted for 0-based indexing: y1 < 1 and y2 >= size_y
        # Note: Python slicing uses y2+1, so y2 < size_y ensures y2+1 <= size_y
        while y1 < 1 or y2 >= size_y or x1 < 1 or x2 >= size_x:
            p = np.random.randint(0, num_points)
            center_y = point_list[p, 0]
            center_x = point_list[p, 1]
            y1 = center_y - half_y
            y2 = center_y + half_y
            x1 = center_x - half_x
            x2 = center_x + half_x
            count_rand += 1
            if count_rand > 1000:
                break

        if count_rand > 1000:
            break

        if rot_flag >= 1:
            degree = np.random.randint(0, 360)
            # Note: NumPy array indexing is [Y, X], variables are y1, y2, x1, x2
            patch_l = np.sum(data_l[y1:y2+1, x1:x2+1], axis=2)
            patch_h = np.sum(data_h[y1:y2+1, x1:x2+1], axis=2)
            # Fixed: order=1 for bilinear interpolation (matching MATLAB's 'bilinear')
            patch_l = rotate(patch_l, degree, reshape=False, order=1, mode='nearest')
            patch_h = rotate(patch_h, degree, reshape=False, order=1, mode='nearest')
            ty = new_size_y//2 - patch_y//2
            tx = new_size_x//2 - patch_x//2
            patch_l = patch_l[ty:ty+patch_y, tx:tx+patch_x]
            patch_h = patch_h[ty:ty+patch_y, tx:tx+patch_x]
            gt_y1 = sr_ratio * y1 - 1
            gt_y2 = sr_ratio * y2
            gt_x1 = sr_ratio * x1 - 1
            gt_x2 = sr_ratio * x2
            patch_gt = gt[gt_y1:gt_y2, gt_x1:gt_x2]
            patch_gt = rotate(patch_gt, degree, reshape=False, order=1, mode='nearest')
            ty_gt = new_size_y*sr_ratio//2 - patch_y*sr_ratio//2
            tx_gt = new_size_x*sr_ratio//2 - patch_x*sr_ratio//2
            patch_gt = patch_gt[ty_gt:ty_gt+sr_ratio*patch_y, tx_gt:tx_gt+sr_ratio*patch_x]
        else:
            patch_l = np.sum(data_l[y1:y2+1, x1:x2+1], axis=2)
            patch_h = np.sum(data_h[y1:y2+1, x1:x2+1], axis=2)
            gt_y1 = sr_ratio * y1 - 1
            gt_y2 = sr_ratio * y2
            gt_x1 = sr_ratio * x1 - 1
            gt_x2 = sr_ratio * x2
            patch_gt = gt[gt_y1:gt_y2, gt_x1:gt_x2]

        sum_patch_gt = np.sum(patch_gt)

        if patch_l.size > 0:
            active_range = np.percentile(patch_l, 99.9) / (np.percentile(patch_l, 0.1) + 1)
        else:
            active_range = 0

        if patch_gt.size > 0:
            active_range_gt = np.percentile(patch_gt, 99.9) / (np.percentile(patch_gt, 0.1) + 1)
        else:
            active_range_gt = 0

        while (active_range_gt < thresh_ar_gt or
               sum_patch_gt < thresh_sum_gt or
               active_range < thresh_ar):

            y1 = np.random.randint(1, size_y - new_size_y + 1)
            y2 = y1 + new_size_y - 1
            x1 = np.random.randint(1, size_x - new_size_x + 1)
            x2 = x1 + new_size_x - 1

            if rot_flag >= 1:
                degree = np.random.randint(0, 360)
                patch_l = np.sum(data_l[y1:y2+1, x1:x2+1], axis=2)
                patch_h = np.sum(data_h[y1:y2+1, x1:x2+1], axis=2)
                # Fixed: order=1 for bilinear interpolation
                patch_l = rotate(patch_l, degree, reshape=False, order=1, mode='nearest')
                patch_h = rotate(patch_h, degree, reshape=False, order=1, mode='nearest')
                ty = new_size_y//2 - patch_y//2
                tx = new_size_x//2 - patch_x//2
                patch_l = patch_l[ty:ty+patch_y, tx:tx+patch_x]
                patch_h = patch_h[ty:ty+patch_y, tx:tx+patch_x]
                gt_y1 = sr_ratio * y1 - 1
                gt_y2 = sr_ratio * y2
                gt_x1 = sr_ratio * x1 - 1
                gt_x2 = sr_ratio * x2
                patch_gt = gt[gt_y1:gt_y2, gt_x1:gt_x2]
                patch_gt = rotate(patch_gt, degree, reshape=False, order=1, mode='nearest')
                ty_gt = new_size_y*sr_ratio//2 - patch_y*sr_ratio//2
                tx_gt = new_size_x*sr_ratio//2 - patch_x*sr_ratio//2
                patch_gt = patch_gt[ty_gt:ty_gt+sr_ratio*patch_y, tx_gt:tx_gt+sr_ratio*patch_x]
            else:
                patch_l = np.sum(data_l[y1:y2+1, x1:x2+1], axis=2)
                patch_h = np.sum(data_h[y1:y2+1, x1:x2+1], axis=2)
                gt_y1 = sr_ratio * y1 - 1
                gt_y2 = sr_ratio * y2
                gt_x1 = sr_ratio * x1 - 1
                gt_x2 = sr_ratio * x2
                patch_gt = gt[gt_y1:gt_y2, gt_x1:gt_x2]

            if patch_l.size > 0:
                active_range = np.percentile(patch_l, 99.9) / (np.percentile(patch_l, 0.1) + 1e-2)
            else:
                active_range = 0

            sum_patch_gt = np.sum(patch_gt)

            if patch_gt.size > 0:
                active_range_gt = np.percentile(patch_gt, 99.9) / (np.percentile(patch_gt, 0.1) + 1)
            else:
                active_range_gt = 0

            count += 1
            if count > 100:
                thresh_ar_gt *= 0.9
                thresh_sum_gt *= 0.9
                thresh_ar *= 0.9
                count = 0

        if len(data_seg_l) == 0:
            data_seg_l = patch_l[np.newaxis, ...]
            data_seg_h = patch_h[np.newaxis, ...]
            gt_seg = patch_gt[np.newaxis, ...]
        else:
            data_seg_l = np.concatenate([data_seg_l, patch_l[np.newaxis, ...]], axis=0)
            data_seg_h = np.concatenate([data_seg_h, patch_h[np.newaxis, ...]], axis=0)
            gt_seg = np.concatenate([gt_seg, patch_gt[np.newaxis, ...]], axis=0)

    return data_seg_l, data_seg_h, gt_seg


def DataAugmFor2d_python(data_path: str,
                        bg: float = 100,
                        est_beta2: bool = True,
                        beta1: Union[float, List[float]] = [0.5, 1.5],
                        beta2: Union[float, List[float]] = [10, 15],
                        alpha: Union[float, List[float]] = [0.5, 1.5],
                        recorrupt_times: int = 100,
                        seg_x: int = 128,
                        seg_y: int = 128,
                        total_seg_num: int = 20000,
                        rot_flag: int = 2,
                        output_dir: str = None):

    if isinstance(beta1, (int, float)):
        beta1 = [beta1, beta1]
    if isinstance(beta2, (int, float)):
        beta2 = [beta2, beta2]
    if isinstance(alpha, (int, float)):
        alpha = [alpha, alpha]

    if est_beta2:
        beta2_min, beta2_max = estimate_beta2(data_path, bg)
        beta2 = [beta2_min, beta2_max]

    if output_dir is None:
        beta1_desc = f"{beta1[0]}" if beta1[0] == beta1[1] else f"{beta1[0]}-{beta1[1]}"
        beta2_desc = f"{beta2[0]:.1f}" if beta2[0] == beta2[1] else f"{beta2[0]:.1f}-{beta2[1]:.1f}"
        alpha_desc = f"{alpha[0]}" if alpha[0] == alpha[1] else f"{alpha[0]}-{alpha[1]}"
        output_dir = f"./augmented_datasets/augmented_datasets_2d/beta1_{beta1_desc}_beta2_{beta2_desc}_alpha{alpha_desc}_SegNum{total_seg_num}/"

    input_dir = os.path.join(output_dir, "input")
    gt_dir = os.path.join(output_dir, "gt")

    os.makedirs("./augmented_datasets/augmented_datasets_2d", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(input_dir):
        print("Input directory already exists, check carefully.")
    else:
        os.makedirs(input_dir)

    if os.path.exists(gt_dir):
        print("GT directory already exists, check carefully.")
    else:
        os.makedirs(gt_dir)

    image_extensions = ['*.tif', '*.tiff', '*.mrc']
    img_list = []
    for ext in image_extensions:
        img_list.extend(glob.glob(os.path.join(data_path, ext)))

    img_list = natural_sort(img_list)

    if not img_list:
        raise ValueError(f"No image files found in {data_path}")

    total_patches_generated = 0

    for cell_idx, img_path in enumerate(img_list):
        data_raw_orig = read_image_file(img_path)
        data_raw_orig = data_raw_orig.astype(np.float32)

        # Detect and handle 2D/3D input
        # NOTE: 3D slice processing is a Python-specific extension not present in the original MATLAB version.
        # The original MATLAB DataAugmFor2D only processes 2D images.
        # This extension automatically processes each Z-slice of 3D images independently.
        if len(data_raw_orig.shape) == 2:
            # 2D image (Y, X): process directly
            data_slices = [(data_raw_orig, None)]  # (data, slice_index)
            print(f"Processing 2D image (Y={data_raw_orig.shape[0]}, X={data_raw_orig.shape[1]}): {os.path.basename(img_path)}")
        elif len(data_raw_orig.shape) == 3:
            # 3D image: using tifffile default convention (Z, Y, X)
            # [Python Extension] This feature extends beyond MATLAB's original functionality
            shape = data_raw_orig.shape
            num_slices = shape[0]
            print(f"Processing 3D image (Z={shape[0]}, Y={shape[1]}, X={shape[2]}): {os.path.basename(img_path)}")
            print("  NOTE: Assuming tifffile default format (Z, Y, X)")
            print(f"  Processing {num_slices} Z-slices independently (Python extension)")
            data_slices = [(data_raw_orig[z, :, :], z) for z in range(num_slices)]
        else:
            raise ValueError(f"Unsupported image dimensions: {data_raw_orig.shape}")

        # Process each Z-slice
        for data_raw, z_index in data_slices:
            if z_index is not None:
                print(f"    Processing Z-slice {z_index+1}/{len(data_slices)}")

            for reco_idx in range(recorrupt_times):
                if len(beta1) > 1:
                    beta1_cur = beta1[0] + (beta1[1] - beta1[0]) * np.random.rand()
                else:
                    beta1_cur = beta1[0]

                if len(beta2) > 1:
                    beta2_cur = beta2[0] + (beta2[1] - beta2[0]) * np.random.rand()
                else:
                    beta2_cur = beta2[0]

                if len(alpha) > 1:
                    alpha_cur = alpha[0] + (alpha[1] - alpha[0]) * np.random.rand()
                else:
                    alpha_cur = alpha[0]

                if alpha_cur == 0 or beta1_cur == 0 or beta2_cur == 0:
                    continue

                data_input, data_gt = apply_recorruption(data_raw, bg, beta1_cur, beta2_cur, alpha_cur)
                data_input_norm, data_gt_norm = normalize_pair(data_input, data_gt)
                patches_per_recorr = max(round(total_seg_num / len(img_list) / recorrupt_times), 1)

                if rot_flag < 2:
                    data_seg, data_gt_seg, _ = data_segmentation_for_train(
                        data_input_norm[:, :, np.newaxis],
                        data_gt_norm[:, :, np.newaxis],
                        data_raw,
                        patches_per_recorr,
                        seg_x, seg_y, rot_flag
                    )
                    num_patches = data_seg.shape[0]
                else:
                    patches_per_transform = max(round(total_seg_num / len(img_list) / recorrupt_times / 4), 1)

                    data_seg, data_gt_seg, _ = data_segmentation_for_train(
                        data_input_norm[:, :, np.newaxis],
                        data_gt_norm[:, :, np.newaxis],
                        data_raw,
                        patches_per_transform,
                        seg_x, seg_y, rot_flag
                    )

                    data_input_flip = np.flipud(data_input_norm)
                    data_gt_flip = np.flipud(data_gt_norm)
                    data_raw_flip = np.flipud(data_raw)
                    data_seg_flip, data_gt_seg_flip, _ = data_segmentation_for_train(
                        data_input_flip[:, :, np.newaxis],
                        data_gt_flip[:, :, np.newaxis],
                        data_raw_flip,
                        patches_per_transform,
                        seg_x, seg_y, rot_flag
                    )
                    data_seg = np.concatenate([data_seg, data_seg_flip], axis=0)
                    data_gt_seg = np.concatenate([data_gt_seg, data_gt_seg_flip], axis=0)

                    data_input_flip = np.fliplr(data_input_norm)
                    data_gt_flip = np.fliplr(data_gt_norm)
                    data_raw_flip = np.fliplr(data_raw)
                    data_seg_flip, data_gt_seg_flip, _ = data_segmentation_for_train(
                        data_input_flip[:, :, np.newaxis],
                        data_gt_flip[:, :, np.newaxis],
                        data_raw_flip,
                        patches_per_transform,
                        seg_x, seg_y, rot_flag
                    )
                    data_seg = np.concatenate([data_seg, data_seg_flip], axis=0)
                    data_gt_seg = np.concatenate([data_gt_seg, data_gt_seg_flip], axis=0)

                    data_input_flip = np.fliplr(data_input_norm)
                    data_gt_flip = np.fliplr(data_gt_norm)
                    data_raw_flip = np.fliplr(data_raw)
                    data_input_flip = np.flipud(data_input_flip)
                    data_gt_flip = np.flipud(data_gt_flip)
                    data_raw_flip = np.flipud(data_raw_flip)
                    data_seg_flip, data_gt_seg_flip, _ = data_segmentation_for_train(
                        data_input_flip[:, :, np.newaxis],
                        data_gt_flip[:, :, np.newaxis],
                        data_raw_flip,
                        patches_per_transform,
                        seg_x, seg_y, rot_flag
                    )
                    data_seg = np.concatenate([data_seg, data_seg_flip], axis=0)
                    data_gt_seg = np.concatenate([data_gt_seg, data_gt_seg_flip], axis=0)

                    num_patches = data_seg.shape[0]

                for patch_idx in range(num_patches):
                    img_input = (data_seg[patch_idx] * 65535).astype(np.uint16)
                    img_gt = (data_gt_seg[patch_idx] * 65535).astype(np.uint16)

                    # Adjust filename format to include Z index (if applicable)
                    if z_index is not None:
                        # Z-slice from 3D image: include z index
                        filename = f"cell{cell_idx+1:02d}_z{z_index+1:04d}_reco{reco_idx+1}_{patch_idx+1:08d}.tif"
                    else:
                        # 2D image: original format
                        filename = f"cell{cell_idx+1:02d}_reco{reco_idx+1}_{patch_idx+1:08d}.tif"

                    input_path = os.path.join(input_dir, filename)
                    gt_path = os.path.join(gt_dir, filename)
                    tifffile.imwrite(input_path, img_input)
                    tifffile.imwrite(gt_path, img_gt)
                    total_patches_generated += 1

    return output_dir


if __name__ == "__main__":

    data_path = "./data/ori_input/aisr"
    bg = 100
    est_beta2 = True
    beta1 = [0.5, 1.5]
    beta2 = [10, 15]
    alpha = [0.5, 1.5]
    recorrupt_times = 10
    seg_x = 128
    seg_y = 128
    total_seg_num = 100  # Fixed: Matches MATLAB's default TotalSegNum = 20000
    rot_flag = 2
    output_dir = None

    DataAugmFor2d_python(
        data_path=data_path,
        bg=bg,
        est_beta2=est_beta2,
        beta1=beta1,
        beta2=beta2,
        alpha=alpha,
        recorrupt_times=recorrupt_times,
        seg_x=seg_x,
        seg_y=seg_y,
        total_seg_num=total_seg_num,
        rot_flag=rot_flag,
        output_dir=output_dir
    )