import numpy as np
import tifffile
import imageio
from pathlib import Path
from scipy.signal import convolve
import argparse

def load_3d_psf(psf_path):
    """Load 3D PSF and normalize.

    Returns:
        Tuple of (psf array, psf info dict)
    """
    psf_raw = np.float32(imageio.mimread(psf_path))
    psf = np.transpose(psf_raw, [1, 2, 0])

    if psf.shape[2] % 2 == 0:
        print("WARNING: PSF depth is even")

    psf = psf / np.sum(psf)
    psf_info = {'sum': np.sum(psf), 'shape': psf.shape}

    return psf, psf_info

def load_3d_image(image_path):
    """Load and normalize 3D image.

    Returns:
        Tuple of (image array, image info dict)
    """
    image_raw = np.array(imageio.mimread(image_path)).astype(np.float32)

    if len(image_raw.shape) == 3:
        image = np.transpose(image_raw, [1, 2, 0])
    elif len(image_raw.shape) == 2:
        image = image_raw[:, :, np.newaxis]
    else:
        raise ValueError(f"Unsupported image dimensions: {image_raw.shape}")

    if np.max(image) > 1.0:
        image = image / 65535.0

    image_info = {'shape': image.shape}
    return image, image_info

def apply_3d_blur(image, psf):
    """Apply 3D convolution blur.

    Returns:
        Tuple of (blurred image, energy conservation ratio)
    """
    blurred = convolve(image, psf, mode='same')
    conservation = np.sum(blurred) / np.sum(image)
    return blurred.astype(np.float32), conservation

def save_test_results(original, blurred, psf, output_dir, input_name, psf_name):
    """Save original and blurred images as 16-bit TIFF."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def normalize_for_save(arr):
        arr_norm = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)
        return (arr_norm * 65535).astype(np.uint16)

    input_stem = Path(input_name).stem
    psf_stem = Path(psf_name).stem

    blurred_uint16 = normalize_for_save(blurred)
    blurred_filename = f"blurred_{input_stem}_{psf_stem}.tif"
    tifffile.imwrite(output_dir / blurred_filename, np.transpose(blurred_uint16, (2, 0, 1)))

    original_uint16 = normalize_for_save(original)
    original_filename = f"original_{input_stem}.tif"
    tifffile.imwrite(output_dir / original_filename, np.transpose(original_uint16, (2, 0, 1)))

    return blurred_filename


def main():
    parser = argparse.ArgumentParser(description='Test 3D PSF convolution')
    parser.add_argument('--input', type=str, required=True, help='Input image filename')
    parser.add_argument('--psf', type=str, required=True, help='PSF filename')

    args = parser.parse_args()

    base_dir = Path(__file__).parent
    input_path = base_dir / "PSFtest" / "PSFtest_input" / args.input
    psf_path = base_dir / "PSFoutput" / "optical" / args.psf
    output_dir = base_dir / "PSFtest" / "PSFtest_output"

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return
    if not psf_path.exists():
        print(f"PSF not found: {psf_path}")
        return

    psf, psf_info = load_3d_psf(psf_path)
    test_image, _ = load_3d_image(input_path)
    blurred_image, conservation = apply_3d_blur(test_image, psf)
    output_file = save_test_results(test_image, blurred_image, psf, output_dir, args.input, args.psf)

    print(f"PSF sum: {psf_info['sum']:.6f}, Energy conservation: {conservation:.4f}")
    print(f"Saved: {output_file}")

if __name__ == "__main__":
    main()

# Example usage:
# python test_3dpsf.py --input roiC_crop_2050.tif --psf PSF_optical_NA1.1_lambda525_size79_Z23.tif