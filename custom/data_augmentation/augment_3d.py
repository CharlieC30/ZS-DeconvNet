#!/usr/bin/env python3
"""
Python implementation of DataAugmFor3D.m
Fully replicated from MATLAB version
"""

import numpy as np
import tifffile
from scipy import ndimage
from pathlib import Path
import glob
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.ndimage import rotate

class DataAugmFor3D:
    def __init__(self,
                 data_path,
                 save_path='./augmented_datasets/augmented_datasets_3d/',
                 seg_x=64,
                 seg_y=64, 
                 seg_z=13,
                 seg_num=10000,
                 rot_flag=True):
        """
        Parameters correspond to MATLAB version
        """
        self.data_path = Path(data_path)
        self.save_path = Path(save_path)
        self.seg_x = seg_x
        self.seg_y = seg_y
        self.seg_z = seg_z
        self.seg_num = seg_num
        self.rot_flag = rot_flag

        # Threshold parameters (matches MATLAB)
        self.thresh_mask = 1e-2
        self.active_range_thresh = 0.5
        self.sum_thresh = 0.01 * seg_x * seg_y * seg_z

        # Create output directories
        self.save_training_path = self.save_path / f'Zsize{seg_z}_Xsize{seg_x}'
        self.input_path = self.save_training_path / 'input'
        self.gt_path = self.save_training_path / 'gt'
        
        self.input_path.mkdir(parents=True, exist_ok=True)
        self.gt_path.mkdir(parents=True, exist_ok=True)

        # Calculate padding for rotation
        if rot_flag:
            self.halfx = int(seg_x * 1.5 / 2)
            self.halfy = int(seg_y * 1.5 / 2)
            self.tx = self.halfx - seg_x // 2
            self.ty = self.halfy - seg_y // 2
        else:
            self.halfx = seg_x // 2
            self.halfy = seg_y // 2
    
    def xx_norm(self, data):
        """Normalize data to [0,1] range (replicated from XxNorm)"""
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max - data_min > 1e-6:
            return (data - data_min) / (data_max - data_min)
        return data
    
    def xx_cal_mask(self, data, sigma=10, threshold=1e-2):
        """
        Calculate valid region mask (replicated from XxCalMask)

        Args:
            data: 3D image array
            sigma: Gaussian filter sigma
            threshold: Threshold value
        """
        # Gaussian smoothing
        smoothed = ndimage.gaussian_filter(data, sigma=sigma)

        # Calculate mask
        mask = smoothed > threshold

        # Morphological operations: remove small regions
        mask = binary_erosion(mask, iterations=1)
        mask = binary_dilation(mask, iterations=2)
        mask = binary_erosion(mask, iterations=1)
        
        return mask
    
    def process_file(self, file_path):
        """Process single file"""
        print(f"Processing file: {file_path}")

        # Read data
        data = tifffile.imread(file_path).astype(np.float32)

        # Handle dimensions
        if len(data.shape) == 2:
            data = data[np.newaxis, ...]  # Add Z dimension

        print(f"Data shape: {data.shape} (Z, Y, X)")

        # Basic processing
        data[data < 0] = 0
        data = self.xx_norm(data)

        # Calculate mask
        cur_thresh = self.thresh_mask
        mask = self.xx_cal_mask(data, sigma=10, threshold=cur_thresh)

        # Lower threshold if valid region is too small
        ntry = 0
        while np.sum(mask) < 100 and ntry < 1000:
            cur_thresh *= 0.8
            mask = self.xx_cal_mask(data, sigma=10, threshold=cur_thresh)
            ntry += 1

        # Get valid point positions
        valid_points = np.where(mask)
        if len(valid_points[0]) == 0:
            print("Warning: No valid region found")
            return 0

        point_list = np.column_stack(valid_points)
        n_points = len(point_list)
        print(f"Found {n_points} valid points")

        # Calculate patches per file
        if self.data_path.is_file():
            n_per_file = self.seg_num
        else:
            files_count = len(glob.glob(str(self.data_path) + "/*"))
            n_per_file = max(self.seg_num // max(files_count, 1), 10)

        n_generated = 0
        max_attempts = 100000
        attempts = 0

        print(f"Generating {n_per_file} patches...")

        while n_generated < n_per_file and attempts < max_attempts:
            attempts += 1

            # Randomly select a point
            idx = np.random.randint(0, n_points)
            z, y, x = point_list[idx]

            # Calculate crop range
            z1 = max(0, z - self.seg_z)
            z2 = min(data.shape[0], z + self.seg_z)
            y1 = max(0, y - self.halfy)
            y2 = min(data.shape[1], y + self.halfy)
            x1 = max(0, x - self.halfx)
            x2 = min(data.shape[2], x + self.halfx)

            # Check sufficient Z slices
            available_z = z2 - z1
            if available_z < self.seg_z * 2:
                continue

            # Z-axis interval sampling (MATLAB behavior)
            # Calculate actual available Z range
            max_z_start = z2 - self.seg_z * 2
            z_start = np.random.randint(z1, max(z1+1, max_z_start+1))

            input_z_indices = np.arange(z_start+1, z_start+1+self.seg_z*2, 2)
            gt_z_indices = np.arange(z_start, z_start+self.seg_z*2, 2)

            # Ensure correct Z slice count
            if len(input_z_indices) != self.seg_z or len(gt_z_indices) != self.seg_z:
                continue

            # Check Z indices in range
            if (np.max(input_z_indices) >= data.shape[0] or
                np.max(gt_z_indices) >= data.shape[0]):
                continue

            # Extract patches
            input_crop = data[input_z_indices, y1:y2, x1:x2]
            gt_crop = data[gt_z_indices, y1:y2, x1:x2]

            # Check patch size
            if (input_crop.shape[1] < self.seg_y or input_crop.shape[2] < self.seg_x or
                gt_crop.shape[1] < self.seg_y or gt_crop.shape[2] < self.seg_x):
                continue

            # Adjust to correct size
            input_crop = input_crop[:self.seg_z, :self.seg_y, :self.seg_x]
            gt_crop = gt_crop[:self.seg_z, :self.seg_y, :self.seg_x]

            # Transpose to (Y, X, Z)
            input_crop = np.transpose(input_crop, (1, 2, 0))
            gt_crop = np.transpose(gt_crop, (1, 2, 0))

            # Quality checks
            # 1. Dynamic range check
            p99 = np.percentile(input_crop, 99.9)
            p01 = np.percentile(input_crop, 0.1) + 1e-2
            if p01 > 0:
                active_range = p99 / p01
                if active_range < self.active_range_thresh:
                    continue

            # 2. Total signal check
            sum_value = np.sum(input_crop)
            if sum_value < self.sum_thresh:
                continue

            # Generate augmented versions
            augmented_pairs = self.generate_augmentations(input_crop, gt_crop)

            # Save all augmented versions
            for inp, gt in augmented_pairs:
                if n_generated >= n_per_file:
                    break

                # Convert to uint16 and save
                inp_uint16 = (np.clip(inp, 0, 1) * 65535).astype(np.uint16)
                gt_uint16 = (np.clip(gt, 0, 1) * 65535).astype(np.uint16)

                # Transpose back to (Z, Y, X) for saving
                inp_save = np.transpose(inp_uint16, (2, 0, 1))
                gt_save = np.transpose(gt_uint16, (2, 0, 1))

                tifffile.imwrite(
                    str(self.input_path / f'{n_generated+1:05d}.tif'),
                    inp_save
                )
                tifffile.imwrite(
                    str(self.gt_path / f'{n_generated+1:05d}.tif'),
                    gt_save
                )

                n_generated += 1
                if n_generated % 10 == 0:
                    print(f"Generated patch {n_generated}/{n_per_file}")

        print(f"Completed: generated {n_generated} patches (attempted {attempts} times)")
        return n_generated
    
    def generate_augmentations(self, input_crop, gt_crop):
        """
        Generate all augmented versions
        Replicates MATLAB augmentation strategy
        """
        augmented_pairs = []

        # 1. Original + optional rotation
        if self.rot_flag:
            angle = np.random.uniform(0, 360)
            inp_rot = self.rotate_volume(input_crop, angle)
            gt_rot = self.rotate_volume(gt_crop, angle)
            augmented_pairs.append((inp_rot, gt_rot))
        else:
            augmented_pairs.append((input_crop.copy(), gt_crop.copy()))

        # 2. Vertical flip
        inp_flip_ud = np.flip(input_crop, axis=0)
        gt_flip_ud = np.flip(gt_crop, axis=0)
        if self.rot_flag:
            angle = np.random.uniform(0, 360)
            inp_flip_ud = self.rotate_volume(inp_flip_ud, angle)
            gt_flip_ud = self.rotate_volume(gt_flip_ud, angle)
        augmented_pairs.append((inp_flip_ud, gt_flip_ud))

        # 3. Horizontal flip
        inp_flip_lr = np.flip(input_crop, axis=1)
        gt_flip_lr = np.flip(gt_crop, axis=1)
        if self.rot_flag:
            angle = np.random.uniform(0, 360)
            inp_flip_lr = self.rotate_volume(inp_flip_lr, angle)
            gt_flip_lr = self.rotate_volume(gt_flip_lr, angle)
        augmented_pairs.append((inp_flip_lr, gt_flip_lr))

        return augmented_pairs
    
    def rotate_volume(self, volume, angle):
        """
        Rotate 3D volume (slice-by-slice)
        Replicates MATLAB imrotate behavior
        """
        if not self.rot_flag:
            return volume

        rotated = np.zeros((self.seg_x, self.seg_y, self.seg_z))

        # Create larger canvas for rotation
        canvas_size = max(int(self.seg_x * 1.5), int(self.seg_y * 1.5))

        for z in range(volume.shape[2]):
            # Place image on larger canvas
            canvas = np.zeros((canvas_size, canvas_size))
            start_y = (canvas_size - volume.shape[0]) // 2
            start_x = (canvas_size - volume.shape[1]) // 2
            canvas[start_y:start_y+volume.shape[0], start_x:start_x+volume.shape[1]] = volume[:, :, z]

            # Rotate
            slice_rot = rotate(canvas, angle, reshape=False, order=1, mode='constant', cval=0)

            # Crop to target size
            center_y = canvas_size // 2
            center_x = canvas_size // 2
            crop_y1 = center_y - self.seg_y // 2
            crop_y2 = crop_y1 + self.seg_y
            crop_x1 = center_x - self.seg_x // 2
            crop_x2 = crop_x1 + self.seg_x

            rotated[:, :, z] = slice_rot[crop_y1:crop_y2, crop_x1:crop_x2]

        return rotated
    
    def run(self):
        """Run data augmentation"""
        print("=" * 60)
        print("Starting Python DataAugmFor3D")
        print("=" * 60)
        print(f"Input path: {self.data_path}")
        print(f"Output path: {self.save_training_path}")
        print(f"Patch size: {self.seg_x}x{self.seg_y}x{self.seg_z}")
        print(f"Target count: {self.seg_num}")
        print(f"Rotation: {'Enabled' if self.rot_flag else 'Disabled'}")
        print()

        # Get all files
        if self.data_path.is_file():
            files = [self.data_path]
        else:
            files = list(self.data_path.glob('*.tif')) + \
                   list(self.data_path.glob('*.tiff'))

        if not files:
            print("Error: No TIFF files found")
            return

        print(f"Found {len(files)} files:")
        for f in files:
            print(f"  - {f.name}")
        print()

        total_generated = 0
        for i, file in enumerate(files):
            print(f"[{i+1}/{len(files)}] Processing file: {file.name}")
            n = self.process_file(file)
            total_generated += n
            print(f"Generated {n} patches from {file.name}")
            print()

        print("=" * 60)
        print("Data augmentation completed!")
        print("=" * 60)
        print(f"Total generated: {total_generated} training samples")
        print(f"Input folder: {self.input_path}")
        print(f"GT folder: {self.gt_path}")
        print()

        # Check output
        input_files = list(self.input_path.glob('*.tif'))
        gt_files = list(self.gt_path.glob('*.tif'))
        print(f"Verification: {len(input_files)} input files, {len(gt_files)} GT files")

        if len(input_files) > 0:
            # Check first file
            sample = tifffile.imread(str(input_files[0]))
            print(f"Sample file shape: {sample.shape}")
        
        return str(self.save_training_path)


# Example usage
if __name__ == "__main__":
    # Parameters correspond to MATLAB version
    augmentor = DataAugmFor3D(
        data_path='../../Raw_Data/lab_data/input/iUExM/iUExM_roi.tif',
        save_path='../../outputs/augmented_datasets/3d/iUExM/',
        seg_x=64,
        seg_y=64,
        seg_z=13,
        seg_num=100,
        rot_flag=True
    )

    # Run augmentation
    augmentor.run()