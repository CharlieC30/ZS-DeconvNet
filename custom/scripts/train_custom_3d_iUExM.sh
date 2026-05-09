cd "$(dirname "$0")/../../Python_MATLAB_Codes/train_inference_python"

python Train_ZSDeconvNet_3D.py \
  --psf_path '../../custom/psf/output/other/PSF_XY1.88um_Z15.04um_oddZ_111.tif' \
  --data_dir '../../outputs/augmented_datasets/3d/iUExM/' \
  --folder 'iUExM_example_run' \
  --test_images_path '../../Raw_Data/examples/test_images/082525-iUExM-roiC_cropcube128.tif' \
  --save_weights_dir '../../outputs/trained_models/3d/' \
  --save_weights_suffix '__PSF_XY1.88um_Z15.04um_oddZ_111__upsample0__dx0.5dz2' \
  --model 'twostage_RCAN3D' \
  --upsample_flag 0 \
  --iterations 500 \
  --test_interval 500 \
  --valid_interval 100 \
  --batch_size 2 \
  --gpu_id 2 \
  --gpu_memory_fraction 0.9 \
  --input_y 64 \
  --input_x 64 \
  --input_z 13 \
  --insert_xy 8 \
  --insert_z 2 \
  --dx 0.5 \
  --dz 2 \
  --dxpsf 1 \
  --dzpsf 1
