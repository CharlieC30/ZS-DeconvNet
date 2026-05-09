cd "$(dirname "$0")/../../Python_MATLAB_Codes/train_inference_python"

export TF_FORCE_GPU_ALLOW_GROWTH=true
# export CUDA_VISIBLE_DEVICES=2

python Infer_3D.py \
  --input_dir '../../Raw_Data/examples/inputs/iUExM.tif' \
  --load_weights_path '../../outputs/trained_models/3d/iUExM_example_run_twostage_RCAN3D__PSF_XY1.88um_Z15.04um_oddZ_111__upsample0__dx0.5dz2/weights_500.h5' \
  --model 'twostage_RCAN3D' \
  --num_seg_window_x 4 \
  --num_seg_window_y 4 \
  --num_seg_window_z 4 \
  --overlap_x 20 \
  --overlap_y 20 \
  --overlap_z 4 \
  --insert_xy 8 \
  --insert_z 2 \
  --upsample_flag 0 \
  --Fourier_damping_flag 0
