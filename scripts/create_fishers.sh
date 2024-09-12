source_path=${SCENE_DATA_PATH}
orig_path=${SCENE_MODEL_PATH}

CUDA_VISIBLE_DEVICES=0 python fisher_pool_xyz_scaling.py \
  -s ${source_path} \
  -m ${orig_path} \
  --iteration 30_000 \
  --pool-resolution 1 \
  --fisher-via-cuda

