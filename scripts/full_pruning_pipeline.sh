source_path=${SCENE_DATA_PATH}
orig_path=${SCENE_MODEL_PATH}
scene_name=${SCENE_NAME}
start_iteration=30000
prune_percent=0.66

directory=./experiments/${scene_name}
mkdir -p $directory/point_cloud/iteration_${start_iteration}
cp -r $orig_path/cameras.json $directory/cameras.json
cp -r $orig_path/cfg_args $directory/cfg_args
cp -r $orig_path/point_cloud/iteration_${start_iteration}/point_cloud.ply $directory/point_cloud/iteration_${start_iteration}/point_cloud.ply

CUDA_VISIBLE_DEVICES=0 python prune_finetune.py \
    -s $source_path \
    -m $directory \
    --eval \
    --start_pointcloud ${directory}/point_cloud/iteration_${start_iteration}/point_cloud.ply \
    --prune_percent $prune_percent \
    --position_lr_max_steps 35000 \
    --iterations 40000 \
    --fisher-via-cuda \
    --port 6071

python render.py \
    --source_path $source_path --skip_train \
    -m $directory

python metrics.py --model_paths $directory
