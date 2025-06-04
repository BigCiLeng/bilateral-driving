## gpu
CUDA_IDX=0

## project path
output_root=../output
project_root=project
data_root=../data/nuscenes/processed_10Hz/trainval

## dataset config
scene_idx_all=(152 164 171 200 209 359 529 916)
dataset=nuscenes/6cams
config_file=configs/omnire_ms_bilateral_extended.yaml

start_timestep=0
end_timestep=-1
test_image_stride=10
load_smpl=False

## train
cd $project_root
export PYTHONPATH=$(pwd)
for scene_idx in "${scene_idx_all[@]}"; do
    CUDA_VISIBLE_DEVICES=$CUDA_IDX python tools/train.py \
        --config_file $config_file \
        --output_root $output_root \
        --project ./ \
        --run_name $scene_idx \
        dataset=$dataset \
        data.data_root=$data_root \
        data.scene_idx=$scene_idx \
        data.start_timestep=$start_timestep \
        data.end_timestep=$end_timestep \
        data.pixel_source.test_image_stride=$test_image_stride \
        data.pixel_source.load_smpl=$load_smpl
done
