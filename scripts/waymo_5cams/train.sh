## gpu
CUDA_IDX=0

## need edit
scene_idx_all=("0" "3" "31" "233" "551" "621")
dataset=waymo/5cams
config_file=configs/omnire_ms_bilateral_extended.yaml

start_timestep=0
end_timestep=-1
test_image_stride=10
load_smpl=False
# output_root=/baai-cwm-1/baai_cwm_ml/algorithm/nan.wang/code/drivestudio/output/nuscenes_6cams/ablation/littletv
output_root="your output path"
# project_root=/baai-cwm-1/baai_cwm_ml/algorithm/nan.wang/code/bilateraldriving_0211/ablation-0305/notv
project_root="your project path"
# data_root=/baai-cwm-1/baai_cwm_ml/algorithm/nan.wang/processed_data/drivestudio/nuscenes/processed_10Hz/trainval
data_root="your data root"

cd $project_root
export PYTHONPATH=$(pwd)
# export TORCH_HOME=/baai-cwm-1/baai_cwm_ml/algorithm/nan.wang/.cache/torch
# export conda_env=/baai-cwm-1/baai_cwm_ml/algorithm/nan.wang/miniconda3/envs/bilateraldriving/bin/python


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
