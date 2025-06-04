## need edit
scene_idx_all=("152" "164" "171" "200" "209" "359" "529" "916")
output_root=/baai-cwm-1/baai_cwm_ml/algorithm/nan.wang/code/drivestudio/output/nuscenes_6cams/ablation/notv

## no need edit
project_root=/baai-cwm-1/baai_cwm_ml/algorithm/nan.wang/code/bilateraldriving_0211/ablation-0305/notv

cd $project_root
export PYTHONPATH=$(pwd)
export TORCH_HOME=/baai-cwm-1/baai_cwm_ml/algorithm/nan.wang/.cache/torch
export conda_env=/baai-cwm-1/baai_cwm_ml/algorithm/nan.wang/miniconda3/envs/bilateraldriving/bin/python

for scene_idx in "${scene_idx_all[@]}"; do
    $conda_env tools/eval_chamfer.py --resume_from $output_root/$scene_idx/checkpoint_final.pth
done

chmod 777 -R $output_root