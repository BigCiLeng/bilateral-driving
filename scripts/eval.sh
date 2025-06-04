## need edit
scene_idx_all=(152 164 171 200 209 359 529 916)
output_root=../ckpts/nuscenes_pretrained_checkpoints

project_root=project

cd $project_root
export PYTHONPATH=$(pwd)

for scene_idx in "${scene_idx_all[@]}"; do
    python tools/eval_metrics.py --resume_from $output_root/$scene_idx/checkpoint_final.pth
done
