# official code of Unifying Appearance Codes and Bilateral Grids for Driving Scene Gaussian Splatting

## Installation

```bash

## code
git clone --recursive https://github.com/BigCiLeng/bilateraldriving.git

## conda (test)
conda create -n bilateraldriving python=3.9 -y
conda activate bilateraldriving
pip install -r requirements.txt
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install git+https://github.com/NVlabs/nvdiffrast

##third party
cd third_party/smplx/
pip install -e .
cd ../..
```
or Following the installation of [README](./README_omnire.md).

## Prepare Data

```bash
## in current code, you should put sequnces into a folder named as its index

data_root/
    0/
        annotations/
        camera0/
        ...
    1/
        annotations/
        ...
```


```
## preprocess
export PYTHONPATH=$(pwd)

python datasets/preprocess.py \
--data_root $data_root \
--target_dir data/leddartech/processed \
--dataset leddartech \
--start_idx 0 \
--num_scenes 1 \
--workers 1 \
--process_keys  images lidar calib dynamic_masks objects

## get sky mask use segformer

python datasets/tools/extract_masks.py \
    --data_root data/leddartech/processed \
    --segformer_path=$segformer_path \
    --checkpoint=$segformer_path/pretrained/segformer.b5.1024x1024.city.160k.pth \
    --start_idx 0 \
    --num_scenes 1 \
```

## Train&eval

```bash
## train
python tools/train.py \
    --config_file configs/omnire_ms_bilateral.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=leddartech/1cams \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep

## eval
python tools/eval_chamfer.py \
    --resume_from $output_root/checkpoint_final.pth \
    render.render_test=False
```
