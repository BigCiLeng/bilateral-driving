# Unifying Appearance Codes and Bilateral Grids for Driving Scene Gaussian Splatting

<div align="center">
<a href="">
  <img alt="arXiv" src="https://img.shields.io/badge/arXiv-PDF-b31b1b">
</a>
<a href="https://bigcileng.github.io/bilateral-driving/">
    <img alt="Project Page" src="docs/media/badge-website.svg">
</a>
</div>

This repo contains the official code of our paper: [Unifying Appearance Codes and Bilateral Grids for Driving Scene Gaussian Splatting]().

> Authors: [Nan Wang](https://bigcileng.github.io/), [Yuantao Chen](https://tao-11-chen.github.io/), [Lixing Xiao](https://li-xingxiao.github.io/homepage/), [Weiqing Xiao](https://scholar.google.com.hk/citations?user=v0iwkScAAAAJ), [Bohan Li](https://scholar.google.com/citations?user=V-YdQiAAAAAJ), [Zhaoxi Chen](https://scholar.google.com/citations?user=HsV0WbwAAAAJ), [Chongjie Ye](https://github.com/hugoycj), [Shaocong Xu](https://daniellli.github.io/), [Saining Zhang](https://scholar.google.com.hk/citations?hl=en&user=P4efBMcAAAAJ), [Ziyang Yan](https://ziyangyan.github.io/), [Pierre Merriaux](https://scholar.google.com.hk/citations?hl=en&user=NMSccqAAAAAJ), [Lei Lei](https://github.com/Crescent-Saturn), [Tianfan Xue](https://tianfan.info/) and [Hao Zhao](https://sites.google.com/view/fromandto/)<sup>†</sup>  

## ✨ News

- June 6, 2025: Release paper.

- June 5, 2025: Release preprocessed data and checkpoints.

- June 4, 2025: Release code and project page.

## 📊 Overview

<p align="center">
  <img src="docs/media/paradigm.png" alt="Overview" width="100%"/>
</p>

We introduced Multi-Scale Bilateral Grids that unifies appearance codes and bilateral grids, significantly improves geometric accuracy in dynamic, decoupled autonomous driving scene reconstruction.

---

## 🚀 Table of Contents

- [Environment Setup](#environment-setup)
- [Dataset Preparation](#Dataset-Preparation)
- [Train Your Model](#train-your-model)
- [Evaluate the Pre-trained Model](#evaluate-the-pre-trained-models)
- [Render the trained models](#render-the-trained-models)
- [Preprocessed Data](#preprocessed-data)

### Environment Setup

First, create a new Conda environment and specify the Python version:

- python: 3.9.21
- pytorch: 2.2.0
- cuda: 12.1

```bash
## code
git clone --recursive https://github.com/BigCiLeng/bilateral-driving.git

## conda
conda create -n bilateraldriving python=3.9 -y
conda activate bilateraldriving

conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install git+https://github.com/NVlabs/nvdiffrast

## third party
cd third_party/smplx/
pip install -e .
cd ../..
```

### Dataset Preparation

You can download our preprocessed [nuScenes](#preprocessed-data) data for quick start.

Download the dataset and arrange it as the following directory tree,
```bash
|-- data
    |-- nuscenes
        |-- processed_10Hz
            ...
    |-- argoverse
        ...
    |-- pandaset
        ...
    |-- waymo
        ...
|-- docs
|-- project
...
```

or following the same data preprocessing pipeline from [drivestudio](https://github.com/ziyc/drivestudio?tab=readme-ov-file#-prepare-data).

<details>
<summary>Click to expand data process instruction</summary>

- Waymo: [Data Process Instruction](docs/Waymo.md)
- NuScenes: [Data Process Instruction](docs/NuScenes.md)
- ArgoVerse: [Data Process Instruction](docs/ArgoVerse.md)
- PandaSet: [Data Process Instruction](docs/Pandaset.md)

</details>


### Train your model


#### train scripts
```bash
bash scripts/train.sh
```

#### train a single scene
```bash
cd project
export PYTHONPATH=$(pwd)
python tools/train.py \
    --config_file configs/omnire_ms_bilateral.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=$dataset \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep \
    data.pixel_source.test_image_stride=$test_image_stride \
    data.pixel_source.load_smpl=$load_smpl
```

### Evaluate the pre-trained models
Download the pre-trained model [checkpoints](#preprocessed-data)  and arrange it as the following directory tree,
```bash
|-- ckpts
    |-- nuscenes_pretrained_checkpoints
    |-- pandaset_pretrained_checkpoints
    |-- waymo_pretrained_checkpoints
    |-- argoverse_pretrained_checkpoints
|-- data
|-- docs
|-- project
...
```
#### eval
```bash
cd project
export PYTHONPATH=$(pwd)
python tools/eval_metrics.py \
    --resume_from $output_root/checkpoint_final.pth
```
### Render the trained models

```bash
cd project
export PYTHONPATH=$(pwd)
python tools/render.py \
    --resume_from $output_root/checkpoint_final.pth
```


### Preprocessed Data

|         Resources       |          Download Link           |
|-------------------------|----------------------------------|
| Pre-trained checkpoints | [Google Drive](https://drive.google.com/file/d/1NJ5bmOARlte4_kMG8yZoU8nMSMVllLZp/view?usp=drive_link) |
| Pre-processed dataset   | [Google Drive](https://drive.google.com/file/d/1NJ5bmOARlte4_kMG8yZoU8nMSMVllLZp/view?usp=drive_link) |

# 🤝 Citation

If you find this repository helpful, please consider citing our paper:

```bibtex
todo
```

# 🙏🏿 Acknowledge
Thansk for these excellent open-source works and models: [DriveStudio](https://github.com/ziyc/drivestudio); [Bilarf](https://github.com/yuehaowang/bilarf); [Street Gaussians](https://github.com/zju3dv/street_gaussians);.