# Unifying Appearance Codes and Bilateral Grids for Driving Scene Gaussian Splatting

<div align="center">
<a href="https://arxiv.org/abs/2506.05280">
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

- [Unifying Appearance Codes and Bilateral Grids for Driving Scene Gaussian Splatting](#unifying-appearance-codes-and-bilateral-grids-for-driving-scene-gaussian-splatting)
  - [✨ News](#-news)
  - [📊 Overview](#-overview)
  - [🚀 Table of Contents](#-table-of-contents)
    - [Environment Setup](#environment-setup)
    - [Dataset Preparation](#dataset-preparation)
    - [Train your model](#train-your-model)
      - [training script](#training-script)
      - [train a single scene](#train-a-single-scene)
    - [Evaluate the pre-trained models](#evaluate-the-pre-trained-models)
      - [evaluation script](#evaluation-script)
      - [Evaluate a single scene](#evaluate-a-single-scene)
    - [Render the trained models](#render-the-trained-models)
    - [Preprocessed Data](#preprocessed-data)
- [🤝 Citation](#-citation)
- [🙏🏿 Acknowledge](#-acknowledge)

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
cd project/third_party/smplx/
pip install -e .
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


#### training script
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
#### evaluation script
```bash
bash scripts/eval.sh
```

#### Evaluate a single scene
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

<table>
  <tr>
    <th>Dataset</th>
    <th>Resources</th>
    <th>Download Link</th>
    <th>Resources</th>
    <th>Download Link</th>
  </tr>
  <tr>
    <td rowspan="2">nuScenes</td>
    <td rowspan="2">preprocessed data</td>
    <td rowspan="2"><a href="https://drive.google.com/file/d/1NJ5bmOARlte4_kMG8yZoU8nMSMVllLZp/view?usp=sharing">Google Drive</a></td>
    <td>pretrained models (3cams)</td>
    <td><a href="https://drive.google.com/file/d/1zjGxcmxk9m2g6YJVBA9SGlv4S43nA6Y1/view?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>pretrained models (6cams)</td>
    <td><a href="https://drive.google.com/file/d/1Yt9aWGpDdfyZis9As7n3LlmAQit2xNP-/view?usp=sharing">Google Drive</a></td>
  </tr>

  <tr>
    <td rowspan="2">Waymo</td>
    <td rowspan="2">preprocessed data</td>
    <td rowspan="2"><a href="https://drive.google.com/file/d/1mhuwCCJYSp7-OQcp4HE_3H6nOjvWezl8/view?usp=sharing">Google Drive</a></td>
    <td>pretrained models (3cams)</td>
    <td><a href="https://drive.google.com/file/d/175kzSnjiXhKKqMQvCKN-VrIP1lHttDjP/view?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>pretrained models (5cams)</td>
    <td><a href="https://drive.google.com/file/d/1OV6g1xyMj5ZAhqAICqKFSFbqHedxOLxH/view?usp=sharing">Google Drive</a></td>
  </tr>

  <tr>
    <td rowspan="2">Pandaset</td>
    <td rowspan="2">preprocessed data</td>
    <td rowspan="2"><a href="https://drive.google.com/file/d/145UQdPjdCFrVHLpQc0ToOyQeUTrabLav/view?usp=sharing">Google Drive</a></td>
    <td>pretrained models (3cams)</td>
    <td><a href="https://drive.google.com/file/d/1pzVPu7ZnVQFqSY9hgQt0Zc-VrMvK_df7/view?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>pretrained models (6cams)</td>
    <td><a href="https://drive.google.com/file/d/1v2iwAAfoLIhrLePyzJ_5MQ8kXSWJoPj1/view?usp=sharing">Google Drive</a></td>
  </tr>

  <tr>
    <td rowspan="2">Arrgoverse</td>
    <td rowspan="2">preprocessed data</td>
    <td rowspan="2"><a href="https://drive.google.com/file/d/1uUYxoql6GghSiW5W9BSDB6QqbJ1OsFIc/view?usp=sharing">Google Drive</a></td>
    <td>pretrained models (3cams)</td>
    <td><a href="https://drive.google.com/file/d/1buNZCgYDlkb0FGmUDzl5cmxdOrjEh-iC/view?usp=sharing">Google Drive</a></td>
  </tr>
  <tr>
    <td>pretrained models (7cams)</td>
    <td><a href="https://drive.google.com/file/d/1yt25-khxVsEvYOCq_BvBHZ9TKlCuNajq/view?usp=sharing">Google Drive</a></td>
  </tr>
</table>

# 🤝 Citation

If you find this repository helpful, please consider citing our paper:

```bibtex
@article{wang2025unifying,
  title={Unifying Appearance Codes and Bilateral Grids for Driving Scene Gaussian Splatting},
  author={Wang, Nan and Chen, Yuantao and Xiao, Lixing and Xiao, Weiqing and Li, Bohan and Chen, Zhaoxi and Ye, Chongjie and Xu, Shaocong and Zhang, Saining and Yan, Ziyang and others},
  journal={arXiv preprint arXiv:2506.05280},
  year={2025}
}
```

# 🙏🏿 Acknowledge
Thansk for these excellent open-source works and models: [DriveStudio](https://github.com/ziyc/drivestudio); [Bilarf](https://github.com/yuehaowang/bilarf); [Street Gaussians](https://github.com/zju3dv/street_gaussians);.