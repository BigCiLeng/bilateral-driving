from typing import Literal, Dict, List, Optional, Callable
from tqdm import tqdm, trange
import numpy as np
import os
import logging
import imageio

import torch
from torch import Tensor
from torch.nn import functional as F
from skimage.metrics import structural_similarity as ssim
import math

from datasets.base import SplitWrapper
from models.trainers.base import BasicTrainer
from utils.visualization import (
    to8b,
    depth_visualizer,
)

logger = logging.getLogger()

def get_numpy(x: Tensor) -> np.ndarray:
    return x.squeeze().cpu().numpy()

def non_zero_mean(x: Tensor) -> float:
    return sum(x) / len(x) if len(x) > 0 else -1

def compute_psnr(prediction: Tensor, target: Tensor) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between the prediction and target tensors.

    Args:
        prediction (torch.Tensor): The predicted tensor.
        target (torch.Tensor): The target tensor.

    Returns:
        float: The PSNR value between the prediction and target tensors.
    """
    if not isinstance(prediction, Tensor):
        prediction = Tensor(prediction)
    if not isinstance(target, Tensor):
        target = Tensor(target).to(prediction.device)
    return (-10 * torch.log10(F.mse_loss(prediction, target))).item()


def render_images(
    trainer: BasicTrainer,
    dataset: SplitWrapper,
    compute_metrics: bool = False,
    compute_geo: bool = False,
    compute_error_map: bool = False,
    vis_indices: Optional[List[int]] = None
):
    """
    Render pixel-related outputs from a model.

    Args:
        ....skip obvious args
        compute_metrics (bool, optional): Whether to compute metrics. Defaults to False.
        vis_indices (Optional[List[int]], optional): Indices to visualize. Defaults to None.
    """
    trainer.set_eval()
    render_results = render(
        dataset,
        trainer=trainer,
        compute_metrics=compute_metrics,
        compute_geo=compute_geo,
        compute_error_map=compute_error_map,
        vis_indices=vis_indices
    )
    if compute_metrics:
        num_samples = len(dataset) if vis_indices is None else len(vis_indices)
        logger.info(f"Eval over {num_samples} images:")
        logger.info(f"\t Full Image  PSNR: {render_results['psnr']:.4f}")
        logger.info(f"\t Full Image  SSIM: {render_results['ssim']:.4f}")
        logger.info(f"\t Full Image LPIPS: {render_results['lpips']:.4f}")
        logger.info(f"\t     Non-Sky PSNR: {render_results['occupied_psnr']:.4f}")
        logger.info(f"\t     Non-Sky SSIM: {render_results['occupied_ssim']:.4f}")
        logger.info(f"\tDynamic-Only PSNR: {render_results['masked_psnr']:.4f}")
        logger.info(f"\tDynamic-Only SSIM: {render_results['masked_ssim']:.4f}")
        logger.info(f"\t  Human-Only PSNR: {render_results['human_psnr']:.4f}")
        logger.info(f"\t  Human-Only SSIM: {render_results['human_ssim']:.4f}")
        logger.info(f"\tVehicle-Only PSNR: {render_results['vehicle_psnr']:.4f}")
        logger.info(f"\tVehicle-Only SSIM: {render_results['vehicle_ssim']:.4f}")
        if compute_geo:
            logger.info(f"\t Chamfer Distance: {render_results['avg_chamfer']:.4f}")
            # add
            logger.info(f"\t Sky Chamfer Distance: {render_results.get('avg_chamfer_sky', -1):.4f}")
            logger.info(f"\t Dynamic Chamfer Distance: {render_results.get('avg_chamfer_dynamic', -1):.4f}")
            logger.info(f"\t Human Chamfer Distance: {render_results.get('avg_chamfer_human', -1):.4f}")
            logger.info(f"\t Vehicle Chamfer Distance: {render_results.get('avg_chamfer_vehicle', -1):.4f}")
            logger.info(f"\t Vehicle Chamfer Distance: {render_results.get('depth_err_median_squared', -1):.4f}")

    return render_results


def render(
    dataset: SplitWrapper,
    trainer: BasicTrainer = None,
    compute_metrics: bool = False,
    compute_geo: bool=False,
    compute_error_map: bool = False,
    vis_indices: Optional[List[int]] = None,
):
    """
    Renders a dataset utilizing a specified render function.

    Parameters:
        dataset: Dataset to render.
        trainer: Gaussian trainer, includes gaussian models and rendering modules
        compute_metrics: Optional; if True, the function will compute and return metrics. Default is False.
        compute_error_map: Optional; if True, the function will compute and return error maps. Default is False.
        vis_indices: Optional; if not None, the function will only render the specified indices. Default is None.
    """
    # rgbs
    rgbs, gt_rgbs, rgb_sky_blend, rgb_sky = [], [], [], []
    Background_rgbs, RigidNodes_rgbs, DeformableNodes_rgbs, SMPLNodes_rgbs, Dynamic_rgbs = [], [], [], [], []
    error_maps = []

    # depths
    depths, lidar_on_images = [], []
    Background_depths, RigidNodes_depths, DeformableNodes_depths, SMPLNodes_depths, Dynamic_depths = [], [], [], [], []

    # sky
    opacities, sky_masks = [], []
    Background_opacities, RigidNodes_opacities, DeformableNodes_opacities, SMPLNodes_opacities, Dynamic_opacities = [], [], [], [], []
    
    # misc
    cam_names, cam_ids = [], []

    if compute_metrics:
        psnrs, ssim_scores, lpipss = [], [], []
        masked_psnrs, masked_ssims = [], []
        human_psnrs, human_ssims = [], []
        vehicle_psnrs, vehicle_ssims = [], []
        occupied_psnrs, occupied_ssims = [], []

        # geometry
        chamfers_pred = []
        chamfers_pred_99 = []
        chamfers_pred_97 = []
        chamfers_pred_95 = []
        
        chamfers_gt = []
        chamfers_gt_99 = []
        chamfers_gt_97 = []
        chamfers_gt_95 = []
        
        if compute_geo:
            chamfers_all = []
            chamfers_all_99 = []
            chamfers_all_97 = []
            chamfers_all_95 = []
            
            depth_errors = []
            depth_errors_99 = []
            depth_errors_97 = []
            depth_errors_95 = []

            median_squared_depth_error = []
            # add
            chamfers_sky = []
            chamfers_dynamic = []
            chamfers_human = []
            chamfers_vehicle = []
            chamfers_background = []

    with torch.no_grad():
        indices = vis_indices if vis_indices is not None else range(len(dataset))
        camera_downscale = trainer._get_downscale_factor()
        for i in tqdm(indices, desc=f"rendering {dataset.split}", dynamic_ncols=True):
            # get image and camera infos
            image_infos, cam_infos = dataset.get_image(i, camera_downscale)
            for k, v in image_infos.items():
                if isinstance(v, Tensor):
                    image_infos[k] = v.cuda(non_blocking=True)
            for k, v in cam_infos.items():
                if isinstance(v, Tensor):
                    cam_infos[k] = v.cuda(non_blocking=True)
            # render the image
            results = trainer(image_infos, cam_infos)
            
            # ------------- clip rgb ------------- #
            for k, v in results.items():
                if isinstance(v, Tensor) and "rgb" in k:
                    results[k] = v.clamp(0., 1.)
            
            # ------------- cam names ------------- #
            cam_names.append(cam_infos["cam_name"])
            cam_ids.append(
                cam_infos["cam_id"].flatten()[0].cpu().numpy()
            )

            # ------------- rgb ------------- #
            rgb = results["rgb"]
            rgbs.append(get_numpy(rgb))
            if "pixels" in image_infos:
                gt_rgbs.append(get_numpy(image_infos["pixels"]))
                
            green_background = torch.tensor([0.0, 177, 64]) / 255.0
            green_background = green_background.to(rgb.device)
            if "Background_rgb" in results:
                Background_rgb = results["Background_rgb"] * results[
                    "Background_opacity"
                ] + green_background * (1 - results["Background_opacity"])
                Background_rgbs.append(get_numpy(Background_rgb))
            if "RigidNodes_rgb" in results:
                RigidNodes_rgb = results["RigidNodes_rgb"] * results[
                    "RigidNodes_opacity"
                ] + green_background * (1 - results["RigidNodes_opacity"])
                RigidNodes_rgbs.append(get_numpy(RigidNodes_rgb))
            if "DeformableNodes_rgb" in results:
                DeformableNodes_rgb = results["DeformableNodes_rgb"] * results[
                    "DeformableNodes_opacity"
                ] + green_background * (1 - results["DeformableNodes_opacity"])
                DeformableNodes_rgbs.append(get_numpy(DeformableNodes_rgb))
            if "SMPLNodes_rgb" in results:
                SMPLNodes_rgb = results["SMPLNodes_rgb"] * results[
                    "SMPLNodes_opacity"
                ] + green_background * (1 - results["SMPLNodes_opacity"])
                SMPLNodes_rgbs.append(get_numpy(SMPLNodes_rgb))
            if "Dynamic_rgb" in results:
                Dynamic_rgb = results["Dynamic_rgb"] * results[
                    "Dynamic_opacity"
                ] + green_background * (1 - results["Dynamic_opacity"])
                Dynamic_rgbs.append(get_numpy(Dynamic_rgb))
            if compute_error_map:
                # cal mean squared error
                error_map = (rgb - image_infos["pixels"]) ** 2
                error_map = error_map.mean(dim=-1, keepdim=True)
                # scale
                error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min())
                error_map = error_map.repeat_interleave(3, dim=-1)
                error_maps.append(get_numpy(error_map))
            if "rgb_sky_blend" in results:
                rgb_sky_blend.append(get_numpy(results["rgb_sky_blend"]))
            if "rgb_sky" in results:
                rgb_sky.append(get_numpy(results["rgb_sky"]))
            # ------------- depth ------------- #
            depth = results["depth"]
            depths.append(get_numpy(depth))
            # ------------- mask ------------- #
            if "opacity" in results:
                opacities.append(get_numpy(results["opacity"]))
            if "Background_depth" in results:
                Background_depths.append(get_numpy(results["Background_depth"]))
                Background_opacities.append(get_numpy(results["Background_opacity"]))
            if "RigidNodes_depth" in results:
                RigidNodes_depths.append(get_numpy(results["RigidNodes_depth"]))
                RigidNodes_opacities.append(get_numpy(results["RigidNodes_opacity"]))
            if "DeformableNodes_depth" in results:
                DeformableNodes_depths.append(get_numpy(results["DeformableNodes_depth"]))
                DeformableNodes_opacities.append(get_numpy(results["DeformableNodes_opacity"]))
            if "SMPLNodes_depth" in results:
                SMPLNodes_depths.append(get_numpy(results["SMPLNodes_depth"]))
                SMPLNodes_opacities.append(get_numpy(results["SMPLNodes_opacity"]))
            if "Dynamic_depth" in results:
                Dynamic_depths.append(get_numpy(results["Dynamic_depth"]))
                Dynamic_opacities.append(get_numpy(results["Dynamic_opacity"]))
            if "sky_masks" in image_infos:
                sky_masks.append(get_numpy(image_infos["sky_masks"]))
                
            # ------------- lidar ------------- #
            if "lidar_depth_map" in image_infos:
                depth_map = image_infos["lidar_depth_map"]
                depth_img = depth_map.cpu().numpy()
                depth_img = depth_visualizer(depth_img, depth_img > 0)
                mask = (depth_map.unsqueeze(-1) > 0).cpu().numpy()
                lidar_on_image = image_infos["pixels"].cpu().numpy() * (1 - mask) + depth_img * mask
                lidar_on_images.append(lidar_on_image)

            if compute_metrics:
                psnr = compute_psnr(rgb, image_infos["pixels"])
                ssim_score = ssim(
                    get_numpy(rgb),
                    get_numpy(image_infos["pixels"]),
                    data_range=1.0,
                    channel_axis=-1,
                )
                lpips = trainer.lpips(
                    rgb[None, ...].permute(0, 3, 1, 2),
                    image_infos["pixels"][None, ...].permute(0, 3, 1, 2)
                )
                logger.info(f"Frame {i}: PSNR {psnr:.4f}, SSIM {ssim_score:.4f}")
                # print(f"Frame {i}: PSNR {psnr:.4f}, SSIM {ssim_score:.4f}")
                psnrs.append(psnr)
                ssim_scores.append(ssim_score)
                lpipss.append(lpips.item())
                
                if "sky_masks" in image_infos:
                    occupied_mask = ~get_numpy(image_infos["sky_masks"]).astype(bool)
                    if occupied_mask.sum() > 0:
                        occupied_psnrs.append(
                            compute_psnr(
                                rgb[occupied_mask], image_infos["pixels"][occupied_mask]
                            )
                        )
                        occupied_ssims.append(
                            ssim(
                                get_numpy(rgb),
                                get_numpy(image_infos["pixels"]),
                                data_range=1.0,
                                channel_axis=-1,
                                full=True,
                            )[1][occupied_mask].mean()
                        )

                if "dynamic_masks" in image_infos:
                    dynamic_mask = get_numpy(image_infos["dynamic_masks"]).astype(bool)
                    if dynamic_mask.sum() > 0:
                        masked_psnrs.append(
                            compute_psnr(
                                rgb[dynamic_mask], image_infos["pixels"][dynamic_mask]
                            )
                        )
                        masked_ssims.append(
                            ssim(
                                get_numpy(rgb),
                                get_numpy(image_infos["pixels"]),
                                data_range=1.0,
                                channel_axis=-1,
                                full=True,
                            )[1][dynamic_mask].mean()
                        )
                
                if "human_masks" in image_infos:
                    human_mask = get_numpy(image_infos["human_masks"]).astype(bool)
                    if human_mask.sum() > 0:
                        human_psnrs.append(
                            compute_psnr(
                                rgb[human_mask], image_infos["pixels"][human_mask]
                            )
                        )
                        human_ssims.append(
                            ssim(
                                get_numpy(rgb),
                                get_numpy(image_infos["pixels"]),
                                data_range=1.0,
                                channel_axis=-1,
                                full=True,
                            )[1][human_mask].mean()
                        )
                
                if "vehicle_masks" in image_infos:
                    vehicle_mask = get_numpy(image_infos["vehicle_masks"]).astype(bool)
                    if vehicle_mask.sum() > 0:
                        vehicle_psnrs.append(
                            compute_psnr(
                                rgb[vehicle_mask], image_infos["pixels"][vehicle_mask]
                            )
                        )
                        vehicle_ssims.append(
                            ssim(
                                get_numpy(rgb),
                                get_numpy(image_infos["pixels"]),
                                data_range=1.0,
                                channel_axis=-1,
                                full=True,
                            )[1][vehicle_mask].mean()
                        )
                
                if "lidar_depth_map" in image_infos and compute_geo:
                    from utils.chamfer_distance import chamfer_distance, depth_map_to_point_cloud, vis_cd
                    # depth_map = image_infos["lidar_depth_map"]
                    if "egocar_masks" in image_infos:
                        # in the case of egocar, we need to mask out the egocar region
                        valid_loss_mask = (1.0 - image_infos["egocar_masks"]).float()
                    else:
                        valid_loss_mask = torch.ones_like(image_infos["sky_masks"])
                    gt_depth = image_infos["lidar_depth_map"] 
                    lidar_hit_mask = (gt_depth > 0).float() * valid_loss_mask
                    pred_depth = results["depth"]

                    pred_depth = pred_depth.squeeze()
                    gt_depth = gt_depth.squeeze()
                    if lidar_hit_mask is not None:
                        pred_depth = pred_depth * lidar_hit_mask
                        gt_depth = gt_depth * lidar_hit_mask
                    # valid_mask = (depth_map > 0).squeeze(-1)
                    valid_mask = (gt_depth > 0.01) & (gt_depth < 80.0) & (pred_depth > 0.0001) & (pred_depth < 80.0)
                    lidar_gt = depth_map_to_point_cloud(gt_depth, cam_infos["intrinsics"], cam_infos["camera_to_world"], valid_mask)
                    pred = depth_map_to_point_cloud(pred_depth, cam_infos["intrinsics"], cam_infos["camera_to_world"], valid_mask)
                    cham_pred, cham_gt = chamfer_distance(pred, lidar_gt)
                    cham_pred_sorted = torch.sort(cham_pred).values
                    cham_gt_sorted = torch.sort(cham_gt).values
                
                    mean_cham_pred = cham_pred.mean().item()
                    mean_cham_gt = cham_gt.mean().item()

                    mean_cham_pred_99 = cham_pred_sorted[0:int(cham_pred_sorted.numel()*0.99)].mean().item()
                    mean_cham_pred_97 = cham_pred_sorted[0:int(cham_pred_sorted.numel()*0.97)].mean().item()
                    mean_cham_pred_95 = cham_pred_sorted[0:int(cham_pred_sorted.numel()*0.95)].mean().item()
                    
                    mean_cham_gt_99 = cham_gt_sorted[0:int(cham_gt_sorted.numel()*0.99)].mean().item()
                    mean_cham_gt_97 = cham_gt_sorted[0:int(cham_gt_sorted.numel()*0.97)].mean().item()
                    mean_cham_gt_95 = cham_gt_sorted[0:int(cham_gt_sorted.numel()*0.95)].mean().item()

                    # rmse
                    pred_depth_0 = pred_depth[valid_mask]
                    gt_depth_0 = gt_depth[valid_mask]
                    depth_err_each_abs = (pred_depth_0-gt_depth_0).abs()
                    depth_err_each_squared = depth_err_each_abs.square()
                    depth_err_rmse = depth_err_each_squared.mean().sqrt().item()
                    depth_err_median_squared = depth_err_each_squared.median().item()

                    depth_err_each_abs_sorted = torch.sort(depth_err_each_abs)[0]
                    depth_err_rmse_99 = depth_err_each_abs_sorted[0:int(depth_err_each_abs_sorted.numel()*0.99)].square().mean().sqrt().item()
                    depth_err_rmse_97 = depth_err_each_abs_sorted[0:int(depth_err_each_abs_sorted.numel()*0.97)].square().mean().sqrt().item()
                    depth_err_rmse_95 = depth_err_each_abs_sorted[0:int(depth_err_each_abs_sorted.numel()*0.95)].square().mean().sqrt().item()


                    chamfers_pred.append(mean_cham_pred)
                    median_squared_depth_error.append(depth_err_median_squared)
                    chamfers_pred_99.append(mean_cham_pred_99)
                    chamfers_pred_97.append(mean_cham_pred_97)
                    chamfers_pred_95.append(mean_cham_pred_95)
                    
                    chamfers_gt.append(mean_cham_gt)
                    chamfers_gt_99.append(mean_cham_gt_99)
                    chamfers_gt_97.append(mean_cham_gt_97)
                    chamfers_gt_95.append(mean_cham_gt_95)
                    
                    chamfers_all.append(mean_cham_pred + mean_cham_gt)
                    chamfers_all_99.append(mean_cham_pred_99 + mean_cham_gt_99)
                    chamfers_all_97.append(mean_cham_pred_97 + mean_cham_gt_97)
                    chamfers_all_95.append(mean_cham_pred_95 + mean_cham_gt_95)

                    depth_errors.append(depth_err_rmse)
                    depth_errors_99.append(depth_err_rmse_99)
                    depth_errors_97.append(depth_err_rmse_97)
                    depth_errors_95.append(depth_err_rmse_95)

                     # add
                    if "sky_masks" in image_infos:
                        sky_mask = image_infos["sky_masks"].squeeze()
                        if sky_mask is not None:
                            sky_pred = pred_depth * sky_mask
                            sky_gt = gt_depth * sky_mask

                            sky_mask = (sky_mask > 0).bool()
                            valid_mask = (valid_mask > 0).bool()
                            sky_mask = sky_mask & valid_mask

                            lidar_sky_gt = depth_map_to_point_cloud(sky_gt, cam_infos["intrinsics"], cam_infos["camera_to_world"], sky_mask)
                            pred_sky = depth_map_to_point_cloud(sky_pred, cam_infos["intrinsics"], cam_infos["camera_to_world"], sky_mask)

                            cham_sky_pred, cham_sky_gt = chamfer_distance(pred_sky, lidar_sky_gt)
                            chamfer_sky_pred = cham_sky_pred.mean().item()
                            chamfer_sky_gt = cham_sky_gt.mean().item()

                            # chamfers_sky.append(chamfer_sky_pred + chamfer_sky_gt)
                            if not math.isnan(chamfer_sky_pred) and not math.isnan(chamfer_sky_gt):
                                chamfers_sky.append(chamfer_sky_pred + chamfer_sky_gt)
        
                    if "dynamic_masks" in image_infos:
                        # dynamic_mask = get_numpy(image_infos["dynamic_masks"].cpu()).astype(bool)
                        dynamic_mask = image_infos["dynamic_masks"].squeeze()
                        if dynamic_mask is not None:
                            dynamic_pred = pred_depth * dynamic_mask
                            dynamic_gt = gt_depth * dynamic_mask
                            
                            dynamic_mask = (dynamic_mask > 0).bool()
                            valid_mask = (valid_mask > 0).bool()
                            dynamic_mask = dynamic_mask & valid_mask

                            lidar_dynamic_gt = depth_map_to_point_cloud(dynamic_gt, cam_infos["intrinsics"], cam_infos["camera_to_world"], dynamic_mask)
                            pred_dynamic = depth_map_to_point_cloud(dynamic_pred, cam_infos["intrinsics"], cam_infos["camera_to_world"], dynamic_mask)

                            cham_dynamic_pred, cham_dynamic_gt = chamfer_distance(pred_dynamic, lidar_dynamic_gt)
                            chamfer_dynamic_pred = cham_dynamic_pred.mean().item()
                            chamfer_dynamic_gt = cham_dynamic_gt.mean().item()

                            # chamfers_dynamic.append(chamfer_dynamic_pred + chamfer_dynamic_gt)
                            if not math.isnan(chamfer_dynamic_pred) and not math.isnan(chamfer_dynamic_gt):
                                chamfers_dynamic.append(chamfer_dynamic_pred + chamfer_dynamic_gt)
    
                    if "human_masks" in image_infos:
                        # human_mask = get_numpy(image_infos["human_masks"].cpu()).astype(bool)
                        human_mask = image_infos["human_masks"].squeeze()
                        if human_mask is not None:
                            human_pred = pred_depth * human_mask
                            human_gt = gt_depth * human_mask

                            human_mask = (human_mask > 0).bool()
                            valid_mask = (valid_mask > 0).bool()
                            human_mask = human_mask & valid_mask
                            lidar_human_gt = depth_map_to_point_cloud(human_gt, cam_infos["intrinsics"], cam_infos["camera_to_world"], human_mask)
                            pred_human = depth_map_to_point_cloud(human_pred, cam_infos["intrinsics"], cam_infos["camera_to_world"], human_mask)

                            cham_human_pred, cham_human_gt = chamfer_distance(pred_human, lidar_human_gt)
                            chamfer_human_pred = cham_human_pred.mean().item()
                            chamfer_human_gt = cham_human_gt.mean().item()
            
                            # chamfers_human.append(chamfer_human_pred + chamfer_human_gt)
                            if not math.isnan(chamfer_human_pred) and not math.isnan(chamfer_human_gt):
                                chamfers_human.append(chamfer_human_pred + chamfer_human_gt)
                    
                    if "vehicle_masks" in image_infos:
                        # vehicle_mask = get_numpy(image_infos["vehicle_masks"].cpu()).astype(bool)
                        vehicle_mask = image_infos["vehicle_masks"].squeeze()
                        if vehicle_mask is not None:
                            vehicle_pred = pred_depth * vehicle_mask
                            vehicle_gt = gt_depth * vehicle_mask

                            vehicle_mask = (vehicle_mask > 0).bool()
                            valid_mask = (valid_mask > 0).bool()
                            vehicle_mask = vehicle_mask & valid_mask
                            lidar_vehicle_gt = depth_map_to_point_cloud(vehicle_gt, cam_infos["intrinsics"], cam_infos["camera_to_world"], vehicle_mask)
                            pred_vehicle = depth_map_to_point_cloud(vehicle_pred, cam_infos["intrinsics"], cam_infos["camera_to_world"], vehicle_mask)

                            cham_vehicle_pred, cham_vehicle_gt = chamfer_distance(pred_vehicle, lidar_vehicle_gt)
                            chamfer_vehicle_pred = cham_vehicle_pred.mean().item()
                            chamfer_vehicle_gt = cham_vehicle_gt.mean().item()
                            # chamfers_vehicle.append(chamfer_vehicle_pred + chamfer_vehicle_gt)
                            if not math.isnan(chamfer_vehicle_pred) and not math.isnan(chamfer_vehicle_gt):
                                chamfers_vehicle.append(chamfer_vehicle_pred + chamfer_vehicle_gt)

                    background_mask =  (~sky_mask) & (~dynamic_mask) & (~human_mask) & (~vehicle_mask)
                    if background_mask is not None:
                        background_pred = pred_depth * background_mask
                        background_gt = gt_depth * background_mask

                        background_mask = (background_mask > 0).bool()
                        valid_mask = (valid_mask > 0).bool()
                        background_mask = background_mask & valid_mask
                        lidar_background_gt = depth_map_to_point_cloud(background_gt, cam_infos["intrinsics"], cam_infos["camera_to_world"], background_mask)
                        pred_background = depth_map_to_point_cloud(background_pred, cam_infos["intrinsics"], cam_infos["camera_to_world"], background_mask)

                        cham_background_pred, cham_background_gt = chamfer_distance(pred_background, lidar_background_gt)
                        chamfer_background_pred = cham_background_pred.mean().item()
                        chamfer_background_gt = cham_background_gt.mean().item()
        
                        # chamfers_background.append(chamfer_background_pred + chamfer_background_gt)
                        if not math.isnan(chamfer_background_pred) and not math.isnan(chamfer_background_gt):
                                chamfers_background.append(chamfer_background_pred + chamfer_background_gt)



    # messy aggregation...
    results_dict = {}
    results_dict["psnr"] = non_zero_mean(psnrs) if compute_metrics else -1
    results_dict["ssim"] = non_zero_mean(ssim_scores) if compute_metrics else -1
    results_dict["lpips"] = non_zero_mean(lpipss) if compute_metrics else -1
    results_dict["occupied_psnr"] = non_zero_mean(occupied_psnrs) if compute_metrics else -1
    results_dict["occupied_ssim"] = non_zero_mean(occupied_ssims) if compute_metrics else -1
    results_dict["masked_psnr"] = non_zero_mean(masked_psnrs) if compute_metrics else -1
    results_dict["masked_ssim"] = non_zero_mean(masked_ssims) if compute_metrics else -1
    results_dict["human_psnr"] = non_zero_mean(human_psnrs) if compute_metrics else -1
    results_dict["human_ssim"] = non_zero_mean(human_ssims) if compute_metrics else -1
    results_dict["vehicle_psnr"] = non_zero_mean(vehicle_psnrs) if compute_metrics else -1
    results_dict["vehicle_ssim"] = non_zero_mean(vehicle_ssims) if compute_metrics else -1
    results_dict["rgbs"] = rgbs
    results_dict["depths"] = depths
    results_dict["cam_names"] = cam_names
    results_dict["cam_ids"] = cam_ids
    
    results_dict["avg_chamfer"] = non_zero_mean(chamfers_all) if compute_metrics and compute_geo else -1
    results_dict["avg_chamfer_99"] = non_zero_mean(chamfers_all_99) if compute_metrics and compute_geo else -1
    results_dict["avg_chamfer_97"] = non_zero_mean(chamfers_all_97) if compute_metrics and compute_geo else -1
    results_dict["avg_chamfer_95"] = non_zero_mean(chamfers_all_95) if compute_metrics and compute_geo else -1
    results_dict["depth_err"] = non_zero_mean(depth_errors) if compute_metrics and compute_geo else -1
    results_dict["depth_err_rmse_99"] = non_zero_mean(depth_errors_99) if compute_metrics and compute_geo else -1
    results_dict["depth_err_rmse_97"] = non_zero_mean(depth_errors_97) if compute_metrics and compute_geo else -1
    results_dict["depth_err_rmse_95"] = non_zero_mean(depth_errors_95) if compute_metrics and compute_geo else -1
    results_dict["depth_err_median_squared"] =  non_zero_mean(median_squared_depth_error) if compute_metrics and compute_geo else -1

    # add
    results_dict["avg_chamfer_sky"] = non_zero_mean(chamfers_sky) if "chamfers_sky" in locals() else -1
    results_dict["avg_chamfer_dynamic"] = non_zero_mean(chamfers_dynamic) if "chamfers_dynamic" in locals() else -1
    results_dict["avg_chamfer_human"] = non_zero_mean(chamfers_human) if "chamfers_human" in locals() else -1
    results_dict["avg_chamfer_vehicle"] = non_zero_mean(chamfers_vehicle) if "chamfers_vehicle" in locals() else -1
    results_dict["avg_chamfer_background"] = non_zero_mean(chamfers_background) if "chamfers_background" in locals() else -1
                 
    if len(opacities) > 0:
        results_dict["opacities"] = opacities
    if len(gt_rgbs) > 0:
        results_dict["gt_rgbs"] = gt_rgbs
    if len(error_maps) > 0:
        results_dict["rgb_error_maps"] = error_maps
    if len(rgb_sky_blend) > 0:
        results_dict["rgb_sky_blend"] = rgb_sky_blend
    if len(rgb_sky) > 0:
        results_dict["rgb_sky"] = rgb_sky
    if len(sky_masks) > 0:
        results_dict["gt_sky_masks"] = sky_masks
    if len(lidar_on_images) > 0:
        results_dict["lidar_on_images"] = lidar_on_images
    if len(Background_rgbs) > 0:
        results_dict["Background_rgbs"] = Background_rgbs
    if len(RigidNodes_rgbs) > 0:
        results_dict["RigidNodes_rgbs"] = RigidNodes_rgbs
    if len(DeformableNodes_rgbs) > 0:
        results_dict["DeformableNodes_rgbs"] = DeformableNodes_rgbs
    if len(SMPLNodes_rgbs) > 0:
        results_dict["SMPLNodes_rgbs"] = SMPLNodes_rgbs
    if len(Dynamic_rgbs) > 0:
        results_dict["Dynamic_rgbs"] = Dynamic_rgbs
    if len(Background_depths) > 0:
        results_dict["Background_depths"] = Background_depths
    if len(RigidNodes_depths) > 0:
        results_dict["RigidNodes_depths"] = RigidNodes_depths
    if len(DeformableNodes_depths) > 0:
        results_dict["DeformableNodes_depths"] = DeformableNodes_depths
    if len(SMPLNodes_depths) > 0:
        results_dict["SMPLNodes_depths"] = SMPLNodes_depths
    if len(Dynamic_depths) > 0:
        results_dict["Dynamic_depths"] = Dynamic_depths
    if len(Background_opacities) > 0:
        results_dict["Background_opacities"] = Background_opacities
    if len(RigidNodes_opacities) > 0:
        results_dict["RigidNodes_opacities"] = RigidNodes_opacities
    if len(DeformableNodes_opacities) > 0:
        results_dict["DeformableNodes_opacities"] = DeformableNodes_opacities
    if len(SMPLNodes_opacities) > 0:
        results_dict["SMPLNodes_opacities"] = SMPLNodes_opacities
    if len(Dynamic_opacities) > 0:
        results_dict["Dynamic_opacities"] = Dynamic_opacities
    return results_dict


def save_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    layout: Callable,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "depths"],
    num_cams: int = 3,
    save_seperate_video: bool = False,
    save_images: bool = False,
    fps: int = 10,
    verbose: bool = True,
):  
    if save_seperate_video:
        return_frame = save_seperate_videos(
            render_results,
            save_pth,
            layout,
            num_timestamps=num_timestamps,
            keys=keys,
            num_cams=num_cams,
            save_images=save_images,
            fps=fps,
            verbose=verbose,
        )
    else:
        return_frame = save_concatenated_videos(
            render_results,
            save_pth,
            layout,
            num_timestamps=num_timestamps,
            keys=keys,
            num_cams=num_cams,
            save_images=save_images,
            fps=fps,
            verbose=verbose,
        )
    return return_frame


def render_novel_views(trainer, render_data: list, save_path: str, fps: int = 30) -> None:
    """
    Perform rendering and save the result as a video.
    
    Args:
        trainer: Trainer object containing the rendering method
        render_data (list): List of dicts, each containing elements required for rendering a single frame
        save_path (str): Path to save the output video
        fps (int): Frames per second for the output video
    """
    trainer.set_eval()  
    
    writer = imageio.get_writer(save_path, mode='I', fps=fps)
    
    with torch.no_grad():
        for frame_data in render_data:
            # Move data to GPU
            for key, value in frame_data["cam_infos"].items():
                frame_data["cam_infos"][key] = value.cuda(non_blocking=True)
            for key, value in frame_data["image_infos"].items():
                frame_data["image_infos"][key] = value.cuda(non_blocking=True)
            
            # Perform rendering
            outputs = trainer(
                image_infos=frame_data["image_infos"],
                camera_infos=frame_data["cam_infos"],
                novel_view=True
            )
            
            # Extract RGB image and mask
            rgb = outputs["rgb"].cpu().numpy().clip(
                min=1.e-6, max=1-1.e-6
            )
            
            # Convert to uint8 and write to video
            rgb_uint8 = (rgb * 255).astype(np.uint8)
            writer.append_data(rgb_uint8)
    
    writer.close()
    print(f"Video saved to {save_path}")


def save_concatenated_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    layout: Callable,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "depths"],
    num_cams: int = 3,
    save_images: bool = False,
    fps: int = 10,
    verbose: bool = True,
):
    if num_timestamps == 1:  # it's an image
        writer = imageio.get_writer(save_pth, mode="I")
        return_frame_id = 0
    else:
        return_frame_id = num_timestamps // 2
        writer = imageio.get_writer(save_pth, mode="I", fps=fps)
    for i in trange(num_timestamps, desc="saving video", dynamic_ncols=True):
        merged_list = []
        cam_names = render_results["cam_names"][i * num_cams : (i + 1) * num_cams]
        for key in keys:
            # skip if the key is not in render_results
            if "mask" in key:
                new_key = key.replace("mask", "opacities")
                if new_key not in render_results or len(render_results[new_key]) == 0:
                    continue
                frames = render_results[new_key][i * num_cams : (i + 1) * num_cams]
            else:
                if key not in render_results or len(render_results[key]) == 0:
                    continue
                frames = render_results[key][i * num_cams : (i + 1) * num_cams]
            # convert to rgb if necessary
            if key == "gt_sky_masks":
                frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
            elif "mask" in key:
                frames = [
                    np.stack([frame, frame, frame], axis=-1) for frame in frames
                ]
            elif "depth" in key:
                try:
                    opacities = render_results[key.replace("depths", "opacities")][
                        i * num_cams : (i + 1) * num_cams
                    ]
                except:
                    if "median" in key:
                        opacities = render_results[
                            key.replace("median_depths", "opacities")
                        ][i * num_cams : (i + 1) * num_cams]
                    else:
                        continue
                frames = [
                    depth_visualizer(frame, opacity)
                    for frame, opacity in zip(frames, opacities)
                ]
            tiled_img = layout(frames, cam_names)
            # frames = np.concatenate(frames, axis=1)
            merged_list.append(tiled_img)
        merged_frame = to8b(np.concatenate(merged_list, axis=0))
        if i == return_frame_id:
            return_frame = merged_frame
        writer.append_data(merged_frame)
    writer.close()
    if verbose:
        logger.info(f"saved video to {save_pth}")
    del render_results
    return {"concatenated_frame": return_frame}


def save_seperate_videos(
    render_results: Dict[str, List[Tensor]],
    save_pth: str,
    layout: Callable,
    num_timestamps: int,
    keys: List[str] = ["gt_rgbs", "rgbs", "depths"],
    num_cams: int = 3,
    fps: int = 10,
    verbose: bool = False,
    save_images: bool = False,
):
    return_frame_id = num_timestamps // 2
    return_frame_dict = {}
    for key in keys:
        tmp_save_pth = save_pth.replace(".mp4", f"_{key}.mp4")
        tmp_save_pth = tmp_save_pth.replace(".png", f"_{key}.png")
        if num_timestamps == 1:  # it's an image
            writer = imageio.get_writer(tmp_save_pth, mode="I")
        else:
            writer = imageio.get_writer(tmp_save_pth, mode="I", fps=fps)
        if "mask" not in key:
            if key not in render_results or len(render_results[key]) == 0:
                continue
        for i in range(num_timestamps):
            cam_names = render_results["cam_names"][i * num_cams : (i + 1) * num_cams]
            # skip if the key is not in render_results
            if "mask" in key:
                new_key = key.replace("mask", "opacities")
                if new_key not in render_results or len(render_results[new_key]) == 0:
                    continue
                frames = render_results[new_key][i * num_cams : (i + 1) * num_cams]
            else:
                if key not in render_results or len(render_results[key]) == 0:
                    continue
                frames = render_results[key][i * num_cams : (i + 1) * num_cams]
            # convert to rgb if necessary
            if key == "gt_sky_masks":
                frames = [np.stack([frame, frame, frame], axis=-1) for frame in frames]
            elif "mask" in key:
                frames = [
                    np.stack([frame, frame, frame], axis=-1) for frame in frames
                ]
            elif "depth" in key:
                try:
                    opacities = render_results[key.replace("depths", "opacities")][
                        i * num_cams : (i + 1) * num_cams
                    ]
                except:
                    if "median" in key:
                        opacities = render_results[
                            key.replace("median_depths", "opacities")
                        ][i * num_cams : (i + 1) * num_cams]
                    else:
                        continue
                if save_images:
                    if i == 0:
                        os.makedirs(tmp_save_pth.replace(".mp4", "_np"), exist_ok=True)
                    for j, frame in enumerate(frames):
                        np.save(
                            tmp_save_pth.replace(".mp4", f"_np/{i:03d}_{j:03d}.npy"),
                            frame,
                        )
                frames = [
                    depth_visualizer(frame, opacity)
                    for frame, opacity in zip(frames, opacities)
                ]
            tiled_img = layout(frames, cam_names)
            if save_images:
                if i == 0:
                    os.makedirs(tmp_save_pth.replace(".mp4", ""), exist_ok=True)
                for j, frame in enumerate(frames):
                    imageio.imwrite(
                        tmp_save_pth.replace(".mp4", f"/{i:03d}_{j:03d}.png"),
                        to8b(frame),
                    )
            # frames = to8b(np.concatenate(frames, axis=1))
            frames = to8b(tiled_img)
            writer.append_data(frames)
            if i == return_frame_id:
                return_frame_dict[key] = frames
        # close the writer
        writer.close()
        del writer
        if verbose:
            logger.info(f"saved video to {tmp_save_pth}")
    del render_results
    return return_frame_dict

def render_original_scene(trainer, dataset: SplitWrapper, vis_indices: Optional[List[int]] = None) -> None:
    """
    Perform rendering for closed loops
    
    Args:
        trainer: Trainer object containing the rendering method
        render_data (list): List of dicts, each containing elements required for rendering a single frame
        save_path (str): Path to save the output video
    """
    # Load the first camera (front) pose to align the world
    rgbs, gt_rgbs = [], []
    # misc
    cam_names, cam_ids = [], []
    trainer.set_eval()  
    with torch.no_grad():
        indices = vis_indices if vis_indices is not None else range(len(dataset))
        camera_downscale = trainer._get_downscale_factor()
        for i in tqdm(indices, desc=f"rendering {dataset.split}", dynamic_ncols=True):
            image_infos, cam_infos = dataset.get_image(i, camera_downscale)

            for k, v in image_infos.items():
                if isinstance(v, Tensor):
                    image_infos[k] = v.cuda(non_blocking=True)
            for k, v in cam_infos.items():
                if isinstance(v, Tensor):
                    cam_infos[k] = v.cuda(non_blocking=True)
            # render the image
            results = trainer(image_infos, cam_infos, novel_view=True)
            for k, v in results.items():
                if isinstance(v, Tensor) and "rgb" in k:
                    results[k] = v.clamp(0., 1.)
            # ------------- cam names ------------- #
            cam_names.append(cam_infos["cam_name"])
            cam_ids.append(
                cam_infos["cam_id"].flatten()[0].cpu().numpy()
            )
            # ------------- rgb ------------- #
            rgb = results["rgb"]
            rgbs.append(get_numpy(rgb))
            if "pixels" in image_infos:
                gt_rgbs.append(get_numpy(image_infos["pixels"]))
    # messy aggregation...
    results_dict = {}
    results_dict["rgbs"] = rgbs
    results_dict["cam_names"] = cam_names
    results_dict["cam_ids"] = cam_ids
    return results_dict