import numpy as np
from pyquaternion import Quaternion
import torch
import cv2
import hashlib
from typing import List, Optional, Tuple

def get_corners(l: float, w: float, h: float):
    """
    Get 8 corners of a 3D bounding box centered at origin.

    Args:
        l, w, h: length, width, height of the box

    Returns:
        (3, 8) array of corner coordinates
    """
    return np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2],
    ])

def color_mapper(id: str) -> tuple:
    # use SHA256 to hash the id
    hash_object = hashlib.sha256(id.encode())
    hash_hex = hash_object.hexdigest()
    
    r = int(hash_hex[0:2], 16)
    g = int(hash_hex[2:4], 16)
    b = int(hash_hex[4:6], 16)
    return (r, g, b)

def dump_3d_bbox_on_image(
    coords, img,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 4,
) -> None:
    canvas = img.copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    coords = coords.astype(np.int32)
    for index in range(coords.shape[0]):
        if isinstance(color, tuple):
            c = color
        elif isinstance(color, list):
            c = color[index]
        projected_points2d = coords[index]
        bbox = projected_points2d.tolist()
        cv2.line(canvas, bbox[0], bbox[1], c, thickness)
        cv2.line(canvas, bbox[0], bbox[4], c, thickness)
        cv2.line(canvas, bbox[0], bbox[3], c, thickness)
        cv2.line(canvas, bbox[1], bbox[2], c, thickness)
        cv2.line(canvas, bbox[1], bbox[5], c, thickness)
        cv2.line(canvas, bbox[2], bbox[3], c, thickness)
        cv2.line(canvas, bbox[2], bbox[6], c, thickness)
        cv2.line(canvas, bbox[3], bbox[7], c, thickness)
        cv2.line(canvas, bbox[4], bbox[7], c, thickness)
        cv2.line(canvas, bbox[4], bbox[5], c, thickness)
        cv2.line(canvas, bbox[5], bbox[6], c, thickness)
        cv2.line(canvas, bbox[6], bbox[7], c, thickness)
    canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    return canvas

def visual_bbox(image, frame_instances, instances_info, frame_idx, camera_intrinsic, cam_to_world):
    lstProj2d = []
    color_list = []
    objects = frame_instances[str(frame_idx)]
    if len(objects) == 0:
        return image
    for obj_id in objects:
        obj_id = str(obj_id)
        idx_in_obj = instances_info[obj_id]['frame_annotations']['frame_idx'].index(frame_idx)
        o2w = np.array(
            instances_info[obj_id]['frame_annotations']['obj_to_world'][idx_in_obj]
        )
        length, width, height = instances_info[obj_id]['frame_annotations']['box_size'][idx_in_obj]
        corners = get_corners(length, width, height)
        
        # Transform corners to world coordinates
        corners_world = (o2w[:3, :3] @ corners + o2w[:3, 3:4]).T

        corner2img = camera_intrinsic @ cam_to_world.inverse()
        
        # Project to 2D
        corners_2d = (
            corner2img[:3, :3] @ corners_world.T + corner2img[:3, 3:4]
        ).T # (num_pts, 3)
        depth = corners_2d[:, 2]
        corners_2d = corners_2d[:, :2] / (depth.unsqueeze(-1) + 1e-6)
        
        # Check if the object is in front of the camera and all corners are in the image
        corners_2d = corners_2d.T.numpy()
        depth = depth.numpy()
        in_front = np.all(depth > 0.1)
        in_image = np.all(corners_2d[0, :] >= 0) & np.all(corners_2d[0, :] < image.shape[1]) & \
                np.all(corners_2d[1, :] >= 0) & np.all(corners_2d[1, :] < image.shape[0])
        ok = in_front and in_image

        if ok:
            projected_points2d = corners_2d[:2, :].T
            lstProj2d.append(projected_points2d)
            color_list.append(color_mapper(str(obj_id)))
    lstProj2d = np.asarray(lstProj2d)
    img_plotted = dump_3d_bbox_on_image(coords=lstProj2d, img=image, color=color_list)
    return img_plotted

def depth_visualizer(depth_img):
    normalized_depth = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    normalized_depth = normalized_depth.astype(np.uint8)
    colormap = cv2.COLORMAP_VIRIDIS
    colored_image = cv2.applyColorMap(normalized_depth, colormap)
    return colored_image

def visual_lidar(image, lidar_points_local, intrinsic_4x4, cam_to_world, lidar_to_world, W, H):
    lidar_points = (
        lidar_to_world[:3, :3] @ lidar_points_local.T + lidar_to_world[:3, 3:4]
    ).T
    lidar2img = intrinsic_4x4 @ cam_to_world.inverse()
    lidar_points = (
        lidar2img[:3, :3] @ lidar_points.T + lidar2img[:3, 3:4]
    ).T # (num_pts, 3)
    
    depth = lidar_points[:, 2]
    cam_points = lidar_points[:, :2] / (depth.unsqueeze(-1) + 1e-6) # (num_pts, 2)
    valid_mask = (
        (cam_points[:, 0] >= 0)
        & (cam_points[:, 0] < W)
        & (cam_points[:, 1] >= 0)
        & (cam_points[:, 1] < H)
        & (depth > 0)
    ) # (num_pts, )
    depth = depth[valid_mask]
    _cam_points = cam_points[valid_mask]
    depth_map = torch.zeros(
        H, W
    )
    depth_map[
        _cam_points[:, 1].long(), _cam_points[:, 0].long()
    ] = depth.squeeze(-1)

    depth_img = depth_map.cpu().numpy()
    colored_image = depth_visualizer(depth_img)
    mask = (depth_map.unsqueeze(-1) > 0).cpu().numpy()
    lidar_on_image = image * (1 - mask) + colored_image * mask
    return lidar_on_image
