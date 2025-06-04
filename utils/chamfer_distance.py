import numpy as np

import cv2
import numpy as np
from typing import Literal, Tuple, Union

import torch
import torch.nn.functional as F

# def chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     NOTE: Not memory efficient, OOM when x,y are large.
#     """
#     assert (x.dim() == y.dim()) and (x.dim() >= 2) and x.shape[-1] == y.shape[-1]
#     x_i = x.unsqueeze(-2) # [..., N1, 1,  D]
#     y_j = x.unsqueeze(-3) # [..., 1,  N2, D]
#     D_ij = ((x_i - y_j)**2).sum(dim=-1) # [..., N1, N2]
#     cham_x = D_ij.min(dim=-1).values
#     cham_y = D_ij.min(dim=-2).values    
#     return cham_x, cham_y
def vis_cd(x, cham_x, name="test.ply"):
    import open3d as o3d
    import matplotlib.pyplot as plt
    points = x[cham_x<5].cpu().numpy()
    errors = cham_x[cham_x<5].cpu().numpy()
    errors_normalized = (errors - errors.min()) / (errors.max() - errors.min())
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    colormap = plt.cm.viridis
    colors = colormap(errors_normalized)[:, :3]
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(name, point_cloud)

def chamfer_distance(x: torch.Tensor, y: torch.Tensor, norm: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    from pytorch3d.ops.knn import knn_points
    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    _x = x[None] if len(x.shape) == 2 else x
    _y = y[None] if len(y.shape) == 2 else y

    if _y.shape[0] != _x.shape[0] or _y.shape[2] != _x.shape[2]:
        raise ValueError("y does not have the correct shape.")

    x_nn = knn_points(_x, _y, norm=norm, K=1)
    y_nn = knn_points(_y, _x, norm=norm, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)
    cham_x = cham_x[0] if len(x.shape) == 2 else cham_x
    cham_y = cham_y[0] if len(y.shape) == 2 else cham_y
    return cham_x, cham_y

def depth_map_to_point_cloud(depth_map, K, c2w, valid_mask=None):
    if valid_mask is None:
        valid_mask = torch.ones_like(depth_map, dtype=torch.bool)
    v_coords, u_coords = torch.where(valid_mask)
    z = depth_map[v_coords, u_coords]

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    x_cam = (u_coords.float() - cx) * z / fx
    y_cam = (v_coords.float() - cy) * z / fy
    z_cam = z
    points_cam = torch.stack((x_cam, y_cam, z_cam), dim=1)

    ones = torch.ones((points_cam.shape[0], 1), dtype=points_cam.dtype, device=points_cam.device)
    points_cam_hom = torch.cat((points_cam, ones), dim=1)

    points_world_hom = (c2w @ points_cam_hom.T).T
    points_world = points_world_hom[:, :3]

    return points_world