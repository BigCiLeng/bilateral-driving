import torch
import numpy as np
import cv2
import open3d as o3d

def pto_ang_map(velo_points, H=64, W=512, slice=1):
    """
    :param H: the row num of depth map, could be 64(default), 32, 16
    :param W: the col num of depth map
    :param slice: output every slice lines
    """

    dtheta = np.radians(0.4 * 64.0 / H)
    dphi = np.radians(90.0 / W)

    x, y, z, i = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2], velo_points[:, 3]

    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.sqrt(x ** 2 + y ** 2)
    d[d == 0] = 0.000001
    r[r == 0] = 0.000001
    phi = np.radians(45.) - np.arcsin(y / r)
    phi_ = (phi / dphi).astype(int)
    phi_[phi_ < 0] = 0
    phi_[phi_ >= W] = W - 1

    theta = np.radians(2.) - np.arcsin(z / d)
    theta_ = (theta / dtheta).astype(int)
    theta_[theta_ < 0] = 0
    theta_[theta_ >= H] = H - 1

    depth_map = - np.ones((H, W, 4))
    depth_map[theta_, phi_, 0] = x
    depth_map[theta_, phi_, 1] = y
    depth_map[theta_, phi_, 2] = z
    depth_map[theta_, phi_, 3] = i
    depth_map = depth_map[0::slice, :, :]
    depth_map = depth_map.reshape((-1, 4))
    depth_map = depth_map[depth_map[:, 0] != -1.0]
    return depth_map

def pto_ang_map_vertical(velo_points, H=64, W=512, slice=1, vertical_fov_up=10.0, vertical_fov_down=-30.0):
    """
    :param H: the row num of depth map
    :param W: the col num of depth map
    :param slice: output every slice lines
    :param vertical_fov_up: upper vertical field of view in degrees (default: 10.0)
    :param vertical_fov_down: lower vertical field of view in degrees (default: -30.0)
    """

    vertical_fov_up_radians = np.radians(vertical_fov_up)
    vertical_fov_down_radians = np.radians(vertical_fov_down)
    total_vertical_fov_radians = vertical_fov_up_radians - vertical_fov_down_radians
    dtheta = total_vertical_fov_radians / H

    x, y, z, i = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2], velo_points[:, 3]

    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.sqrt(x ** 2 + y ** 2)
    d[d == 0] = 0.000001
    r[r == 0] = 0.000001

    phi = np.radians(45.) - np.arcsin(y / r)
    phi_ = (phi / np.radians(90.0 / W)).astype(int)
    phi_[phi_ < 0] = 0
    phi_[phi_ >= W] = W - 1

    theta = np.arcsin(z / d)

    # Filter points based on the specified vertical FOV
    valid_indices = np.logical_and(theta >= vertical_fov_down_radians, theta <= vertical_fov_up_radians)
    x = x[valid_indices]
    y = y[valid_indices]
    z = z[valid_indices]
    i = i[valid_indices]
    theta = theta[valid_indices]

    # Shift and map theta to the range [0, H-1]
    theta_shifted = theta - vertical_fov_down_radians
    theta_ = (theta_shifted / dtheta).astype(int)
    theta_[theta_ < 0] = 0
    theta_[theta_ >= H] = H - 1

    depth_map = - np.ones((H, W, 4))
    depth_map[theta_, phi_[valid_indices], 0] = x
    depth_map[theta_, phi_[valid_indices], 1] = y
    depth_map[theta_, phi_[valid_indices], 2] = z
    depth_map[theta_, phi_[valid_indices], 3] = i
    depth_map = depth_map[0::slice, :, :]
    depth_map = depth_map.reshape((-1, 4))
    depth_map = depth_map[depth_map[:, 0] != -1.0]
    return depth_map


def depth_map_to_lidar_points(depth_map, cam_intrinsics, cam_to_world, width, height, lidar_to_world, H=32, W=256, slice=1, downscale_when_loading=1):
    """
    Converts a depth map to a point cloud in lidar coordinates.

    Args:
        depth_map (torch.Tensor): Depth map of shape (height, width).
        cam_intrinsics (torch.Tensor): Camera intrinsic matrix (3x3 or 4x4).
        cam_to_world (torch.Tensor): Camera to world transformation matrix (4x4).
        width (int): Width of the image.
        height (int): Height of the image.
        lidar_to_world (torch.Tensor): Transformation matrix from lidar to world coordinates (4x4).
        undistort (bool): Whether the depth map was generated using undistorted images.
        distortions (torch.Tensor, optional): Camera distortion coefficients.

    Returns:
        np.ndarray: Point cloud in lidar coordinates (N, 3).
    """
    device = depth_map.device

    fx = cam_intrinsics[0].item() / downscale_when_loading
    fy = cam_intrinsics[1].item() / downscale_when_loading
    cx = cam_intrinsics[2].item() / downscale_when_loading
    cy = cam_intrinsics[3].item() / downscale_when_loading

    # Create grid of pixel coordinates
    u, v = torch.meshgrid(torch.arange(width, dtype=torch.float32), torch.arange(height, dtype=torch.float32), indexing='xy')
    u = u.to(device)
    v = v.to(device)

    # Get depth values
    depth = depth_map[v.long(), u.long()]

    # Filter out invalid depth values
    valid_mask = (depth > 0.0) & (depth < 80.0)
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    depth_valid = depth[valid_mask]

    # 3D points in camera coordinates
    x_camera = (u_valid - cx) * depth_valid / fx
    y_camera = (v_valid - cy) * depth_valid / fy
    z_camera = depth_valid
    points_camera = torch.stack([x_camera, y_camera, z_camera, torch.ones_like(z_camera)], dim=1).T

    # Transform to world coordinates

    points_world_homogeneous = cam_to_world @ points_camera
    points_world = points_world_homogeneous[:3, :].T

    # Transform world coordinates to lidar coordinates
    world_to_lidar = torch.inverse(lidar_to_world)
    points_world_homogeneous = torch.cat([points_world, torch.ones((points_world.shape[0], 1), device=device)], dim=1).T
    points_lidar_homogeneous = world_to_lidar @ points_world_homogeneous
    pc_velo = points_lidar_homogeneous[:3, :].T.cpu().numpy()
    # pc_velo = pc_velo[:, [1, 0, 2]]
    pc_velo = np.concatenate([pc_velo, np.ones((pc_velo.shape[0], 1))], 1)
    # depth, width, height
    valid_inds = (pc_velo[:, 0] < 80) & \
                 (pc_velo[:, 0] >= 0) & \
                 (pc_velo[:, 1] < 50) & \
                 (pc_velo[:, 1] >= -50) & \
                 (pc_velo[:, 2] < 3) & \
                 (pc_velo[:, 2] >= -2.5)
    pc_velo = pc_velo[valid_inds]
    # lidars = pto_ang_map(pc_velo, H=H, W=W, slice=slice)[..., :3]
    lidars = pto_ang_map_vertical(pc_velo, H=H, W=W, slice=slice)[..., :3]
    # lidars = lidars[:, [1,0,2]]
    return lidars

def save_ply(pcd, name="output.ply"):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pcd)
    o3d.io.write_point_cloud(name, point_cloud)
