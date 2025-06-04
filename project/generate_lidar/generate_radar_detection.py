import numpy as np
from generate_lidar.visual_bbox_lidar import get_corners
import torch
import json
import os

def generate_cornrs(frame_idx, instances_info, frame_instances, lidar_to_world, output_path, remove_obj_id_list=None):
    """
    Generate corners for the given image and frame index.

    Args:
        image (np.ndarray): The input image.
        frame_idx (int): The index of the frame.
        instances_info (dict): Dictionary containing instance information.
        frame_instances (dict): Dictionary containing frame instances.

    Returns:
        np.ndarray: The image with corners drawn on it.
    """
    # Get the objects in the current frame
    bbox = {}
    objects = frame_instances[str(frame_idx)]
    if len(objects) == 0:
        return
    for obj_id in objects:
        obj_id = str(obj_id)
        if remove_obj_id_list is not None:
            if obj_id in remove_obj_id_list:
                continue
        idx_in_obj = instances_info[obj_id]['frame_annotations']['frame_idx'].index(frame_idx)
        o2w = np.array(
            instances_info[obj_id]['frame_annotations']['obj_to_world'][idx_in_obj]
        )
        length, width, height = instances_info[obj_id]['frame_annotations']['box_size'][idx_in_obj]
        corners = get_corners(length, width, height)
        corners_world = (o2w[:3, :3] @ corners + o2w[:3, 3:4]).T

        world_to_lidar = lidar_to_world.inverse().numpy()

        corners_lidar = (
            world_to_lidar[:3, :3] @ corners_world.T + world_to_lidar[:3, 3:4]
        ).T
        # bbox.append(corners_lidar)
        bbox[obj_id] = {
            "class_name": instances_info[obj_id]['class_name'],
            "bbox": corners_lidar.tolist()
        }
    # bbox = np.stack(bbox, axis=0)
    # print(bbox.shape)
    # np.save(os.path.join(output_path, f"{frame_idx:0>3d}.npy"), bbox)
    with open(os.path.join(output_path, f"{frame_idx:0>3d}.json"), 'w') as f:
        json.dump(bbox, f, indent=4)
