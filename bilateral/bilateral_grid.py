import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorly as tl
tl.set_backend('pytorch')
from tensorly.decomposition import parafac
from bilateral.slice import bilateral_slice


def color_affine_transform(affine_mats, rgb):
    """Applies color affine transformations.

    Args:
        affine_mats (torch.Tensor): Affine transformation matrices. Supported shape: $(..., 3, 4)$.
        rgb  (torch.Tensor): Input RGB values. Supported shape: $(..., 3)$.

    Returns:
        Output transformed colors of shape $(..., 3)$.
    """
    return torch.matmul(affine_mats[..., :3], rgb.unsqueeze(-1)).squeeze(-1) + affine_mats[..., 3]


_num_tensor_elems = lambda t: max(torch.prod(torch.tensor(t.size()[1:]).float()), 1.)

def total_variation_loss(x):
    """Returns total variation on multi-dimensional tensors.

    Args:
        x (torch.Tensor): The input tensor with shape $(B, C, ...)$, where $B$ is the batch size and $C$ is the channel size.
    """
    batch_size = x.shape[0]
    tv = 0
    for i in range(2, len(x.shape)):
        n_res = x.shape[i]
        idx1 = torch.arange(1, n_res).to(x.device)
        idx2 = torch.arange(0, n_res - 1).to(x.device)
        x1 = x.index_select(i, idx1)
        x2 = x.index_select(i, idx2)
        count = _num_tensor_elems(x1)
        tv += torch.pow((x1 - x2), 2).sum() / count
    return tv / batch_size


class BilateralGrid(nn.Module):
    """Class for 3D bilateral grids.

    Holds one or more than one bilateral grids.
    """

    def __init__(self, num, grid_X=16, grid_Y=16, grid_W=8):
        """
        Args:
            num (int): The number of bilateral grids (i.e., the number of views).
            grid_X (int): Defines grid width $W$.
            grid_Y (int): Defines grid height $H$.
            grid_W (int): Defines grid guidance dimension $L$.
        """
        super(BilateralGrid, self).__init__()

        self.grid_width = grid_X
        """Grid width. Type: int."""
        self.grid_height = grid_Y
        """Grid height. Type: int."""
        self.grid_guidance = grid_W
        """Grid guidance dimension. Type: int."""
        
        # self.use_mutual = False
        # self.use_depth = False
        
        # Initialize grids.
        grid = self._init_identity_grid()
        self.grids = nn.Parameter(grid.tile(num, 1, 1, 1, 1)) # (N, 12, L, H, W)
        # if self.use_mutual:
        #     self.depth_grids = nn.Parameter(grid.tile(num, 1, 1, 1, 1))
        """ A 5-D tensor of shape $(N, 12, L, H, W)$."""

        # Weights of BT601 RGB-to-gray.
        self.register_buffer('rgb2gray_weight', torch.Tensor([0.299, 0.587, 0.114]))
        # Define the lambda function to convert RGB to grayscale
        self.rgb2gray = lambda rgb: torch.tensordot(rgb, self.rgb2gray_weight, dims=([-1], [0]))
        """ A function that converts RGB to gray-scale guidance in $[-1, 1]$."""

    def _init_identity_grid(self):
        grid = torch.tensor([1., 0, 0, 0, 0, 1., 0, 0, 0, 0, 1., 0,]).float()
        grid = grid.repeat([self.grid_guidance * self.grid_height * self.grid_width, 1])  # (L * H * W, 12)
        grid = grid.reshape(1, self.grid_guidance, self.grid_height, self.grid_width, -1) # (1, L, H, W, 12)
        grid = grid.permute(0, 2, 3, 1, 4)  # (1, 12, L, H, W)
        return grid

    def tv_loss(self):
        """Computes and returns total variation loss on the bilateral grids. 
        """
        # if self.use_mutual:
        #     return total_variation_loss(self.grids) + total_variation_loss(self.depth_grids)
        # else:
        return total_variation_loss(self.grids)

    def forward(self, rgb, depth, idx=None):
        """Bilateral grid slicing. Supports 2-D, 3-D, 4-D, and 5-D input.
        For the 2-D, 3-D, and 4-D cases, please refer to `slice`.
        For the 5-D cases, `idx` will be unused and the first dimension of `xy` should be
        equal to the number of bilateral grids. Then this function becomes PyTorch's
        [`F.grid_sample`](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html).

        Args:
            grid_xy (torch.Tensor): The x-y coordinates in the range of $[0,1]$.
            rgb (torch.Tensor): The RGB values in the range of $[0,1]$.
            idx (torch.Tensor): The bilateral grid indices.

        Returns:
            Sliced affine matrices of shape $(..., 3, 4)$.
        """

        grid = self.grids[idx]
        rgb = rgb.permute(1, 2, 0)
        gray_img = self.rgb2gray(rgb)
        affine_mat = bilateral_slice(grid, gray_img)
        affine_mat = affine_mat.reshape(-1, 3, 4)
        rgb = rgb.reshape(-1, 3)
        res_rgb = color_affine_transform(affine_mat, rgb)
        # if self.use_mutual:
        #     depth_grid = self.depth_grids[idx] 
        #     depth = depth.squeeze(0)
        #     depth = depth / torch.max(depth)
        #     depth_affine = bilateral_slice(depth_grid, depth)
        #     depth_affine = depth_affine.reshape(-1, 3, 4)
        #     res_rgb = color_affine_transform(depth_affine, rgb)
        # elif self.use_depth:
        #     depth = depth.squeeze(0)
        #     depth = depth / torch.max(depth)
        #     depth_affine = bilateral_slice(grid, depth)
        #     depth_affine = depth_affine.reshape(-1, 3, 4)
        #     res_rgb = color_affine_transform(depth_affine, rgb)
        return affine_mat, res_rgb
    