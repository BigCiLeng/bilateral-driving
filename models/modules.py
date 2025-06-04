import torch
from typing import Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pytorch3d.ops import knn_points
import nvdiffrast.torch as dr
from utils.geometry import rotation_6d_to_matrix
# from bilateral.slice import bilateral_slice
from bilateral.lib_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss, NeuralBilateralGrid, slice_feature

logger = logging.getLogger()

class XYZ_Encoder(nn.Module):
    encoder_type = "XYZ_Encoder"
    """Encode XYZ coordinates or directions to a vector."""

    def __init__(self, n_input_dims):
        super().__init__()
        self.n_input_dims = n_input_dims

    @property
    def n_output_dims(self) -> int:
        raise NotImplementedError

class SinusoidalEncoder(XYZ_Encoder):
    encoder_type = "SinusoidalEncoder"
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(
        self,
        n_input_dims: int = 3,
        min_deg: int = 0,
        max_deg: int = 10,
        enable_identity: bool = True,
    ):
        super().__init__(n_input_dims)
        self.n_input_dims = n_input_dims
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.enable_identity = enable_identity
        self.register_buffer(
            "scales", Tensor([2**i for i in range(min_deg, max_deg + 1)])
        )

    @property
    def n_output_dims(self) -> int:
        return (
            int(self.enable_identity) + (self.max_deg - self.min_deg + 1) * 2
        ) * self.n_input_dims

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [..., n_input_dims]
        Returns:
            encoded: [..., n_output_dims]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[..., None, :] * self.scales[:, None]),
            list(x.shape[:-1])
            + [(self.max_deg - self.min_deg + 1) * self.n_input_dims],
        )
        encoded = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))
        if self.enable_identity:
            encoded = torch.cat([x] + [encoded], dim=-1)
        return encoded

class MLP(nn.Module):
    """A simple MLP with skip connections."""

    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        num_layers: int = 3,
        hidden_dims: Optional[int] = 256,
        skip_connections: Optional[Tuple[int]] = [0],
    ) -> None:
        super().__init__()
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.n_output_dims = out_dims
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(in_dims, out_dims))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    layers.append(nn.Linear(in_dims, hidden_dims))
                elif i in skip_connections:
                    layers.append(nn.Linear(in_dims + hidden_dims, hidden_dims))
                else:
                    layers.append(nn.Linear(hidden_dims, hidden_dims))
            layers.append(nn.Linear(hidden_dims, out_dims))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        input = x
        for i, layer in enumerate(self.layers):
            if i in self.skip_connections:
                x = torch.cat([x, input], -1)
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nn.functional.relu(x)
        return x
    
class SkyModel(nn.Module):
    def __init__(
        self,
        class_name: str,
        n: int, 
        head_mlp_layer_width: int = 64,
        enable_appearance_embedding: bool = True,
        appearance_embedding_dim: int = 16,
        device: torch.device = torch.device("cuda")
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.device = device
        self.direction_encoding = SinusoidalEncoder(
            n_input_dims=3, min_deg=0, max_deg=6
        )
        self.direction_encoding.requires_grad_(False)
        
        self.enable_appearance_embedding = enable_appearance_embedding
        if self.enable_appearance_embedding:
            self.appearance_embedding_dim = appearance_embedding_dim
            self.appearance_embedding = nn.Embedding(n, appearance_embedding_dim, dtype=torch.float32)
            
        in_dims = self.direction_encoding.n_output_dims + appearance_embedding_dim \
            if self.enable_appearance_embedding else self.direction_encoding.n_output_dims
        self.sky_head = MLP(
            in_dims=in_dims,
            out_dims=3,
            num_layers=3,
            hidden_dims=head_mlp_layer_width,
            skip_connections=[1],
        )
        self.in_test_set = False
    
    def forward(self, image_infos):
        directions = image_infos["viewdirs"]
        self.device = directions.device
        prefix = directions.shape[:-1]
        
        dd = self.direction_encoding(directions.reshape(-1, 3)).to(self.device)
        if self.enable_appearance_embedding:
            # optionally add appearance embedding
            if "img_idx" in image_infos and not self.in_test_set:
                appearance_embedding = self.appearance_embedding(image_infos["img_idx"]).reshape(-1, self.appearance_embedding_dim)
            else:
                # use mean appearance embedding
                appearance_embedding = torch.ones(
                    (*dd.shape[:-1], self.appearance_embedding_dim),
                    device=dd.device,
                ) * self.appearance_embedding.weight.mean(dim=0)
            dd = torch.cat([dd, appearance_embedding], dim=-1)
        rgb_sky = self.sky_head(dd).to(self.device)
        rgb_sky = F.sigmoid(rgb_sky)
        return rgb_sky.reshape(prefix + (3,))
    
    def get_param_groups(self):
        return {
            self.class_prefix+"all": self.parameters(),
        }
        
class EnvLight(torch.nn.Module):

    def __init__(
        self,
        class_name: str,
        resolution=1024,
        device: torch.device = torch.device("cuda"),
        **kwargs
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.device = device
        self.to_opengl = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32, device="cuda")
        self.base = torch.nn.Parameter(
            0.5 * torch.ones(6, resolution, resolution, 3, requires_grad=True),
        )
        
    def forward(self, image_infos):
        l = image_infos["viewdirs"]
        
        l = (l.reshape(-1, 3) @ self.to_opengl.T).reshape(*l.shape)
        l = l.contiguous()
        prefix = l.shape[:-1]
        if len(prefix) != 3:  # reshape to [B, H, W, -1]
            l = l.reshape(1, 1, -1, l.shape[-1])

        light = dr.texture(self.base[None, ...], l, filter_mode='linear', boundary_mode='cube')
        light = light.view(*prefix, -1)
        # light = F.sigmoid(light)
        # light = torch.clamp(light, max=1.0)
        return light

    def get_param_groups(self):
        return {
            self.class_prefix+"all": self.parameters(),
        }
        
class AffineTransform(nn.Module):
    def __init__(
        self,
        class_name: str,
        n: int, 
        embedding_dim: int = 4,
        pixel_affine: bool = False,
        base_mlp_layer_width: int = 64,
        device: torch.device = torch.device("cuda")
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.device = device
        self.embedding_dim = embedding_dim
        self.pixel_affine = pixel_affine
        self.embedding = nn.Embedding(n, embedding_dim, dtype=torch.float32)
        
        input_dim = (embedding_dim + 2)if self.pixel_affine else embedding_dim
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, base_mlp_layer_width),
            nn.ReLU(),
            nn.Linear(base_mlp_layer_width, 12),
        )
        self.in_test_set = False
        
        self.zero_init()
        
    def zero_init(self):
        torch.nn.init.zeros_(self.embedding.weight)
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
    
    def forward(self, image_infos):
        if "img_idx" in image_infos and not self.in_test_set:
            embedding = self.embedding(image_infos["img_idx"])
        else:
            # use mean appearance embedding
            embedding = torch.ones(
                (*image_infos["viewdirs"].shape[:-1], self.embedding_dim),
                device=image_infos["viewdirs"].device,
            ) * self.embedding.weight.mean(dim=0)
        if self.pixel_affine:
            embedding = torch.cat([embedding, image_infos["pixel_coords"]], dim=-1)
        affine = self.decoder(embedding)
        affine = affine.reshape(*embedding.shape[:-1], 3, 4)
        
        affine[..., :3, :3] = affine[..., :3, :3] + torch.eye(3, device=affine.device).reshape(1, 3, 3)
        return affine

    def get_param_groups(self):
        return {
            self.class_prefix+"all": self.parameters(),
        }
class _ScaledTanh(nn.Module):
    def __init__(self, s=2.0):
        super().__init__()
        self.scaler = s
    def forward(self, x):
        return torch.tanh(self.scaler * x)

class BilateralAffineTransform(nn.Module):
    def __init__(self, class_name, n, grid_X, grid_Y, grid_W, device="cuda"):
        """
        Args:
            num (int): The number of bilateral grids (i.e., the number of views).
            grid_X (int): Defines grid width $W$.
            grid_Y (int): Defines grid height $H$.
            grid_W (int): Defines grid guidance dimension $L$.
        """
        super(BilateralAffineTransform, self).__init__()
        self.bil_grids = BilateralGrid(
                num=n,
                grid_X=grid_X,
                grid_Y=grid_Y,
                grid_W=grid_W,
            )
        """ A 5-D tensor of shape $(N, 12, L, H, W)$."""
        # Weights of BT601 RGB-to-gray.
        self.register_buffer('rgb2gray_weight', torch.Tensor([0.299, 0.587, 0.114]))
        # Define the lambda function to convert RGB to grayscale
        # if self.learn_gray:
        # self.gray_mlp_width=8,
        # self.gray_mlp_depth=2,
        # rgb2gray_mlp_linear = lambda l: nn.Linear(self.gray_mlp_width, self.gray_mlp_width if l < self.gray_mlp_depth - 1 else 1)
        # rgb2gray_mlp_actfn = lambda _: nn.ReLU(inplace=True)
        # self.rgb2gray = nn.Sequential(
        #     *([nn.Linear(3, self.gray_mlp_width)] + \
        #         [nn_module(l) for l in range(1, self.gray_mlp_depth) for nn_module in [rgb2gray_mlp_actfn, rgb2gray_mlp_linear]] + \
        #         [_ScaledTanh(2.)]))
        
        self.rgb2gray = lambda rgb: torch.tensordot(rgb, self.rgb2gray_weight, dims=([-1], [0]))
        """ A function that converts RGB to gray-scale guidance in $[-1, 1]$."""
        self.class_prefix = class_name + "#"
        self.device = device
        self.in_test_set = False
    
    def tv_loss(self):
        """Computes and returns total variation loss on the bilateral grids. 
        """
        return total_variation_loss(self.bil_grids.grids)

    
    def forward(self, rgb, image_infos):
        assert "img_idx" in image_infos
        cam_idx = int(image_infos["img_idx"][0][0])
        H, W, _ = rgb.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1.0, H, device=self.device),
            torch.linspace(0, 1.0, W, device=self.device),
            indexing="ij",
        )
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        if "img_idx" in image_infos and not self.in_test_set:
            out = slice(
                bil_grids=self.bil_grids,
                rgb=rgb.unsqueeze(0),
                xy=grid_xy,
                grid_idx=torch.tensor(cam_idx, device=self.device, dtype=torch.long),
            )
            return out["rgb_affine_mats"]
        else:
            near_idx_list = self.training_indices_for_test[cam_idx]
            out_tmp = torch.zeros([1, H, W, 3, 4], device=self.device)
            for near_idx in near_idx_list:
                out_tmp = out_tmp + slice(
                    bil_grids=self.bil_grids,
                    rgb=rgb.unsqueeze(0),
                    xy=grid_xy,
                    grid_idx=torch.tensor(near_idx, device=self.device, dtype=torch.long),
                )["rgb_affine_mats"]
            out_tmp = out_tmp / len(near_idx_list)
            return out_tmp

    def get_param_groups(self):
        return {
            self.class_prefix+"all": self.bil_grids.parameters(),
        }
def affine_to_homogeneous_batch(affine_matrices):
    # affine_matrices 形状为 (1, h, w, 3, 4)，需要扩展为 (1, h, w, 4, 4)
    batch_size, h, w, _, _ = affine_matrices.shape
    homogeneous_matrices = torch.zeros((batch_size, h, w, 4, 4), device=affine_matrices.device)
    homogeneous_matrices[:, :, :, :3, :] = affine_matrices
    homogeneous_matrices[:, :, :, 3, 3] = 1
    return homogeneous_matrices

def visual_grid(grid, image_name="test.png"):
    """
       grid: (1, H, W, 3, 4)
    """ 
    from sklearn.manifold import TSNE
    from PIL import Image
    import numpy as np
    _, H, W, _, _ = grid.shape
    flat_tensor = grid.view(-1, 12).cpu().numpy()
    n_samples = flat_tensor.shape[0]
    perplexity = min(30, n_samples // 3)
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42)
    reduced = tsne.fit_transform(flat_tensor)
    reduced_tensor = reduced.reshape(H, W, 3)
    reduced_tensor_normalized = (reduced_tensor - reduced_tensor.min()) / (reduced_tensor.max() - reduced_tensor.min())
    image_data = (reduced_tensor_normalized * 255).astype(np.uint8)
    image = Image.fromarray(image_data)
    image.save(image_name)
def transform_rgb(rgb, affine_trs):
    """
       rgb: (batch_size, h, w, 3)
       affine_matrices: (batch_size, h, w, 3, 4)
    """
    affine_trs = affine_trs.reshape(rgb.shape[0], rgb.shape[1], 3, 4)
    rgb_transformed = (affine_trs[..., :3, :3] @ rgb[..., None] + affine_trs[..., :3, 3:])[..., 0]
    return rgb_transformed

def visual_rgb(rgb, image_name="test.png"):
    import numpy as np
    import imageio
    rgb = (255 * np.clip(rgb.cpu().numpy(), 0, 1)).astype(np.uint8)
    imageio.imwrite(image_name, rgb)

def combine_matrix(H, W, matrix_list):
    device = matrix_list[-1].device

    mat = torch.eye(4, device=device).view(1, 1, 1, 4, 4).repeat(1, H, W, 1, 1)
    for arr in matrix_list:
        flatten = arr.flatten(start_dim=-2)
        arr_full = F.interpolate(
            flatten.permute(0, 3, 1, 2), # (1,c,h,w)
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1).view(1, H, W, 3, 4)
        hom_arr = affine_to_homogeneous_batch(arr_full)
        mat = hom_arr * mat

    return mat
def fill_matrix_res(H, W, matrix):
    B, H_m, W_m, _, _ = matrix.shape
    if H_m == H and W_m == W:
        return matrix
    flatten = matrix.flatten(start_dim=-2)
    arr_full = F.interpolate(
        flatten.permute(0, 3, 1, 2), # (1,c,h,w)
        size=(H, W),
        mode='bilinear',
        align_corners=False
    ).permute(0, 2, 3, 1).view(1, H, W, 3, 4)
    return arr_full

class MultiScaleBilateralAffineTransform(nn.Module):
    def __init__(self, class_name, n, grid, device="cuda"):
        """
        Args:
            num (int): The number of bilateral grids (i.e., the number of views).
            grid_X (int): Defines grid width $W$.
            grid_Y (int): Defines grid height $H$.
            grid_W (int): Defines grid guidance dimension $L$.
        """
        super(MultiScaleBilateralAffineTransform, self).__init__()
        self.grid_size = grid
        self.tv_weight = []
        mode = "bilinear"
        for i in range(len(self.grid_size)):
            if i == (len(self.grid_size) - 1):
                mode = "bilinear"
            setattr(self, f"bil_grids{i}", BilateralGrid(
                num=n,
                grid_X=grid[i][0],
                grid_Y=grid[i][1],
                grid_W=grid[i][2],
                mode=mode,
            ))
            self.tv_weight.append(0.5*(grid[i][0]*grid[i][1]*grid[i][2])**0.5)

        # self.concat_conv = nn.Conv2d(in_channels=36, out_channels=12, kernel_size=1)
        """ A 5-D tensor of shape $(N, 12, L, H, W)$."""
        # Weights of BT601 RGB-to-gray.
        self.register_buffer('rgb2gray_weight', torch.Tensor([0.299, 0.587, 0.114]))
        # Define the lambda function to convert RGB to grayscale
        self.rgb2gray = lambda rgb: torch.tensordot(rgb, self.rgb2gray_weight, dims=([-1], [0]))
        # self.gray_mlp_width=8,
        # self.gray_mlp_depth=2,
        # rgb2gray_mlp_linear = lambda l: nn.Linear(self.gray_mlp_width, self.gray_mlp_width if l < self.gray_mlp_depth - 1 else 1)
        # rgb2gray_mlp_actfn = lambda _: nn.ReLU(inplace=True)
        # self.rgb2gray = nn.Sequential(
        #     *([nn.Linear(3, self.gray_mlp_width)] + \
        #         [nn_module(l) for l in range(1, self.gray_mlp_depth) for nn_module in [rgb2gray_mlp_actfn, rgb2gray_mlp_linear]] + \
        #         [_ScaledTanh(2.)]))
        """ A function that converts RGB to gray-scale guidance in $[-1, 1]$."""
        self.class_prefix = class_name + "#"
        self.device = device
        self.in_test_set = False
    
    def tv_loss(self):
        """Computes and returns total variation loss on the bilateral grids. 
        """
        loss = 0
        for i in range(len(self.grid_size)):
            loss += total_variation_loss(getattr(self, f"bil_grids{i}").grids) * self.tv_weight[i]
        return loss
    
    def inverse_loss(self, gt, render):
        shape = self.save_matrix[0].shape
        B, H, W, _, _ = shape

        mat = torch.eye(4, device=self.device).view(1, 1, 1, 4, 4).repeat(1, H, W, 1, 1)
        for arr in self.save_matrix:
            hom_arr = affine_to_homogeneous_batch(arr)
            mat = hom_arr * mat
        
        flattened = mat.view(-1, 4, 4)
        inverses = torch.inverse(flattened).view(B, H, W, 4, 4)

        inverse_affine = inverses[:, :, :, :3, :]

        inverse_affine = inverse_affine.reshape(gt.shape[0], gt.shape[1], 3, 4)
        gt_transformed = (inverse_affine[..., :3, :3] @ gt[..., None] + inverse_affine[..., :3, 3:])[..., 0]

        loss = torch.abs(gt_transformed - render).mean()
        return loss

    def get_sample_grid(self, factor, H, W, rgb):
        H_down = H // factor
        W_down = W // factor
        rgb_lowres = F.interpolate(rgb.permute(2, 0, 1).unsqueeze(0), size=(H_down, W_down), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).squeeze(0)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1.0, H_down, device=self.device),
            torch.linspace(0, 1.0, W_down, device=self.device),
            indexing="ij",
        )
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        return grid_xy, rgb_lowres
    def forward(self, rgb, image_infos, guidance_factor=[4,4,2]):
        assert "img_idx" in image_infos
        cam_idx = int(image_infos["img_idx"][0][0])
        H, W, _ = rgb.shape
        if guidance_factor is not None:
            factor = guidance_factor
            out_list = []
            if "img_idx" in image_infos and not self.in_test_set:
                for i in range(len(self.grid_size)):
                    grid_xy, rgb_lowres = self.get_sample_grid(factor[i], H, W, rgb)
                    out_tmp = slice(
                        bil_grids=getattr(self, f"bil_grids{i}"),
                        rgb=rgb_lowres.unsqueeze(0),
                        xy=grid_xy,
                        grid_idx=torch.tensor(cam_idx, device=self.device, dtype=torch.long),
                    )["rgb_affine_mats"]
                    out_tmp = fill_matrix_res(H, W, out_tmp)
                    out_list.append(out_tmp)
            else:
                near_idx_list = self.training_indices_for_test[cam_idx]
                condition = 0
                for i in range(len(self.grid_size)):
                    grid_xy, rgb_lowres = self.get_sample_grid(factor[i], H, W, rgb)
                    H_down, W_down, _ = rgb_lowres.shape
                    if condition == 0:
                        out_tmp = torch.zeros([1, H_down, W_down, 3, 4], device=self.device)
                        for near_idx in near_idx_list:
                            out_tmp = out_tmp + slice(
                                bil_grids=getattr(self, f"bil_grids{i}"),
                                rgb=rgb_lowres.unsqueeze(0),
                                xy=grid_xy,
                                grid_idx=torch.tensor(near_idx, device=self.device, dtype=torch.long),
                            )["rgb_affine_mats"]
                        out_tmp = out_tmp / len(near_idx_list)
                    elif condition == 1:
                        out_tmp = slice(
                                bil_grids=getattr(self, f"bil_grids{i}"),
                                rgb=rgb_lowres.unsqueeze(0),
                                xy=grid_xy,
                                grid_idx=torch.tensor(near_idx_list[0], device=self.device, dtype=torch.long),
                            )["rgb_affine_mats"]
                    out_tmp = fill_matrix_res(H, W, out_tmp)
                    out_list.append(out_tmp)
        else: # no downsample guidance map
            out_list = []
            grid_xy, _ = self.get_sample_grid(1, H, W, rgb)
            if "img_idx" in image_infos and not self.in_test_set:
                for i in range(len(self.grid_size)):
                    out_tmp = slice(
                        bil_grids=getattr(self, f"bil_grids{i}"),
                        rgb=rgb.unsqueeze(0),
                        xy=grid_xy,
                        grid_idx=torch.tensor(cam_idx, device=self.device, dtype=torch.long),
                    )["rgb_affine_mats"]
                    out_list.append(out_tmp)
            else:
                near_idx_list = self.training_indices_for_test[cam_idx]
                condition = 0
                for i in range(len(self.grid_size)):
                    if condition == 0:
                        out_tmp = torch.zeros([1, H, W, 3, 4], device=self.device)
                        for near_idx in near_idx_list:
                            out_tmp = out_tmp + slice(
                                bil_grids=getattr(self, f"bil_grids{i}"),
                                rgb=rgb.unsqueeze(0),
                                xy=grid_xy,
                                grid_idx=torch.tensor(near_idx, device=self.device, dtype=torch.long),
                            )["rgb_affine_mats"]
                        out_tmp = out_tmp / len(near_idx_list)
                    elif condition == 1:
                        out_tmp = slice(
                                bil_grids=getattr(self, f"bil_grids{i}"),
                                rgb=rgb.unsqueeze(0),
                                xy=grid_xy,
                                grid_idx=torch.tensor(near_idx_list[0], device=self.device, dtype=torch.long),
                            )["rgb_affine_mats"]
                    out_list.append(out_tmp)

        self.save_matrix = out_list
        return out_list


    def get_param_groups(self):
        param_groups = {}
        for i in range(len(self.grid_size)):
            bil_grids=getattr(self, f"bil_grids{i}")
            group_name = f"{self.class_prefix}grid{i}"
            param_groups[group_name] = bil_grids.parameters()
        return param_groups

class NeuralBilateralAffineTransform(nn.Module):
    def __init__(self, class_name, n, grid_X, grid_Y, grid_W, feature_dim, hidden_dim, device="cuda"):
        """
        Args:
            num (int): The number of bilateral grids (i.e., the number of views).
            grid_X (int): Defines grid width $W$.
            grid_Y (int): Defines grid height $H$.
            grid_W (int): Defines grid guidance dimension $L$.
        """
        super(NeuralBilateralAffineTransform, self).__init__()
        self.bil_grids = NeuralBilateralGrid(
                num=n,
                grid_X=grid_X,
                grid_Y=grid_Y,
                grid_W=grid_W,
                feature_dim=feature_dim,
            )
        """ A 5-D tensor of shape $(N, 12, L, H, W)$."""
        # Weights of BT601 RGB-to-gray.
        self.register_buffer('rgb2gray_weight', torch.Tensor([0.299, 0.587, 0.114]))
        self.affine_network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, 12, bias=False),
        )
        self.feature_dim = feature_dim
        self.rgb2gray = lambda rgb: torch.tensordot(rgb, self.rgb2gray_weight, dims=([-1], [0]))
        """ A function that converts RGB to gray-scale guidance in $[-1, 1]$."""
        self.class_prefix = class_name + "#"
        self.device = device
        self.in_test_set = False
    
    def tv_loss(self):
        """Computes and returns total variation loss on the bilateral grids. 
        """
        return total_variation_loss(self.bil_grids.grids)

    def forward(self, rgb, image_infos):
        assert "img_idx" in image_infos
        cam_idx = int(image_infos["img_idx"][0][0])
        H, W, _ = rgb.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1.0, H, device=self.device),
            torch.linspace(0, 1.0, W, device=self.device),
            indexing="ij",
        )
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        if "img_idx" in image_infos and not self.in_test_set:
            out = slice_feature(
                bil_grids=self.bil_grids,
                rgb=rgb.unsqueeze(0),
                xy=grid_xy,
                grid_idx=torch.tensor(cam_idx, device=self.device, dtype=torch.long),
            )["affine_features"]
        else:
            near_idx_list = self.training_indices_for_test[cam_idx]
            out_tmp = torch.zeros([1, H, W, self.feature_dim], device=self.device)
            for near_idx in near_idx_list:
                out_tmp = out_tmp + slice_feature(
                    bil_grids=self.bil_grids,
                    rgb=rgb.unsqueeze(0),
                    xy=grid_xy,
                    grid_idx=torch.tensor(near_idx, device=self.device, dtype=torch.long),
                )["affine_features"]
            out = out_tmp / len(near_idx_list)
        out = self.affine_network(out)
        affine_matrix = out.reshape([1, H, W, 3, 4])
        return affine_matrix

    def get_param_groups(self):
        return {
            self.class_prefix+"all": self.parameters(),
        }
    
class MultiScaleNeuralBilateralAffineTransform(nn.Module):
    def __init__(self, class_name, n, grid, feature_dim, hidden_dim, device="cuda"):
        """
        Args:
            num (int): The number of bilateral grids (i.e., the number of views).
            grid_X (int): Defines grid width $W$.
            grid_Y (int): Defines grid height $H$.
            grid_W (int): Defines grid guidance dimension $L$.
        """
        super(MultiScaleNeuralBilateralAffineTransform, self).__init__()
        self.grid_size = grid
        self.tv_weight = []
        mode = "bilinear"
        for i in range(len(self.grid_size)):
            if i == (len(self.grid_size) - 1):
                mode = "bilinear"
            setattr(self, f"bil_grids{i}", NeuralBilateralGrid(
                num=n,
                grid_X=grid[i][0],
                grid_Y=grid[i][1],
                grid_W=grid[i][2],
                feature_dim=feature_dim,
                mode=mode,
            ))
            self.tv_weight.append(0.5*(grid[i][0]*grid[i][1]*grid[i][2])**0.5)
        """ A 5-D tensor of shape $(N, 12, L, H, W)$."""
        # Weights of BT601 RGB-to-gray.
        self.register_buffer('rgb2gray_weight', torch.Tensor([0.299, 0.587, 0.114]))
        input_dim = len(self.grid_size) * feature_dim
        self.feature_dim = feature_dim
        self.affine_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, 12, bias=False)
        )
        self.rgb2gray = lambda rgb: torch.tensordot(rgb, self.rgb2gray_weight, dims=([-1], [0]))
        """ A function that converts RGB to gray-scale guidance in $[-1, 1]$."""
        self.class_prefix = class_name + "#"
        self.device = device
        self.in_test_set = False
    
    def tv_loss(self):
        """Computes and returns total variation loss on the bilateral grids. 
        """
        loss = 0
        for i in range(len(self.grid_size)):
            loss += total_variation_loss(getattr(self, f"bil_grids{i}").grids) * self.tv_weight[i]
        return loss
    
    def get_sample_grid(self, factor, H, W, rgb):
        H_down = H // factor
        W_down = W // factor
        rgb_lowres = F.interpolate(rgb.permute(2, 0, 1).unsqueeze(0), size=(H_down, W_down), mode='bilinear', align_corners=False).permute(0, 2, 3, 1).squeeze(0)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1.0, H_down, device=self.device),
            torch.linspace(0, 1.0, W_down, device=self.device),
            indexing="ij",
        )
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        return grid_xy, rgb_lowres
    
    def forward(self, rgb, image_infos, guidance_factor=None):
        assert "img_idx" in image_infos
        cam_idx = int(image_infos["img_idx"][0][0])
        H, W, _ = rgb.shape
        if guidance_factor is not None:
            factor = guidance_factor
            out_list = []
            if "img_idx" in image_infos and not self.in_test_set:
                for i in range(len(self.grid_size)):
                    grid_xy, rgb_lowres = self.get_sample_grid(factor[i], H, W, rgb)
                    out_tmp = slice_feature(
                        bil_grids=getattr(self, f"bil_grids{i}"),
                        rgb=rgb_lowres.unsqueeze(0),
                        xy=grid_xy,
                        grid_idx=torch.tensor(cam_idx, device=self.device, dtype=torch.long),
                    )["affine_features"]
                    out_tmp = fill_matrix_res(H, W, out_tmp)
                    out_list.append(out_tmp)
            else:
                near_idx_list = self.training_indices_for_test[cam_idx]
                condition = 0
                for i in range(len(self.grid_size)):
                    grid_xy, rgb_lowres = self.get_sample_grid(factor[i], H, W, rgb)
                    H_down, W_down, _ = rgb_lowres.shape
                    if condition == 0:
                        out_tmp = torch.zeros([1, H_down, W_down, self.feature_dim], device=self.device)
                        for near_idx in near_idx_list:
                            out_tmp = out_tmp + slice_feature(
                                bil_grids=getattr(self, f"bil_grids{i}"),
                                rgb=rgb_lowres.unsqueeze(0),
                                xy=grid_xy,
                                grid_idx=torch.tensor(near_idx, device=self.device, dtype=torch.long),
                            )["affine_features"]
                        out_tmp = out_tmp / len(near_idx_list)
                    elif condition == 1:
                        out_tmp = slice_feature(
                                bil_grids=getattr(self, f"bil_grids{i}"),
                                rgb=rgb_lowres.unsqueeze(0),
                                xy=grid_xy,
                                grid_idx=torch.tensor(near_idx_list[0], device=self.device, dtype=torch.long),
                            )["affine_features"]
                    out_tmp = fill_matrix_res(H, W, out_tmp)
                    out_list.append(out_tmp)
        else: # no downsample guidance map
            out_list = []
            grid_xy, _ = self.get_sample_grid(1, H, W, rgb)
            if "img_idx" in image_infos and not self.in_test_set:
                for i in range(len(self.grid_size)):
                    out_tmp = slice_feature(
                        bil_grids=getattr(self, f"bil_grids{i}"),
                        rgb=rgb.unsqueeze(0),
                        xy=grid_xy,
                        grid_idx=torch.tensor(cam_idx, device=self.device, dtype=torch.long),
                    )["affine_features"]
                    out_list.append(out_tmp)
            else:
                near_idx_list = self.training_indices_for_test[cam_idx]
                condition = 0
                for i in range(len(self.grid_size)):
                    if condition == 0:
                        out_tmp = torch.zeros([1, H, W, self.feature_dim], device=self.device)
                        for near_idx in near_idx_list:
                            out_tmp = out_tmp + slice_feature(
                                bil_grids=getattr(self, f"bil_grids{i}"),
                                rgb=rgb.unsqueeze(0),
                                xy=grid_xy,
                                grid_idx=torch.tensor(near_idx, device=self.device, dtype=torch.long),
                            )["affine_features"]
                        out_tmp = out_tmp / len(near_idx_list)
                    elif condition == 1:
                        out_tmp = slice_feature(
                                bil_grids=getattr(self, f"bil_grids{i}"),
                                rgb=rgb.unsqueeze(0),
                                xy=grid_xy,
                                grid_idx=torch.tensor(near_idx_list[0], device=self.device, dtype=torch.long),
                            )["affine_features"]
                    out_list.append(out_tmp)

        self.save_matrix = out_list
        sliced_feature = torch.cat(out_list, dim=-1)
        sliced_feature = self.affine_network(sliced_feature)
        return sliced_feature.reshape([1, H, W, 3, 4])

    def get_param_groups(self):
        return {
            self.class_prefix+"all": self.parameters(),
        }
        
class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(
        self,
        class_name: str,
        n: int,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.device = device
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
        
        self.zero_init() # important for initialization !!

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_shape = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_shape, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)

    def get_param_groups(self):
        return {
            self.class_prefix+"all": self.parameters(),
        }

def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class DeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=59, x_multires=10, t_multires=10):
        super(DeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.x_multires = x_multires
        self.t_multires = t_multires
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(self.x_multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch

        self.linear = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                for i in range(D - 1)]
        )

        self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)
        self.gaussian_scaling = nn.Linear(W, 3)

    def forward(self, x, t):
        t_emb = self.embed_time_fn(t)
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, h], -1)

        d_xyz = self.gaussian_warp(h)
        scaling = self.gaussian_scaling(h)
        rotation = self.gaussian_rotation(h)

        return d_xyz, rotation, scaling
    
    
class ConditionalDeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, embed_dim=10,
                 x_multires=10, t_multires=10, 
                 deform_quat=True, deform_scale=True):
        super(ConditionalDeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.embed_dim = embed_dim
        self.deform_quat = deform_quat
        self.deform_scale = deform_scale
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(x_multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch + embed_dim

        self.linear = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                for i in range(D - 1)]
        )

        self.gaussian_warp = nn.Linear(W, 3)
        if self.deform_quat:
            self.gaussian_rotation = nn.Linear(W, 4)
        if self.deform_scale:
            self.gaussian_scaling = nn.Linear(W, 3)

    def forward(self, x, t, condition):
        t_emb = self.embed_time_fn(t)
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb, condition], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, condition, h], -1)

        d_xyz = self.gaussian_warp(h)
        scaling, rotation = None, None
        if self.deform_scale: 
            scaling = self.gaussian_scaling(h)
        if self.deform_quat:
            rotation = self.gaussian_rotation(h)

        return d_xyz, rotation, scaling

class VoxelDeformer(nn.Module):
    def __init__(
        self,
        vtx,
        vtx_features,
        resolution_dhw=[8, 32, 32],
        short_dim_dhw=0,  # 0 is d, corresponding to z
        long_dim_dhw=1,
        is_resume=False
    ) -> None:
        super().__init__()
        # vtx B,N,3, vtx_features: B,N,J
        # d-z h-y w-x; human is facing z; dog is facing x, z is upward, should compress on y
        B = vtx.shape[0]
        assert vtx.shape[0] == vtx_features.shape[0], "Batch size mismatch"

        # * Prepare Grid
        self.resolution_dhw = resolution_dhw
        device = vtx.device
        d, h, w = self.resolution_dhw

        self.register_buffer(
            "ratio",
            torch.Tensor(
                [self.resolution_dhw[long_dim_dhw] / self.resolution_dhw[short_dim_dhw]]
            ).squeeze(),
        )
        self.ratio_dim = -1 - short_dim_dhw
        x_range = (
            (torch.linspace(-1, 1, steps=w, device=device))
            .view(1, 1, 1, w)
            .expand(1, d, h, w)
        )
        y_range = (
            (torch.linspace(-1, 1, steps=h, device=device))
            .view(1, 1, h, 1)
            .expand(1, d, h, w)
        )
        z_range = (
            (torch.linspace(-1, 1, steps=d, device=device))
            .view(1, d, 1, 1)
            .expand(1, d, h, w)
        )
        grid = (
            torch.cat((x_range, y_range, z_range), dim=0)
            .reshape(1, 3, -1)
            .permute(0, 2, 1)
        )
        grid = grid.expand(B, -1, -1)

        gt_bbox_min = (vtx.min(dim=1).values).to(device)
        gt_bbox_max = (vtx.max(dim=1).values).to(device)
        offset = (gt_bbox_min + gt_bbox_max) * 0.5
        self.register_buffer(
            "global_scale", torch.Tensor([1.2]).squeeze()
        )  # from Fast-SNARF
        scale = (
            (gt_bbox_max - gt_bbox_min).max(dim=-1).values / 2 * self.global_scale
        ).unsqueeze(-1)

        corner = torch.ones_like(offset) * scale
        corner[:, self.ratio_dim] /= self.ratio
        min_vert = (offset - corner).reshape(-1, 1, 3)
        max_vert = (offset + corner).reshape(-1, 1, 3)
        self.bbox = torch.cat([min_vert, max_vert], dim=1)

        self.register_buffer("scale", scale.unsqueeze(1)) # [B, 1, 1]
        self.register_buffer("offset", offset.unsqueeze(1)) # [B, 1, 3]

        grid_denorm = self.denormalize(
            grid
        )  # grid_denorm is in the same scale as the canonical body

        if not is_resume:
            weights = (
                self._query_weights_smpl(
                    grid_denorm,
                    smpl_verts=vtx.detach().clone(),
                    smpl_weights=vtx_features.detach().clone(),
                )
                .detach()
                .clone()
            )
        else:
            # random initialization
            weights = torch.randn(
                B, vtx_features.shape[-1], *resolution_dhw
            ).to(device)

        self.register_buffer("lbs_voxel_base", weights.detach())
        self.register_buffer("grid_denorm", grid_denorm)

        self.num_bones = vtx_features.shape[-1]

        # # debug
        # import numpy as np
        # np.savetxt("./debug/dbg.xyz", grid_denorm[0].detach().cpu())
        # np.savetxt("./debug/vtx.xyz", vtx[0].detach().cpu())
        return

    def enable_voxel_correction(self):
        voxel_w_correction = torch.zeros_like(self.lbs_voxel_base)
        self.voxel_w_correction = nn.Parameter(voxel_w_correction)

    def enable_additional_correction(self, additional_channels, std=1e-4):
        additional_correction = (
            torch.ones(
                self.lbs_voxel_base.shape[0],
                additional_channels,
                *self.lbs_voxel_base.shape[2:]
            )
            * std
        )
        self.additional_correction = nn.Parameter(additional_correction)

    @property
    def get_voxel_weight(self):
        w = self.lbs_voxel_base
        if hasattr(self, "voxel_w_correction"):
            w = w + self.voxel_w_correction
        if hasattr(self, "additional_correction"):
            w = torch.cat([w, self.additional_correction], dim=1)
        return w

    def get_tv(self, name="dc"):
        if name == "dc":
            if not hasattr(self, "voxel_w_correction"):
                return torch.zeros(1).squeeze().to(self.lbs_voxel_base.device)
            d = self.voxel_w_correction
        elif name == "rest":
            if not hasattr(self, "additional_correction"):
                return torch.zeros(1).squeeze().to(self.lbs_voxel_base.device)
            d = self.additional_correction
        tv_x = torch.abs(d[:, :, 1:, :, :] - d[:, :, :-1, :, :]).mean()
        tv_y = torch.abs(d[:, :, :, 1:, :] - d[:, :, :, :-1, :]).mean()
        tv_z = torch.abs(d[:, :, :, :, 1:] - d[:, :, :, :, :-1]).mean()
        return (tv_x + tv_y + tv_z) / 3.0
        # tv_x = torch.abs(d[:, :, 1:, :, :] - d[:, :, :-1, :, :]).sum()
        # tv_y = torch.abs(d[:, :, :, 1:, :] - d[:, :, :, :-1, :]).sum()
        # tv_z = torch.abs(d[:, :, :, :, 1:] - d[:, :, :, :, :-1]).sum()
        # return tv_x + tv_y + tv_z

    def get_mag(self, name="dc"):
        if name == "dc":
            if not hasattr(self, "voxel_w_correction"):
                return torch.zeros(1).squeeze().to(self.lbs_voxel_base.device)
            d = self.voxel_w_correction
        elif name == "rest":
            if not hasattr(self, "additional_correction"):
                return torch.zeros(1).squeeze().to(self.lbs_voxel_base.device)
            d = self.additional_correction
        return torch.norm(d, dim=1).mean()

    def forward(self, xc, mode="bilinear"):
        shape = xc.shape  # ..., 3
        # xc = xc.reshape(1, -1, 3)
        w = F.grid_sample(
            self.get_voxel_weight,
            self.normalize(xc)[:, :, None, None],
            align_corners=True,
            mode=mode,
            padding_mode="border",
        )
        w = w.squeeze(3, 4).permute(0, 2, 1)
        w = w.reshape(*shape[:-1], -1)
        # * the w may have more channels
        return w

    def normalize(self, x):
        x_normalized = x.clone()
        x_normalized -= self.offset
        x_normalized /= self.scale
        x_normalized[..., self.ratio_dim] *= self.ratio
        return x_normalized

    def denormalize(self, x):
        x_denormalized = x.clone()
        x_denormalized[..., self.ratio_dim] /= self.ratio
        x_denormalized *= self.scale
        x_denormalized += self.offset
        return x_denormalized

    def _query_weights_smpl(self, x, smpl_verts, smpl_weights):
        # adapted from https://github.com/jby1993/SelfReconCode/blob/main/model/Deformer.py
        dist, idx, _ = knn_points(x, smpl_verts.detach(), K=30) # [B, N, 30]
        dist = dist.sqrt().clamp_(0.0001, 1.0)
        expanded_smpl_weights = smpl_weights.unsqueeze(2).expand(-1, -1, idx.shape[2], -1) # [B, N, 30, J]
        weights = expanded_smpl_weights.gather(1, idx.unsqueeze(-1).expand(-1, -1, -1, expanded_smpl_weights.shape[-1])) # [B, N, 30, J]

        ws = 1.0 / dist
        ws = ws / ws.sum(-1, keepdim=True)
        weights = (ws[..., None] * weights).sum(-2)

        b = x.shape[0]
        c = smpl_weights.shape[-1]
        d, h, w = self.resolution_dhw
        weights = weights.permute(0, 2, 1).reshape(b, c, d, h, w)
        for _ in range(30):
            mean = (
                weights[:, :, 2:, 1:-1, 1:-1]
                + weights[:, :, :-2, 1:-1, 1:-1]
                + weights[:, :, 1:-1, 2:, 1:-1]
                + weights[:, :, 1:-1, :-2, 1:-1]
                + weights[:, :, 1:-1, 1:-1, 2:]
                + weights[:, :, 1:-1, 1:-1, :-2]
            ) / 6.0
            weights[:, :, 1:-1, 1:-1, 1:-1] = (
                weights[:, :, 1:-1, 1:-1, 1:-1] - mean
            ) * 0.7 + mean
            sums = weights.sum(1, keepdim=True)
            weights = weights / sums
        return weights.detach()