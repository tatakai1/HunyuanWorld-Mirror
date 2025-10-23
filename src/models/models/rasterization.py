from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy

from src.models.utils.frustum import calculate_unprojected_mask
from src.models.utils.geometry import depth_to_world_coords_points, closed_form_inverse_se3
from src.models.utils.camera_utils import vector_to_camera_matrices
from src.models.utils import sh_utils, act_gs


class Rasterizer:
    def __init__(self, rasterization_mode="classic", packed=True, abs_grad=True, with_eval3d=False,
                 camera_model="pinhole", sparse_grad=False, distributed=False, grad_strategy=DefaultStrategy):
        self.rasterization_mode = rasterization_mode
        self.packed = packed
        self.abs_grad = abs_grad
        self.camera_model = camera_model
        self.sparse_grad = sparse_grad
        self.grad_strategy = grad_strategy
        self.distributed = distributed
        self.with_eval3d = with_eval3d

    def rasterize_splats(
        self,
        means,
        quats,
        scales,
        opacities,
        colors,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        render_colors, render_alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.packed,
            absgrad=(
                self.abs_grad
                if isinstance(self.grad_strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.sparse_grad,
            rasterize_mode=self.rasterization_mode,
            distributed=self.distributed,
            camera_model=self.camera_model,
            with_eval3d=self.with_eval3d,
            render_mode="RGB+ED",
            **kwargs,
        )
        return render_colors[..., :3], render_colors[..., 3:], render_alphas

    def rasterize_batches(self, means, quats, scales, opacities, colors, viewmats, Ks, width, height, **kwargs):
        rendered_colors, rendered_depths, rendered_alphas = [], [], []
        batch_size = len(means)
        
        for i in range(batch_size):
            means_i = means[i]  # [N, 4]
            quats_i = quats[i]  # [N, 4]
            scales_i = scales[i]  # [N, 3]
            opacities_i = opacities[i]  # [N,]
            colors_i = colors[i]  # [N, 3]
            viewmats_i = viewmats[i]  # [V, 4, 4]
            Ks_i = Ks[i]  # [V, 3, 3]
            
            render_colors_i, render_depths_i, render_alphas_i = self.rasterize_splats(
                means_i, quats_i, scales_i, opacities_i, colors_i, viewmats_i, Ks_i, width, height, **kwargs
            )
            
            rendered_colors.append(render_colors_i)  # V H W 3
            rendered_depths.append(render_depths_i)  # V H W 1
            rendered_alphas.append(render_alphas_i)  # V H W 1
            
        rendered_colors = torch.stack(rendered_colors, dim=0)  # B V H W 3
        rendered_depths = torch.stack(rendered_depths, dim=0)  # B V H W 1
        rendered_alphas = torch.stack(rendered_alphas, dim=0)  # B V H W 1
        
        return rendered_colors, rendered_depths, rendered_alphas
    

class GaussianSplatRenderer(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256,       # Output channels of gs_feat_head
        sh_degree: int = 0,
        predict_offset: bool = False,
        predict_residual_sh: bool = True,
        enable_prune: bool = True,
        voxel_size: float = 0.002,    # Default voxel size for prune_gs
        using_gtcamera_splat: bool = False,
        render_novel_views: bool = False,
        enable_conf_filter: bool = False,  # Enable confidence filtering
        conf_threshold_percent: float = 30.0,  # Confidence threshold percentage
        max_gaussians: int = 5000000,  # Maximum number of Gaussians
        debug=False,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.sh_degree = sh_degree
        self.nums_sh = (sh_degree + 1) ** 2
        self.predict_offset = predict_offset
        self.predict_residual_sh = predict_residual_sh
        self.voxel_size = voxel_size
        self.enable_prune = enable_prune
        self.using_gtcamera_splat = using_gtcamera_splat
        self.render_novel_views = render_novel_views
        self.enable_conf_filter = enable_conf_filter
        self.conf_threshold_percent = conf_threshold_percent
        self.max_gaussians = max_gaussians
        self.debug = debug

        # Predict Gaussian parameters from GS features (quaternions/scales/opacities/SH/weights/optional offsets)
        if self.predict_offset:
            splits_and_inits = [
                (4, 1.0, 0.0),                # quats
                (3, 0.00003, -7.0),           # scales
                (1, 1.0, -2.0),               # opacities
                (3 * self.nums_sh, 1.0, 0.0), # residual_sh
                (1, 1.0, -2.0),               # weights
                (3, 0.001, 0.001),            # offsets
            ]
            gaussian_raw_channels = 4 + 3 + 1 + self.nums_sh * 3 + 1 + 3
        else:
            splits_and_inits = [
                (4, 1.0, 0.0),                # quats
                (3, 0.00003, -7.0),           # scales
                (1, 1.0, -2.0),               # opacities
                (3 * self.nums_sh, 1.0, 0.0), # residual_sh
                (1, 1.0, -2.0),               # weights
            ]
            gaussian_raw_channels = 4 + 3 + 1 + self.nums_sh * 3 + 1

        self.gs_head = nn.Sequential(
            nn.Conv2d(feature_dim // 2, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(feature_dim, gaussian_raw_channels, kernel_size=1),
        )
        # Initialize weights and biases of the final layer by segments
        final_conv_layer = self.gs_head[-1]
        start_channels = 0
        for out_channel, s, b in splits_and_inits:
            nn.init.xavier_uniform_(final_conv_layer.weight[start_channels:start_channels+out_channel], s)
            nn.init.constant_(final_conv_layer.bias[start_channels:start_channels+out_channel], b)
            start_channels += out_channel

        # Rasterizer
        self.rasterizer = Rasterizer()

    # ======== Main entry point: Complete GS rendering and fill results back to predictions ========
    def render(
        self,
        gs_feats: torch.Tensor,                    # [B, S(+V), 3, H, W]
        images: torch.Tensor,                      # [B, S+V, 3, H, W]
        predictions: Dict[str, torch.Tensor],      # From WorldMirror: pose/depth/pts3d etc
        views: Dict[str, torch.Tensor],
        context_predictions = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns predictions with the following fields filled:
        - rendered_colors / rendered_depths / (rendered_alphas during training)
        - gt_colors / gt_depths / valid_masks
        - splats / rendered_extrinsics / rendered_intrinsics
        """
        H, W = images.shape[-2:]
        S = views["context_nums"] if "context_nums" in views else images.shape[1]
        V = images.shape[1] - S 

        # 1) Predict GS features from tokens, then convert to Gaussian parameters
        gs_feats_reshape = rearrange(gs_feats, "b s c h w -> (b s) c h w")
        gs_params = self.gs_head(gs_feats_reshape)
        
        # 2) Select predicted cameras
        Bx = images.shape[0]
        pred_all_extrinsic, pred_all_intrinsic = self.prepare_prediction_cameras(predictions, S + V, hw=(H, W))
        pred_all_extrinsic = pred_all_extrinsic.reshape(Bx, S + V, 4, 4)
        pred_all_source_extrinsic = pred_all_extrinsic[:, :S]

        scale_factor = 1.0
        if context_predictions is not None:
            pred_source_extrinsic, _ = self.prepare_prediction_cameras(context_predictions, S, hw=(H, W))
            pred_source_extrinsic = pred_source_extrinsic.reshape(Bx, S, 4, 4)
            scale_factor = pred_source_extrinsic[:, :, :3, 3].mean(dim=(1, 2), keepdim=True) / (
                pred_all_source_extrinsic[:, :, :3, 3].mean(dim=(1, 2), keepdim=True) + 1e-6
            )

        pred_all_extrinsic[..., :3, 3] = pred_all_extrinsic[..., :3, 3] * scale_factor

        render_viewmats, render_Ks = pred_all_extrinsic, pred_all_intrinsic
        render_images = images
        gt_colors = render_images.permute(0, 1, 3, 4, 2)
        
        # 3) Generate splats from gs_params + predictions, and perform voxel merging
        splats = self.prepare_splats(views, predictions, images, gs_params, S, V, position_from="gsdepth+predcamera", context_predictions=context_predictions, debug=False)
        splats_raw = {k: v.clone() for k, v in splats.items()}

        # Apply confidence filtering before pruning
        if self.enable_conf_filter and "depth_conf" in predictions:
            splats = self.apply_confidence_filter(splats, predictions["depth_conf"])
        
        if self.enable_prune:
            splats = self.prune_gs(splats, voxel_size=self.voxel_size)
        
        # 4) Rasterization rendering (training: chunked rendering + novel view valid mask correction; evaluation: view-by-view)

        # Prevent OOM by using chunked rendering
        rendered_colors_list, rendered_depths_list, rendered_alphas_list = [], [], []
        chunk_size = 4
        for i in range(0, gt_colors.shape[1], chunk_size):
            end_idx = min(i + chunk_size, gt_colors.shape[1])
            viewmats_i = render_viewmats[:, i:end_idx]
            Ks_i = render_Ks[:, i:end_idx]

            rendered_colors, rendered_depths, rendered_alphas = self.rasterizer.rasterize_batches(
                splats["means"], splats["quats"], splats["scales"], splats["opacities"],
                splats["sh"] if "sh" in splats else splats["colors"],
                viewmats_i.detach(), Ks_i.detach(),
                width=render_images.shape[-1], height=render_images.shape[-2],
                sh_degree=min(self.sh_degree, 0) if "sh" in splats else None,
            )
            rendered_colors_list.append(rendered_colors)
            rendered_depths_list.append(rendered_depths)
            rendered_alphas_list.append(rendered_alphas)

        rendered_colors = torch.cat(rendered_colors_list, dim=1)
        rendered_depths = torch.cat(rendered_depths_list, dim=1)
        rendered_alphas = torch.cat(rendered_alphas_list, dim=1)

        # 5) return predictions
        predictions["splats"] = splats

        return predictions

    def apply_confidence_filter(self, splats, gs_depth_conf):
        """
        Apply confidence filtering to Gaussian splats before pruning.
        Discard bottom p% confidence points, keep top (100-p)%.
        
        Args:
            splats: Dictionary containing Gaussian parameters
            gs_depth_conf: Confidence tensor [B, S, H, W]
        
        Returns:
            Filtered splats dictionary
        """
        if not self.enable_conf_filter or gs_depth_conf is None:
            return splats

        device = splats["means"].device
        B, N = splats["means"].shape[:2]

        # Flatten confidence: [B, S, H, W] -> [B, N]
        conf = gs_depth_conf.flatten(1).to(device)
        # Mask invalid/very small values
        conf = conf.masked_fill(conf <= 1e-5, float("-inf"))

        # Keep top (100-p)% points, discard bottom p%
        if self.conf_threshold_percent > 0:
            keep_from_percent = int(np.ceil(N * (100.0 - self.conf_threshold_percent) / 100.0))
        else:
            keep_from_percent = N
        K = max(1, min(self.max_gaussians, keep_from_percent))

        # Select top-K indices for each batch (deterministic, no randomness)
        topk_idx = torch.topk(conf, K, dim=1, largest=True, sorted=False).indices  # [B, K]
        
        filtered = {}
        mask_keys = ["means", "quats", "scales", "opacities", "sh", "weights"]
        
        for key in splats.keys():
            if key in mask_keys and key in splats:
                x = splats[key]
                if x.ndim == 2:  # [B, N]
                    filtered[key] = torch.gather(x, 1, topk_idx)
                else:
                    # Expand indices to match tensor dimensions
                    expand_idx = topk_idx.clone()
                    for i in range(x.ndim - 2):
                        expand_idx = expand_idx.unsqueeze(-1)
                    expand_idx = expand_idx.expand(-1, -1, *x.shape[2:])
                    filtered[key] = torch.gather(x, 1, expand_idx)
            else:
                filtered[key] = splats[key]

        return filtered

    def prune_gs(self, splats, voxel_size=0.002):
        """
        Prune Gaussian splats by merging those in the same voxel.
        
        Args:
            splats: Dictionary containing Gaussian parameters
            voxel_size: Size of voxels for spatial grouping
            
        Returns:
            Dictionary with pruned splats
        """
        B = splats["means"].shape[0]
        merged_splats_list = []
        device = splats["means"].device

        for i in range(B):
            # Extract splats for current batch
            splats_i = {k: splats[k][i] for k in ["means", "quats", "scales", "opacities", "sh", "weights"]}
            
            # Compute voxel indices
            coords = splats_i["means"]
            voxel_indices = (coords / voxel_size).floor().long()
            min_indices = voxel_indices.min(dim=0)[0]
            voxel_indices = voxel_indices - min_indices
            max_dims = voxel_indices.max(dim=0)[0] + 1
            
            # Flatten 3D voxel indices to 1D
            flat_indices = (voxel_indices[:, 0] * max_dims[1] * max_dims[2] + 
                           voxel_indices[:, 1] * max_dims[2] + 
                           voxel_indices[:, 2])
            
            # Find unique voxels and inverse mapping
            unique_voxels, inverse_indices = torch.unique(flat_indices, return_inverse=True)
            K = len(unique_voxels)

            # Initialize merged splats
            merged = {
                "means": torch.zeros((K, 3), device=device),
                "quats": torch.zeros((K, 4), device=device),
                "scales": torch.zeros((K, 3), device=device),
                "opacities": torch.zeros(K, device=device),
                "sh": torch.zeros((K, self.nums_sh, 3), device=device)
            }
            
            # Get weights and compute weight sums per voxel
            weights = splats_i["weights"]
            weight_sums = torch.zeros(K, device=device)
            weight_sums.scatter_add_(0, inverse_indices, weights)
            weight_sums = torch.clamp(weight_sums, min=1e-8)

            # Merge means (weighted average)
            for d in range(3):
                merged["means"][:, d].scatter_add_(0, inverse_indices, 
                                                 splats_i["means"][:, d] * weights)
            merged["means"] = merged["means"] / weight_sums.unsqueeze(1)

            # Merge spherical harmonics (weighted average)
            for d in range(3):
                merged["sh"][:, 0, d].scatter_add_(0, inverse_indices, 
                                                  splats_i["sh"][:, 0, d] * weights)
            merged["sh"] = merged["sh"] / weight_sums.unsqueeze(-1).unsqueeze(-1)

            # Merge opacities (weighted sum of squares)
            merged["opacities"].scatter_add_(0, inverse_indices, weights * weights)
            merged["opacities"] = merged["opacities"] / weight_sums

            # Merge scales (weighted average)
            for d in range(3):
                merged["scales"][:, d].scatter_add_(0, inverse_indices, 
                                                  splats_i["scales"][:, d] * weights)
            merged["scales"] = merged["scales"] / weight_sums.unsqueeze(1)

            # Merge quaternions (weighted average + normalization)
            for d in range(4):
                merged["quats"][:, d].scatter_add_(0, inverse_indices, 
                                                 splats_i["quats"][:, d] * weights)
            quat_norms = torch.norm(merged["quats"], dim=1, keepdim=True)
            merged["quats"] = merged["quats"] / torch.clamp(quat_norms, min=1e-8)

            merged_splats_list.append(merged)

        # Reorganize output
        output = {}
        for key in ["means", "sh", "opacities", "scales", "quats"]:
            output[key] = [merged[key] for merged in merged_splats_list]
        
        return output

    def prepare_splats(self, views, predictions, images, gs_params, context_nums, target_nums, context_predictions=None, position_from="gsdepth+predcamera", debug=False):
        """
        Prepare Gaussian splats from model predictions and input data.
        
        Args:
            views: Dictionary containing view data (camera poses, intrinsics, etc.)
            predictions: Model predictions including depth, pose_enc, etc.
            images: Input images [B, S_all, 3, H, W]
            gs_params: Gaussian splatting parameters from model
            context_nums: Number of context views (S)
            target_nums: Number of target views (V)
            context_predictions: Optional context predictions for camera poses
            position_from: Method to compute 3D positions ("pts3d", "preddepth+predcamera", "gsdepth+predcamera", "gsdepth+gtcamera")
            debug: Whether to use debug mode with ground truth data
            
        Returns:
            splats: Dictionary containing prepared Gaussian splat parameters
        """
        B, S_all, _, H, W = images.shape
        S, V = context_nums, target_nums
        splats = {}
        
        # Only take parameters from source view branch
        gs_params = rearrange(gs_params, "(b s) c h w -> b s h w c", b=B)[:, :S]
        splats["gs_feats"] = gs_params.reshape(B, S*H*W, -1)

        # Split Gaussian parameters based on whether offset prediction is enabled
        if self.predict_offset:
            quats, scales, opacities, residual_sh, weights, offsets = torch.split(
                gs_params, [4, 3, 1, self.nums_sh * 3, 1, 3], dim=-1
            )
            offsets = act_gs.reg_dense_offsets(offsets.reshape(B, S * H * W, 3))
            splats["offsets"] = offsets
        else:
            quats, scales, opacities, residual_sh, weights = torch.split(
                gs_params, [4, 3, 1, self.nums_sh * 3, 1], dim=-1
            )
            offsets = 0.

        # Apply activation functions to Gaussian parameters
        splats["quats"] = act_gs.reg_dense_rotation(quats.reshape(B, S * H * W, 4))
        splats["scales"] = act_gs.reg_dense_scales(scales.reshape(B, S * H * W, 3)).clamp_max(0.3)
        splats["opacities"] = act_gs.reg_dense_opacities(opacities.reshape(B, S * H * W))
        residual_sh = act_gs.reg_dense_sh(residual_sh.reshape(B, S * H * W, self.nums_sh * 3))

        # Handle spherical harmonics (SH) coefficients
        if self.predict_residual_sh:
            new_sh = torch.zeros_like(residual_sh)
            new_sh[..., 0, :] = sh_utils.RGB2SH(
                images[:, :S].permute(0, 1, 3, 4, 2).reshape(B, S * H * W, 3)
            )
            splats['sh'] = new_sh + residual_sh
            splats['residual_sh'] = residual_sh
        else:
            splats['sh'] = residual_sh

        splats["weights"] = act_gs.reg_dense_weights(weights.reshape(B, S * H * W))

        # Compute 3D positions based on specified method
        if position_from == "pts3d":
            pts3d = predictions["pts3d"][:, :S].reshape(B, S * H * W, 3)
            splats["means"] = pts3d + offsets
            
        elif position_from == "preddepth+predcamera":
            depth = predictions["depth"][:, :S].reshape(B * S, H, W)
            if context_predictions is not None:
                pose3x4, intrinsic = vector_to_camera_matrices(
                    context_predictions["camera_params"][:, :S].reshape(B * S, -1), (H, W)
                )
            else:
                pose3x4, intrinsic = vector_to_camera_matrices(
                    predictions["camera_params"][:, :S].reshape(B * S, -1), (H, W)
                )
            pose4x4 = torch.eye(4, device=pose3x4.device, dtype=pose3x4.dtype)[None].repeat(B * S, 1, 1)
            pose4x4[:, :3, :4] = pose3x4
            extrinsics = closed_form_inverse_se3(pose4x4)
            pts3d, _, _ = depth_to_world_coords_points(depth, extrinsics.detach(), intrinsic.detach())
            pts3d = pts3d.reshape(B, S * H * W, 3)
            splats["means"] = pts3d + offsets
            
        elif position_from == "gsdepth+predcamera":
            depth = predictions["gs_depth"][:, :S].reshape(B * S, H, W)
            if context_predictions is not None:
                pose3x4, intrinsic = vector_to_camera_matrices(
                    context_predictions["camera_params"][:, :S].reshape(B * S, -1), (H, W)
                )
            else:
                pose3x4, intrinsic = vector_to_camera_matrices(
                    predictions["camera_params"][:, :S].reshape(B * S, -1), (H, W)
                )
            pose4x4 = torch.eye(4, device=pose3x4.device, dtype=pose3x4.dtype)[None].repeat(B * S, 1, 1)
            pose4x4[:, :3, :4] = pose3x4
            extrinsics = closed_form_inverse_se3(pose4x4)
            pts3d, _, _ = depth_to_world_coords_points(depth, extrinsics.detach(), intrinsic.detach())
            pts3d = pts3d.reshape(B, S * H * W, 3)
            splats["means"] = pts3d + offsets
            
        elif position_from == "gsdepth+gtcamera":
            depth = predictions["gs_depth"][:, :S].reshape(B * S, H, W)
            pose4x4 = views["camera_pose"][:, :S].reshape(B * S, 4, 4)
            intrinsic = views["camera_intrinsics"][:, :S].reshape(B * S, 3, 3)
            extrinsics = pose4x4
            pts3d, _, _ = depth_to_world_coords_points(depth, extrinsics.detach(), intrinsic.detach())
            pts3d = pts3d.reshape(B, S * H * W, 3)
            splats["means"] = pts3d + offsets
            
        else:
            raise ValueError(f"Invalid position_from={position_from}")

        return splats

    def prepare_cameras(self, views, nums):
        viewmats = views['camera_pose'][:, :nums]
        Ks = views['camera_intrinsics'][:, :nums]
        return viewmats, Ks

    def prepare_prediction_cameras(self, predictions, nums, hw: Tuple[int, int]):
        """
        Prepare camera matrices from predicted pose encodings.
        
        Args:
            predictions: Dictionary containing pose_enc predictions
            nums: Number of views to process
            hw: Tuple of (height, width)
            
        Returns:
            viewmats: Camera view matrices [B, S, 4, 4]
            Ks: Camera intrinsic matrices [B, S, 3, 3]
        """
        B = predictions["camera_params"].shape[0]
        H, W = hw
        
        # Convert pose encoding to extrinsics and intrinsics
        pose3x4, intrinsic = vector_to_camera_matrices(
            predictions["camera_params"][:, :nums].reshape(B * nums, -1), (H, W)
        )
        
        # Convert to homogeneous coordinates and compute view matrices
        pose4x4 = torch.eye(4, device=pose3x4.device, dtype=pose3x4.dtype)[None].repeat(B * nums, 1, 1)
        pose4x4[:, :3, :4] = pose3x4

        viewmats = closed_form_inverse_se3(pose4x4).reshape(B, nums, 4, 4)
        Ks = intrinsic.reshape(B, nums, 3, 3)
        
        return viewmats, Ks
            
        
        
if __name__ == "__main__":
    device = "cuda:0"
    means = torch.randn((100, 3), device=device)
    quats = torch.randn((100, 4), device=device)
    scales = torch.rand((100, 3), device=device) * 0.1  
    opacities = torch.rand((100,), device=device)
    colors = torch.rand((100, 3), device=device)

    viewmats = torch.eye(4, device=device)[None, :, :].repeat(10, 1, 1)
    Ks = torch.tensor([
    [300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :].repeat(10, 1, 1)
    width, height = 300, 200

    rasterizer = Rasterizer()
    splats = {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
    }
    colors, alphas, _ = rasterizer.rasterize_splats(splats, viewmats, Ks, width, height)
    