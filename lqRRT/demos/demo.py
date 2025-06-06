import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS
from .base_head import BaseTaskHead


@MODELS.register_module()
class Gaussian2DBEVHead(BaseTaskHead):
    """
    2D Gaussian head for BEV segmentation using 3D aggregator.
    Converts 2D Gaussians to 3D with fixed Z and uses existing 3D aggregator.
    """
    def __init__(
        self, 
        init_cfg=None,
        apply_loss_type=None,
        num_classes=3,  # e.g., drivable, lane, vehicle
        bev_h=200,
        bev_w=200,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        bev_z=0.0,  # Fixed Z coordinate for BEV plane
        z_scale=0.1,  # Small Z scale to make Gaussians flat
        empty_args=None,
        with_empty=False,
        cuda_kwargs=None,
        dataset_type='nusc',
        empty_label=0,
        use_localaggprob=True,
        use_localaggprob_fast=True,
        combine_geosem=False,
        gaussian_scale_factor=1.0,
        **kwargs,
    ):
        super().__init__(init_cfg)
        
        self.num_classes = num_classes
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.bev_z = bev_z
        self.z_scale = z_scale
        self.gaussian_scale_factor = gaussian_scale_factor
        
        # BEV grid setup - create 3D query points with fixed Z
        x_coords = torch.linspace(pc_range[0] + (pc_range[3] - pc_range[0])/(2*bev_w), 
                                pc_range[3] - (pc_range[3] - pc_range[0])/(2*bev_w), bev_w)
        y_coords = torch.linspace(pc_range[1] + (pc_range[4] - pc_range[1])/(2*bev_h), 
                                pc_range[4] - (pc_range[4] - pc_range[1])/(2*bev_h), bev_h)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Create 3D query points with fixed Z for BEV
        bev_query_points = torch.stack([
            grid_x.flatten(),
            grid_y.flatten(),
            torch.full_like(grid_x.flatten(), bev_z)  # Fixed Z coordinate
        ], dim=-1)  # (H*W, 3)
        
        self.register_buffer('bev_query_points', bev_query_points.unsqueeze(0))  # (1, H*W, 3)
        
        # Use existing 3D aggregator with adapted parameters
        self.use_localaggprob = use_localaggprob
        if use_localaggprob:
            if use_localaggprob_fast:
                import local_aggregate_prob_fast
                # Adapt cuda_kwargs for BEV usage
                bev_cuda_kwargs = cuda_kwargs.copy() if cuda_kwargs else {}
                bev_cuda_kwargs.update({
                    'H': bev_h,
                    'W': bev_w,
                    'D': 1,  # Single layer for BEV
                    'origin': [pc_range[0], pc_range[1], bev_z],
                    'voxel_size': [(pc_range[3] - pc_range[0])/bev_w, 
                                 (pc_range[4] - pc_range[1])/bev_h, 
                                 1.0]  # Z voxel size doesn't matter
                })
                self.aggregator = local_aggregate_prob_fast.LocalAggregator(**bev_cuda_kwargs)
            else:
                import local_aggregate_prob
                self.aggregator = local_aggregate_prob.LocalAggregator(**cuda_kwargs)
        else:
            import local_aggregate
            self.aggregator = local_aggregate.LocalAggregator(**cuda_kwargs)
        
        self.combine_geosem = combine_geosem
        
        # Empty class handling
        if with_empty:
            self.empty_scalar = nn.Parameter(torch.ones(1, dtype=torch.float) * 10.0)
            # Empty Gaussian at BEV center with large XY scale, small Z scale
            empty_mean = [0.0, 0.0, bev_z]
            empty_scale = empty_args.get('scale', [50.0, 50.0, z_scale]) if empty_args else [50.0, 50.0, z_scale]
            
            self.register_buffer('empty_mean', torch.tensor(empty_mean)[None, None, :])
            self.register_buffer('empty_scale', torch.tensor(empty_scale)[None, None, :])
            self.register_buffer('empty_rot', torch.tensor([1., 0., 0., 0.])[None, None, :])  # Identity quaternion
            self.register_buffer('empty_sem', torch.zeros(self.num_classes)[None, None, :])
            self.register_buffer('empty_opa', torch.ones(1)[None, None, :])
            
        self.with_empty = with_empty
        self.empty_args = empty_args
        self.dataset_type = dataset_type
        self.empty_label = empty_label

        # Loss application strategy
        if apply_loss_type == 'all':
            self.apply_loss_type = 'all'
        elif 'random' in apply_loss_type:
            self.apply_loss_type = 'random'
            self.random_apply_loss_layers = int(apply_loss_type.split('_')[1])
        elif 'fixed' in apply_loss_type:
            self.apply_loss_type = 'fixed'
            self.fixed_apply_loss_layers = [int(item) for item in apply_loss_type.split('_')[1:]]
            print(f"Supervised fixed layers: {self.fixed_apply_loss_layers}")
        else:
            raise NotImplementedError
            
        self.register_buffer('zero_tensor', torch.zeros(1, dtype=torch.float))

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def convert_2d_gaussians_to_3d(self, gaussians):
        """
        Convert 2D Gaussians to 3D format for use with 3D aggregator.
        
        Expected 2D gaussian format:
        - means: (B, G, 2) - XY coordinates  
        - scales: (B, G, 2) - XY scales
        - rotations: (B, G, 1) or (B, G) - rotation angles around Z
        - semantics: (B, G, C) - semantic logits
        - opacities: (B, G, 1) - opacity values
        
        Returns 3D format:
        - means: (B, G, 3) - XYZ coordinates (Z fixed)
        - scales: (B, G, 3) - XYZ scales (Z small)
        - rotations: (B, G, 4) - quaternions (only Z rotation)
        """
        means_2d = gaussians.means  # (B, G, 2)
        scales_2d = gaussians.scales * self.gaussian_scale_factor  # (B, G, 2)
        rotations_2d = gaussians.rotations  # (B, G, 1) or (B, G)
        semantics = gaussians.semantics  # (B, G, C)
        opacities = gaussians.opacities  # (B, G, 1)
        
        B, G = means_2d.shape[:2]
        
        # Convert to 3D means: add fixed Z coordinate
        means_3d = torch.cat([
            means_2d,
            torch.full((B, G, 1), self.bev_z, device=means_2d.device, dtype=means_2d.dtype)
        ], dim=-1)  # (B, G, 3)
        
        # Convert to 3D scales: add small Z scale
        scales_3d = torch.cat([
            scales_2d,
            torch.full((B, G, 1), self.z_scale, device=scales_2d.device, dtype=scales_2d.dtype)
        ], dim=-1)  # (B, G, 3)
        
        # Convert 2D rotations to 3D quaternions
        if rotations_2d.dim() == 3:
            rotations_2d = rotations_2d.squeeze(-1)  # (B, G)
        
        # Convert angle to quaternion (rotation around Z axis)
        half_angles = rotations_2d * 0.5
        cos_half = torch.cos(half_angles)
        sin_half = torch.sin(half_angles)
        
        # Quaternion: [w, x, y, z] for rotation around Z
        rotations_3d = torch.stack([
            cos_half,                                    # w
            torch.zeros_like(cos_half),                  # x  
            torch.zeros_like(cos_half),                  # y
            sin_half                                     # z
        ], dim=-1)  # (B, G, 4)
        
        # Create 3D gaussian object
        class Gaussian3D:
            def __init__(self, means, scales, rotations, semantics, opacities):
                self.means = means
                self.scales = scales  
                self.rotations = rotations
                self.semantics = semantics
                self.opacities = opacities
        
        return Gaussian3D(means_3d, scales_3d, rotations_3d, semantics, opacities)

    def prepare_gaussian_args(self, gaussians_3d):
        """
        Prepare 3D Gaussian arguments using the same logic as original GaussianHead.
        """
        means = gaussians_3d.means  # (B, G, 3)
        scales = gaussians_3d.scales  # (B, G, 3)
        rotations = gaussians_3d.rotations  # (B, G, 4)
        opacities = gaussians_3d.semantics  # (B, G, C)
        origi_opa = gaussians_3d.opacities  # (B, G, 1)
        
        if origi_opa.numel() == 0:
            origi_opa = torch.ones_like(opacities[..., :1], requires_grad=False)
            
        # Handle empty class (same logic as original)
        if self.with_empty:
            assert opacities.shape[-1] == self.num_classes - 1
            if self.empty_label == 0:
                opacities = torch.cat([torch.zeros_like(opacities[..., :1]), opacities], dim=-1)
            else:
                opacities = torch.cat([opacities, torch.zeros_like(opacities[..., :1])], dim=-1)
                
            means = torch.cat([means, self.empty_mean.expand(means.shape[0], -1, -1)], dim=1)
            scales = torch.cat([scales, self.empty_scale.expand(scales.shape[0], -1, -1)], dim=1)
            rotations = torch.cat([rotations, self.empty_rot.expand(rotations.shape[0], -1, -1)], dim=1)
            
            empty_sem = self.empty_sem.clone().expand(means.shape[0], -1, -1)
            empty_sem[..., self.empty_label] += self.empty_scalar
            opacities = torch.cat([opacities, empty_sem], dim=1)
            origi_opa = torch.cat([origi_opa, self.empty_opa.expand(origi_opa.shape[0], -1, -1)], dim=1)
            
        elif self.use_localaggprob:
            assert opacities.shape[-1] == self.num_classes - 1
            opacities = opacities.softmax(dim=-1)
            if self.empty_label == 0:
                opacities = torch.cat([torch.zeros_like(opacities[..., :1]), opacities], dim=-1)
            else:
                opacities = torch.cat([opacities, torch.zeros_like(opacities[..., :1])], dim=-1)

        # Compute 3D covariance matrices (same as original)
        bs, g, _ = means.shape
        S = torch.zeros(bs, g, 3, 3, dtype=means.dtype, device=means.device)
        S[..., 0, 0] = scales[..., 0]
        S[..., 1, 1] = scales[..., 1]
        S[..., 2, 2] = scales[..., 2]
        
        # Convert quaternion to rotation matrix (use your existing utility)
        from ..utils.utils import get_rotation_matrix
        R = get_rotation_matrix(rotations)  # (B, G, 3, 3)
        M = torch.matmul(S, R)
        Cov = torch.matmul(M.transpose(-1, -2), M)
        CovInv = Cov.cpu().inverse().cuda()  # (B, G, 3, 3)
        
        return means, origi_opa, opacities, scales, CovInv

    def forward(
        self,
        representation,
        metas=None,
        **kwargs
    ):
        num_decoder = len(representation)
        
        # Determine which layers to apply loss (same as original)
        if not self.training:
            apply_loss_layers = [num_decoder - 1]
        elif self.apply_loss_type == "all":
            apply_loss_layers = list(range(num_decoder))
        elif self.apply_loss_type == "random":
            if self.random_apply_loss_layers > 1:
                apply_loss_layers = np.random.choice(num_decoder - 1, self.random_apply_loss_layers - 1, False)
                apply_loss_layers = apply_loss_layers.tolist() + [num_decoder - 1]
            else:
                apply_loss_layers = [num_decoder - 1]
        elif self.apply_loss_type == 'fixed':
            apply_loss_layers = self.fixed_apply_loss_layers
        else:
            raise NotImplementedError

        # Process each decoder layer
        bev_predictions = []
        bin_logits = []
        density = []
        
        # BEV query points (1, H*W, 3) with fixed Z
        query_points = self.bev_query_points.repeat(representation[0]['gaussian'].means.shape[0], 1, 1)
        
        for idx in apply_loss_layers:
            gaussians_2d = representation[idx]['gaussian']
            
            # Convert 2D Gaussians to 3D format
            gaussians_3d = self.convert_2d_gaussians_to_3d(gaussians_2d)
            
            # Prepare arguments (same as original GaussianHead)
            means, origi_opa, opacities, scales, CovInv = self.prepare_gaussian_args(gaussians_3d)
            
            bs, g = means.shape[:2]
            
            # Use existing 3D aggregator
            if self.use_localaggprob:
                semantics, bin_log, dens = self.aggregator(
                    query_points.clone().float(), 
                    means, 
                    origi_opa.reshape(bs, g),
                    opacities,
                    scales,
                    CovInv
                )
                
                if self.combine_geosem:
                    sem = semantics[0][:, :-1] * semantics[1].unsqueeze(-1)
                    geo = 1 - semantics[1].unsqueeze(-1)
                    geosem = torch.cat([sem, geo], dim=-1)
                    bev_pred = geosem[None].transpose(1, 2)  # (B, H*W, C)
                else:
                    bev_pred = semantics[0][None].transpose(1, 2)  # (B, H*W, C)
                    
                bin_logits.append(semantics[1][None])
                density.append(semantics[2][None])
            else:
                semantics = self.aggregator(
                    query_points.clone().float(), 
                    means, 
                    origi_opa.reshape(bs, g),
                    opacities,
                    scales,
                    CovInv
                )
                bev_pred = semantics[None].transpose(1, 2)  # (B, H*W, C)
            
            # Reshape to BEV format (B, C, H, W)
            bev_pred_map = bev_pred.reshape(-1, self.bev_h, self.bev_w, self.num_classes)
            bev_pred_map = bev_pred_map.permute(0, 3, 1, 2)  # (B, C, H, W)
            
            bev_predictions.append(bev_pred_map)

        # Generate final prediction
        if self.use_localaggprob and not self.combine_geosem:
            threshold = kwargs.get("sigmoid_thresh", 0.5)
            final_semantics = bev_predictions[-1].argmax(dim=1)  # (B, H, W)
            final_occupancy = bin_logits[-1] > threshold
            final_occupancy = final_occupancy.reshape(-1, self.bev_h, self.bev_w)
            
            final_prediction = torch.ones_like(final_semantics) * self.empty_label
            final_prediction[final_occupancy] = final_semantics[final_occupancy]
        else:
            final_prediction = bev_predictions[-1].argmax(dim=1)  # (B, H, W)

        return {
            'bev_seg': bev_predictions,    # List of (B, C, H, W)
            'bin_logits': bin_logits,      # List of (B, H*W) if use_localaggprob
            'density': density,            # List of (B, H*W) if use_localaggprob  
            'final_bev': final_prediction, # (B, H, W)
            'gaussian': representation[-1]['gaussian'],
            'gaussians': [r['gaussian'] for r in representation],
            'bev_shape': (self.bev_h, self.bev_w),
            'pc_range': self.pc_range,
        }

    def loss(self, predictions, targets):
        """
        Compute BEV segmentation loss.
        """
        bev_predictions = predictions['bev_seg']
        
        losses = {}
        total_loss = 0
        
        # Apply loss to each prediction layer
        for i, pred in enumerate(bev_predictions):
            if len(targets.shape) == 3:  # (B, H, W)
                loss = F.cross_entropy(pred, targets.long(), ignore_index=-1)
            else:  # (B, C, H, W)
                loss = F.binary_cross_entropy_with_logits(pred, targets.float())
            
            losses[f'bev_seg_loss_{i}'] = loss
            total_loss += loss
            
        # Additional occupancy loss
        if self.use_localaggprob and 'bin_logits' in predictions:
            for i, bin_log in enumerate(predictions['bin_logits']):
                if bin_log is not None:
                    occupancy_target = (targets != self.empty_label).float()
                    occupancy_target = occupancy_target.reshape(-1, self.bev_h * self.bev_w)
                    
                    occ_loss = F.binary_cross_entropy_with_logits(bin_log, occupancy_target)
                    losses[f'occupancy_loss_{i}'] = occ_loss * 0.1
                    total_loss += losses[f'occupancy_loss_{i}']
        
        losses['total_bev_loss'] = total_loss
        return losses
