"""
Model for Laplace-NeuralAngelo
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type

import torch
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import interlevel_loss
from nerfstudio.model_components.renderers import UncertaintyRenderer
from nerfstudio.models.neuralangelo import NeuralangeloModel, NeuralangeloModelConfig

from la_nerf.la_sdf_field import LaSDFField


@dataclass
class LaSDFModelConfig(NeuralangeloModelConfig):
    
    """Configuration for the SDFModel."""

    _target: Type = field(default_factory=lambda: LaSDFModel)

    # laplace backend
    laplace_backend: Literal["nnj", "backpack", "none"] = "nnj"

    # laplace method
    laplace_method: Literal["laplace", "linearized-laplace"] = "laplace"

    # number of samples for laplace
    laplace_num_samples: int = 100

    # hessian shape
    laplace_hessian_shape: Literal["diag", "kron", "full"] = "diag"

class LaSDFModel(NeuralangeloModel):
    """SDF Model"""
    config: LaSDFModelConfig

    def populate_modules(self):
        """Required to use L1 Loss."""
        super().populate_modules()

        # Fields
        self.field = LaSDFField(
            self.config.sdf_field,
            aabb = self.scene_box.aabb,
            num_images = self.num_train_data,
        )

        self.renderer_uq = UncertaintyRenderer()

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # curvature loss
        if self.training and self.config.curvature_loss_multi > 0.0:
            delta = self.field.numerical_gradients_delta
            centered_sdf = outputs['field_outputs'][FieldHeadNames.SDF]
            sourounding_sdf = outputs['field_outputs']["sampled_sdf"]
            
            sourounding_sdf = sourounding_sdf.reshape(centered_sdf.shape[:2] + (3, 2))
            
            # (a - b)/d - (b -c)/d = (a + c - 2b)/d
            # ((a - b)/d - (b -c)/d)/d = (a + c - 2b)/(d*d)
            curvature = (sourounding_sdf.sum(dim=-1) - 2 * centered_sdf) / (delta * delta)
            loss_dict["curvature_loss"] = torch.abs(curvature).mean() * self.config.curvature_loss_multi * self.curvature_loss_multi_factor
            
        return loss_dict
    
    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples = self.sampler(ray_bundle, sdf_fn=self.field.get_sdf)
        field_outputs = self.field(ray_samples, return_alphas=True)

        weights, transmittance = ray_samples.get_weights_and_transmittance_from_alphas(
            field_outputs[FieldHeadNames.ALPHA]
        )

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        if "rgb_mu" in field_outputs:
            rgb_mu = self.renderer_rgb(rgb=field_outputs["rgb_mu"], weights=weights)
            outputs["rgb_mu"] = rgb_mu

        if "rgb_sigma" in field_outputs:
            rgb_sigma = self.renderer_uq(betas=field_outputs["rgb_sigma"].sum(-1, keepdim=True), weights=weights)
            outputs["rgb_sigma"] = rgb_sigma

        if "density_sigma" in field_outputs:
            density_sigma = self.renderer_uq(betas=field_outputs["density_sigma"].sum(-1, keepdim=True), weights=weights)
            outputs["density_sigma"] = density_sigma

        return outputs