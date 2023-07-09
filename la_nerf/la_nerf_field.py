"""
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""


from typing import Dict, Literal, Optional, Tuple

import torch
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, shift_directions_for_tcnn
from torch import Tensor, nn

from torch.nn.utils import parameters_to_vector

import nnj
import pytorch_laplace


def get_mlp(in_dim, hidden_dim, out_dim, num_layers, activation, out_activation):

    layers = [nn.Linear(in_dim, hidden_dim)]
    layers.append(activation)
    for _ in range(num_layers):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(activation)
    layers.append(nn.Linear(hidden_dim, out_dim))

    if out_activation is not None:
        layers.append(out_activation)

    return nn.Sequential(*layers)


class LaNerfactoField(Field):
    """Compound Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        num_layers_transient: number of hidden layers for transient network
        hidden_dim_color: dimension of hidden layers for color network
        hidden_dim_transient: dimension of hidden layers for transient network
        appearance_embedding_dim: dimension of appearance embedding
        transient_embedding_dim: dimension of transient embedding
        use_transient_embedding: whether to use transient embedding
        use_average_appearance_embedding: whether to use average appearance embedding or zeros for inference
        spatial_distortion: spatial distortion to apply to the scene
        laplace_backend: which laplace backend to use
        laplace_method: which laplace method to use
        laplace_num_samples: number of samples to use for laplace
        laplace_hessian_shape: shape of the hessian to use for laplace
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        features_per_level: int = 2,
        hidden_dim_color: int = 64,
        appearance_embedding_dim: int = 32,
        use_transient_embedding: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        laplace_backend: Literal["pytorch-laplace", "laplace-redux","none"]="pytorch-laplace",
        laplace_method: Literal["laplace", "linearized-laplace"]="linearized-laplace",
        laplace_num_samples: int = 10,
        laplace_hessian_shape: Literal["diag", "kron", "full"]="diag",
    ) -> None:
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        self.embedding_appearance = Embedding(
            self.num_images, self.appearance_embedding_dim
        )
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_transient_embedding = use_transient_embedding
        self.base_res = base_res

        self.direction_encoding = SHEncoding(
            levels=4,
            implementation="tcnn",
        )

        self.position_encoding = NeRFEncoding(
            in_dim=3,
            num_frequencies=2,
            min_freq_exp=0,
            max_freq_exp=2 - 1,
            implementation="tcnn",
        )

        self.base_grid = HashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation="tcnn",
        )

        self.density_mlp = get_mlp(
            in_dim=self.base_grid.get_out_dim(),
            hidden_dim=hidden_dim,
            out_dim=1 + self.geo_feat_dim,
            num_layers=num_layers,
            activation=nn.ReLU(),
            out_activation=None,
        )

        self.rgb_mlp = get_mlp(
            in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim, 
            hidden_dim=hidden_dim_color, 
            out_dim=3,
            num_layers=num_layers_color, 
            activation=nn.ReLU(), 
            out_activation=nn.Sigmoid(),
        )

        self.laplace_backend = laplace_backend
        self.laplace_method = laplace_method
        self.laplace_num_samples = laplace_num_samples
        self.laplace_hessian_shape = laplace_hessian_shape

        if self.laplace_backend == "pytorch-laplace":

            # initialize hessian
            self.hessian = 10**6 * torch.ones_like(
                parameters_to_vector(self.rgb_mlp.parameters()), 
                device="cuda:0"
            )
            self.density_hessian = 10**6 * torch.ones_like(
                parameters_to_vector(self.density_mlp.parameters()),
                device="cuda:0"
            )

            # convert to nnj.Sequential
            self.rgb_mlp = nnj.utils.convert_to_nnj(self.rgb_mlp)
            self.density_mlp = nnj.utils.convert_to_nnj(self.density_mlp)

            # initalize hessian calculator and sampler
            self.hessian_calculator = pytorch_laplace.MSEHessianCalculator(
                hessian_shape=laplace_hessian_shape,
                approximation_accuracy="exact",
            )
            self.la_sampler = pytorch_laplace.DiagLaplace(
                backend="nnj",
            )

        elif self.laplace_backend == "laplace-redux":
            raise NotImplementedError

        # parameter to keep track of when to resample parameters.
        # if the we are not in training, but the last call was in training,
        # we need to resample parameters.
        self.resample_parameters = True
        self.resample_density_parameters = True

    def forward(self, ray_samples: RaySamples, compute_normals: bool = False) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        if compute_normals:
            with torch.enable_grad():
                density, density_embedding, density_mu, density_sigma = self.get_density(ray_samples)
                raise NotImplementedError
        else:
            density, density_embedding, density_mu, density_sigma = self.get_density(ray_samples)

        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

        if density_mu is not None:
            field_outputs["density_mu"] = density_mu  # type: ignore

        if density_sigma is not None:
            field_outputs["density_sigma"] = density_sigma  # type: ignore

        if compute_normals:
            with torch.enable_grad():
                normals = self.get_normals()
            field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs

            
    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(
                ray_samples.frustums.get_positions(), self.aabb
            )
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        grid_features = self.base_grid(positions_flat).float()
        h = self.density_mlp(grid_features)
        h_flat = h.view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(
            h_flat, [1, self.geo_feat_dim], dim=-1
        )
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]

        # laplace approximation on the densities
        density_mu, density_sigma = None, None
        if not self.training:

            if self.laplace_backend == "pytorch-laplace":
                if self.laplace_method == "laplace":
                    density_mu, density_sigma = None, None
                    
                    # resample weights from posterior
                    if self.resample_density_parameters:
                        sigma_q = self.la_sampler.posterior_scale(hessian=self.density_hessian)
                        mu_q = parameters_to_vector(self.density_mlp.parameters())
                        self.density_weight_samples = self.la_sampler.sample_from_normal(mu_q, sigma_q, self.laplace_num_samples)
                        self.resample_density_parameters = False

                    # compute mean and variance in output space 
                    # from sampled weights
                    h_mu, h_sigma = self.la_sampler.normal_from_samples(x=grid_features, samples=self.density_weight_samples, model=self.density_mlp)
                    h_mu_flat = h_mu.view(*ray_samples.frustums.shape, -1)
                    h_sigma_flat = h_sigma.view(*ray_samples.frustums.shape, -1)
                    density_before_activation_mu, _ = torch.split(h_mu_flat, [1, self.geo_feat_dim], dim=-1)
                    density_before_activation_sigma, _ = torch.split(h_sigma_flat, [1, self.geo_feat_dim], dim=-1)

                    density_mu = trunc_exp(density_before_activation_mu.to(positions))
                    density_sigma = trunc_exp(density_before_activation_sigma.to(positions))

                    density_mu = density_mu * selector[..., None]
                    density_sigma = density_sigma * selector[..., None]

                elif self.laplace_method == "linearized-laplace":
                    raise NotImplementedError

            elif self.laplace_backend == "laplace-redux":
                raise NotImplementedError
        
        else:
            # we are currently in training mode, 
            # next time we are not, then we need to resample parameters
            self.resample_density_parameters = True
            
            if self.laplace_backend == "pytorch-laplace":
                # update hessian estimate
                with torch.no_grad():
                    hessian_batch = self.hessian_calculator.compute_hessian(
                        x=grid_features,
                        # val=h,
                        model=self.density_mlp,
                    )
                    # momentum like update
                    self.density_hessian = 0.999 * self.density_hessian + hessian_batch

            elif self.laplace_backend == "laplace-redux":
                raise NotImplementedError

        return density, base_mlp_out, density_mu, density_sigma

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = shift_directions_for_tcnn(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # appearance
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
        else:
            if self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim),
                    device=directions.device,
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.appearance_embedding_dim),
                    device=directions.device,
                )

        h = torch.cat(
            [
                d,
                density_embedding.view(-1, self.geo_feat_dim),
                embedded_appearance.view(-1, self.appearance_embedding_dim),
            ],
            dim=-1,
        )

        rgb = self.rgb_mlp(h)
        outputs.update({FieldHeadNames.RGB: rgb.view(*outputs_shape, -1).to(directions)})

        # laplace approximation on the rgb values
        if not self.training:

            if self.laplace_backend == "pytorch-laplace":
                if self.laplace_method == "laplace":
                    
                    # resample weights from posterior
                    if self.resample_parameters:
                        sigma_q = self.la_sampler.posterior_scale(hessian=self.hessian)
                        mu_q = parameters_to_vector(self.rgb_mlp.parameters())
                        self.weight_samples = self.la_sampler.sample_from_normal(mu_q, sigma_q, self.laplace_num_samples)
                        self.resample_parameters = False

                    # compute mean and variance in output space 
                    # from sampled weights
                    rgb_mu, rgb_sigma = self.la_sampler.normal_from_samples(x=h, samples=self.weight_samples, model=self.rgb_mlp)

                elif self.laplace_method == "linearized-laplace":
                    rgb_mu, rgb_sigma = self.la_sampler.linearized_laplace(
                        x=h,
                        model=self.rgb_mlp,
                        hessian=self.hessian,
                    )

                outputs.update({"rgb_sigma": rgb_sigma.view(*outputs_shape, -1).to(directions)})
                outputs.update({"rgb_mu": rgb_mu.view(*outputs_shape, -1).to(directions)})

            elif self.laplace_backend == "laplace-redux":
                raise NotImplementedError
        
        else:
            # we are currently in training mode, 
            # next time we are not, then we need to resample parameters
            self.resample_parameters = True
            
            if self.laplace_backend == "pytorch-laplace":
                # update hessian estimate
                with torch.no_grad():
                    hessian_batch = self.hessian_calculator.compute_hessian(
                        x=h,
                        val=rgb,
                        model=self.rgb_mlp,
                    )
                    # momentum like update
                    self.hessian = 0.999 * self.hessian + hessian_batch

            elif self.laplace_backend == "laplace-redux":
                raise NotImplementedError

        return outputs
