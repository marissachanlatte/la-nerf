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
        laplace_backend: Literal["pytorch-laplace", "laplace-redux"]="pytorch-laplace",
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

        self.mlp_base_grid = HashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation="tcnn",
        )
        self.mlp_base_mlp = MLP(
            in_dim=self.mlp_base_grid.get_out_dim(),
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + self.geo_feat_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation="tcnn",
        )
        self.mlp_base = torch.nn.Sequential(self.mlp_base_grid, self.mlp_base_mlp)

        self.mlp_head = get_mlp(
            self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim, 
            hidden_dim_color, 
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
                parameters_to_vector(self.mlp_head.parameters()), 
                device="cuda:0"
            )

            # convert to nnj.Sequential
            self.mlp_head = nnj.utils.convert_to_nnj(self.mlp_head)

            # initalize hessian calculator and sampler
            self.hessian_calculator = pytorch_laplace.MSEHessianCalculator(
                hessian_shape=laplace_hessian_shape,
                approximation_accuracy="exact",
            )
            self.la_sampler = pytorch_laplace.DiagLaplace(
                backend="nnj",
            )

        else:
            raise NotImplementedError
            
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
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(
            h, [1, self.geo_feat_dim], dim=-1
        )
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]
        return density, base_mlp_out

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


        if self.laplace_backend == "pytorch-laplace" and not self.train:

            if self.laplace_method == "laplace":
                rgb, rgb_sigma = self.la_sampler.laplace(
                    x=h,
                    model=self.mlp_head,
                    hessian=self.hessian,
                    n_samples=1 if self.training else self.laplace_num_samples,
                )
            elif self.laplace_method == "linearized-laplace":
                rgb, rgb_sigma = self.la_sampler.linearized_laplace(
                    x=h,
                    model=self.mlp_head,
                    hessian=self.hessian,
                )


            outputs.update({"rgb_sigma": rgb_sigma.view(*outputs_shape, -1).to(directions)})
            outputs.update({FieldHeadNames.RGB: rgb.view(*outputs_shape, -1).to(directions)})
        elif self.laplace_backend == "laplace-redux" and not self.train:
            raise NotImplementedError
        else:
            rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
            outputs.update({FieldHeadNames.RGB: rgb})

        if self.train:
            # update hessian estimate

            with torch.no_grad():
                hessian_batch = self.hessian_calculator.compute_hessian(
                    x=h,
                    # val=rgb, TODO: this is not needed for the hessian
                    model=self.mlp_head,
                )
                # momentum like update
                self.hessian = 0.999 * self.hessian + hessian_batch

        return outputs
