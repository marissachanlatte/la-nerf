"""
LERF configuration file.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, AdamWOptimizerConfig 
from nerfstudio.engine.schedulers import MultiStepWarmupSchedulerConfig
from nerfstudio.configs.base_config import Config, TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig

from la_nerf.la_sdf import LaSDFModelConfig
from la_nerf.la_sdf_field import LaSDFFieldConfig


la_sdf_method = MethodSpecification(
    config = Config(
        method_name="la_sdf",
        trainer=TrainerConfig(
            steps_per_eval_image=500,
            steps_per_eval_batch=500,
            steps_per_save=20000,
            steps_per_eval_all_images=1000000,  # set to a very large model so we don't eval with all images
            max_num_iterations=500_001,
            mixed_precision=False,
        ),
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=SDFStudioDataParserConfig(),
                train_num_rays_per_batch=32,
                eval_num_rays_per_batch=32,
                camera_optimizer=CameraOptimizerConfig(
                    mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=LaSDFModelConfig(
                sdf_field=LaSDFFieldConfig(
                    use_grid_feature=True,
                    num_layers=1,
                    num_layers_color=4,
                    hidden_dim=256,
                    hidden_dim_color=256,
                    geometric_init=True,
                    bias=0.5,
                    beta_init=0.3,
                    inside_outside=False,
                    use_appearance_embedding=False,
                    position_encoding_max_degree=6,
                    use_numerical_gradients=True,
                    base_res=64,
                    max_res=4096,
                    log2_hashmap_size=22,
                    hash_features_per_level=8,
                    hash_smoothstep=False,
                    use_position_encoding=False,
                ),
                background_model="mlp",
                enable_progressive_hash_encoding=True,
                enable_curvature_loss_schedule=True,
                enable_numerical_gradients_schedule=True,
            ),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamWOptimizerConfig(lr=1e-3, weight_decay=0.01, eps=1e-15),
                # "scheduler": NeuSSchedulerConfig(warm_up_end=5000, learning_rate_alpha=0.05, max_steps=500000),
                "scheduler": MultiStepWarmupSchedulerConfig(warm_up_end=5000, milestones=[300_000, 400_000], gamma=0.1),
            },
            "field_background": {
                "optimizer": AdamWOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": MultiStepWarmupSchedulerConfig(warm_up_end=5000, milestones=[300_000, 400_000], gamma=0.1),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="wandb",
    ),
    description="Base config for La-SDF.",
)