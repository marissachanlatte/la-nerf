from __future__ import annotations

import math
import random
from typing import Tuple, Optional, Union

import fifteen
import flax
import jax
import jax_dataclasses as jdc
import tree_math as tm
import optax
from jax import numpy as jnp
from tqdm.auto import tqdm
from typing_extensions import Annotated, assert_never

from . import data, networks, render, tensor_vm, train_config, utils
from tensorf.training import TrainState

def random_split_like_tree(rng_key, target=None, treedef=None):
    # https://github.com/google/jax/discussions/9508
    if treedef is None:
        treedef = jax.tree_util.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


def tree_random_normal_like(rng_key, target, n_samples: Optional[int] = None):
    # https://github.com/google/jax/discussions/9508
    keys_tree = random_split_like_tree(rng_key, target)
    if n_samples is None:
        return jax.tree_util.tree_map(
            lambda l, k: jax.random.normal(k, l.shape, l.dtype),
            target,
            keys_tree,
        )
    else:
        return jax.tree_util.tree_map(
            lambda l, k: jax.random.normal(k, (n_samples,) + l.shape, l.dtype),
            target,
            keys_tree,
        )
    
@jdc.pytree_dataclass
class HessianState(jdc.EnforcedAnnotationsMixin):
    config: jdc.Static[train_config.TensorfConfig]

    # Representation/parameters.
    appearance_mlp: jdc.Static[networks.FeatureMlp]
    learnable_params: render.LearnableParams

    # Current axis-aligned bounding box.
    aabb: Annotated[jnp.ndarray, jnp.floating, (2, 3)]

    # Misc.
    prng_key: jax.random.KeyArray
    step: Annotated[jnp.ndarray, jnp.integer, ()]

    # hessian
    JtJe: render.LearnableParams

    # linearized laplace
    # linearized_laplace : Optional[bool] = False

    @staticmethod
    @jdc.jit
    def initialize(
        config: jdc.Static[train_config.TensorfConfig],
        prng_key: jax.random.KeyArray,
        num_cameras: jdc.Static[int],
        learnable_params: render.LearnableParams,
    ) -> HessianState:
        prng_keys = jax.random.split(prng_key, 5)

        def make_mlp() -> networks.FeatureMlp:
            appearance_mlp = networks.FeatureMlp(
                feature_n_freqs=config.feature_n_freqs,
                viewdir_n_freqs=config.viewdir_n_freqs,
                # If num_cameras is set, camera embeddings are enabled.
                num_cameras=num_cameras if config.camera_embeddings else None,
            )
            return appearance_mlp

        appearance_mlp = make_mlp()

        return HessianState(
            config=config,
            appearance_mlp=appearance_mlp,
            learnable_params=learnable_params,
            JtJe=jax.tree_map(lambda x: x * 0, learnable_params), # initalize with zeros
            aabb=jnp.array([config.initial_aabb_min, config.initial_aabb_max]),
            prng_key=prng_keys[4],
            step=jnp.array(0),
        )

    @jdc.jit(donate_argnums=0)
    def fit_hessian_step(
        self, minibatch: data.RenderedRays
    ) -> Tuple[HessianState, fifteen.experiments.TensorboardLogData]:
        """Single training step."""
        render_prng_key, new_prng_key = jax.random.split(self.prng_key)

        # If in mixed-precision mode, we render and backprop in float16.
        if self.config.mixed_precision:
            compute_dtype = jnp.float16
        else:
            compute_dtype = jnp.float32    

        # Compute gradients.
        # log_data: fifteen.experiments.TensorboardLogData

        learnable_params = jax.tree_map(
            # Cast parameters to desired precision.
            lambda x: x.astype(compute_dtype),
            self.learnable_params,
        )

        def render_rays(
            minibatch : data.RenderedRays,
            learnable_params: render.LearnableParams,
        ) -> Tuple[jnp.ndarray, fifteen.experiments.TensorboardLogData]:
            # Compute sample counts from grid dimensionality.
            # TODO: move heuristics into config?
            grid_dim = self.learnable_params.appearance_tensor.grid_dim()
            assert grid_dim == self.learnable_params.density_tensor.grid_dim()
            density_samples_per_ray = int(
                math.sqrt(3 * grid_dim**2) * self.config.train_ray_sample_multiplier
            )
            appearance_samples_per_ray = int(0.15 * density_samples_per_ray)

            # Render and compute loss.
            rendered = render.render_rays(
                appearance_mlp=self.appearance_mlp,
                learnable_params=learnable_params,
                aabb=self.aabb,
                rays_wrt_world=minibatch.rays_wrt_world,
                prng_key=render_prng_key,
                config=render.RenderConfig(
                    near=self.config.render_near,
                    far=self.config.render_far,
                    mode=render.RenderMode.RGB,
                    density_samples_per_ray=density_samples_per_ray,
                    appearance_samples_per_ray=appearance_samples_per_ray,
                ),
                dtype=compute_dtype,
            )

            return rendered

        # Estimate diagonal of GGN
        def estimate_diag_ggn(model_fn, minibatch, params, key: jax.random.PRNGKeyArray, S: int = 50):
            eps = tree_random_normal_like(key, params, n_samples=S)
            lmbd = lambda p: model_fn(minibatch, p)

            _, f_l = jax.linearize(lmbd, params)
            f_lt_tuple = jax.linear_transpose(f_l, params)
            def diag_est(eps): 
                return (tm.Vector(eps) * tm.Vector(f_lt_tuple(f_l(eps))[0])).tree
            JtJe = jax.vmap(diag_est)(eps)
            JtJe = jax.tree_map(lambda t: jnp.mean(t, axis=0), JtJe)
            # ggn_diag = jax.tree_map(lambda x: jnp.reshape(x, (-1,)), JtJe)
            # ggn_diag = jnp.concatenate(jax.tree_util.tree_flatten(ggn_diag)[0], axis=-1)
            return JtJe

        JtJe = estimate_diag_ggn(
            render_rays, 
            minibatch, 
            learnable_params, 
            render_prng_key
        )

        with jdc.copy_and_mutate(self, validate=True) as new_state:

            new_state.JtJe = jax.tree_map(lambda a,b: a+b, new_state.JtJe, JtJe)
            new_state.prng_key = new_prng_key
            new_state.step = new_state.step + 1
        return new_state #, log_data.prefix("fit/")


def fit_laplace(
    config: train_config.TensorfConfig,
) -> None:
    """Full training loop implementation."""

    # Set up our experiment: for checkpoints, logs, metadata, etc.
    experiment = fifteen.experiments.Experiment(data_dir=config.run_dir)
    experiment.assert_exists()
    config = experiment.read_metadata("config", train_config.TensorfConfig)

    # create hessian experiment for saving hessian data
    hessian_experiment = fifteen.experiments.Experiment(data_dir=config.run_dir / "hessian")

    # only works with mini batch size 1...
    minibatch_size = 1

    # Load dataset.
    dataset = data.make_dataset(
        config.dataset_type,
        config.dataset_path,
        config.scene_scale,
    )
    num_cameras = len(dataset.get_cameras())
    experiment.write_metadata("num_cameras", num_cameras)

    # Restore training state.
    train_state: TrainState
    train_state = TrainState.initialize(
        config,
        grid_dim=config.grid_dim_init,
        prng_key=jax.random.PRNGKey(94709),
        num_cameras=num_cameras,
    )
    train_state = experiment.restore_checkpoint(train_state)
    
    hessian_state: HessianState
    hessian_state = HessianState.initialize(
        config,
        learnable_params=train_state.learnable_params,
        num_cameras=num_cameras,
        prng_key=jax.random.PRNGKey(94709),
    )
    

    dataloader = fifteen.data.InMemoryDataLoader(
        dataset=dataset.get_training_rays(),
        minibatch_size=minibatch_size,
    )
    print("mini batch size:", minibatch_size)
    minibatches = fifteen.data.cycled_minibatches(dataloader, shuffle_seed=0)
    minibatches = iter(minibatches)

    # Run!
    print("Fitting laplace with config:", config)
    loop_metrics: fifteen.utils.LoopMetrics
    for loop_metrics in tqdm(
        fifteen.utils.range_with_metrics(config.n_iters),
        desc="Fit Laplace",
    ):
        # Load minibatch.
        minibatch = next(minibatches)
        assert minibatch.get_batch_axes() == (minibatch_size,)
        assert minibatch.colors.shape == (minibatch_size, 3)
        
        # Fit step.
        hessian_state = hessian_state.fit_hessian_step(minibatch)

        # Log & checkpoint.
        hessian_step = int(hessian_state.step)

        if hessian_step % 1000 == 0:
            hessian_experiment.save_checkpoint(
                hessian_state,
                step=hessian_step,
                keep_every_n_steps=2000,
            )