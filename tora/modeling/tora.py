"""TORA: Rectified Flow for Point Cloud Assembly with representation alignment."""

import math
import time
import warnings
from functools import partial
from typing import Callable

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..eval.evaluator import Evaluator
from ..procrustes import fit_transformations
from ..sampler import get_sampler
from ..utils.checkpoint import get_rng_state, load_checkpoint_for_module, set_rng_state
from ..utils.logging import MetricsMeter, log_metrics_on_step, log_metrics_on_epoch


# ----------------------------------------------------------------------
# Flow matching utilities
# ----------------------------------------------------------------------

def compute_flow_target(x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the learning target of rectified flow.

    Args:
        x_0: Ground truth point cloud (B, N, 3)
        x_1: Noise point cloud (B, N, 3)
        t: Timesteps (B,)

    Returns:
        x_t: Linear interpolation (B, N, 3)
        v_t: Velocity field (B, N, 3)
    """
    t = t.view(-1, 1, 1)
    x_t = (1 - t) * x_0 + t * x_1
    v_t = x_1 - x_0
    return x_t, v_t


def compute_v_pred(out: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, kind: str = "v") -> torch.Tensor:
    """Compute predicted velocity field from model output.

    Following Back2basics, Li & He, 2025.

    Args:
        out: Model output
        x_t: Interpolated point cloud
        t: Timesteps (B,)
        kind: "v" for velocity, "x" for position, "eps" for noise prediction

    Returns:
        v_pred: Predicted velocity field
    """
    t = t.view(-1, 1, 1)
    if kind == "v":
        return out
    elif kind == "x":
        return (x_t - out) / torch.clamp(t, min=0.05)
    elif kind == "eps":
        return (out - x_t) / torch.clamp(1 - t, min=0.05)
    else:
        raise ValueError(f"Invalid kind: {kind}")


def sample_timesteps(
    batch_size: int,
    device: torch.device,
    strategy: str = "u_shaped",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    mode_scale: float = 2.0,
    a: float = 4.0,
    eps: float = 0.01,
) -> torch.Tensor:
    """Sample timesteps based on weighting scheme.

    Args:
        batch_size: Number of timesteps to sample.
        device: Device for the output tensor.
        strategy: One of "u_shaped", "logit_normal", "mode", "uniform".
        logit_mean: Mean for logit_normal strategy.
        logit_std: Standard deviation for logit_normal strategy.
        mode_scale: Scale for mode strategy.
        a: Concentration parameter for u_shaped strategy.
        eps: Minimum clamp value to reduce loss spikes.

    Returns:
        Tensor of shape (batch_size,) with sampled timesteps in [eps, 1.0].
    """
    if strategy == "u_shaped":
        u = torch.rand(batch_size, device=device) * 2 - 1
        u = torch.asinh(u * math.sinh(a)) / a
        u = (u + 1) / 2
    elif strategy == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device)
        u = torch.sigmoid(u)
    elif strategy == "mode":
        u = torch.rand(size=(batch_size,), device=device)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    elif strategy == "uniform":
        u = torch.rand(size=(batch_size,), device=device)
    else:
        raise ValueError(f"Invalid timestep sampling mode: {strategy}")
    return u.clamp(eps, 1.0)


class TORA(L.LightningModule):
    """Rectified Flow model for point cloud assembly with optional representation alignment.

    Args:
        feature_extractor: Encoder that produces per-point latent features.
        flow_model: DiT-style denoising network.
        teacher: Optional frozen teacher encoder for REPA alignment.
        projector: Optional projector mapping student intermediate features to teacher space.
        alignment_loss: Optional alignment loss module (e.g. CosineLoss, MSELoss).
        optimizer: Partial optimizer constructor.
        lr_scheduler: Partial LR scheduler constructor.
        encoder_ckpt: Path to a pretrained encoder checkpoint.
        flow_model_ckpt: Path to a pretrained flow model checkpoint.
        frozen_encoder: Whether to freeze the encoder during training.
        anchor_free: If True, do not enforce anchor part constraints.
        use_repa: Enable representation alignment with teacher/projector/alignment_loss.
        pred_type: Prediction type: "v" (velocity), "x" (position), "eps" (noise).
        repa_stop: Epoch at which to disable REPA alignment (None = never stop).
        extra_metrics: List of extra metric keys to compute during eval.
        loss_type: Flow loss type: "mse", "l1", or "huber".
        timestep_sampling: Timestep sampling strategy.
        inference_sampling_steps: Number of ODE integration steps at inference.
        inference_sampler: ODE sampler name ("euler", "rk2", "rk4").
        n_generations: Number of sample generations during testing (for best-of-N).
        pred_proc_fn: Optional post-processing function for forward output dict.
        save_results: Whether to save per-sample results during testing.
        gt_algo: Algorithm for ground-truth transform estimation ("icp" or "procrustes").
        use_spatial_norm: Whether to apply spatial normalization to teacher targets.
        spatial_norm_gamma: Gamma for spatial normalization.
    """

    def __init__(
        self,
        feature_extractor: L.LightningModule,
        flow_model: nn.Module,
        optimizer: "partial[torch.optim.Optimizer]",
        lr_scheduler: "partial[torch.optim.lr_scheduler._LRScheduler]" = None,
        teacher: nn.Module | None = None,
        projector: nn.Module | None = None,
        alignment_loss: nn.Module | None = None,
        encoder_ckpt: str = None,
        flow_model_ckpt: str = None,
        frozen_encoder: bool = False,
        anchor_free: bool = True,
        use_repa: bool = False,
        pred_type: str = "v",
        repa_stop: int | None = None,
        extra_metrics: list[str] = [],
        loss_type: str = "mse",
        timestep_sampling: str = "u_shaped",
        inference_sampling_steps: int = 20,
        inference_sampler: str = "euler",
        n_generations: int = 1,
        pred_proc_fn: Callable | None = None,
        save_results: bool = False,
        gt_algo: str = "icp",
        use_spatial_norm: bool = False,
        spatial_norm_gamma: float = 1.0,
        precompute_teacher: bool = False,
        teacher_cache_dir: str = "teacher_cache",
        free_teacher_after_cache: bool = True,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.flow_model = flow_model
        self.teacher = teacher if use_repa else None
        self.projector = projector if use_repa else None
        self.alignment_loss = alignment_loss if use_repa else None
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.frozen_encoder = frozen_encoder
        self.anchor_free = anchor_free
        self.use_repa = use_repa
        self.pred_type = pred_type
        self.repa_stop = repa_stop
        self.extra_metrics = extra_metrics
        self.loss_type = loss_type
        self.timestep_sampling = timestep_sampling
        self.inference_sampling_steps = inference_sampling_steps
        self.inference_sampler = inference_sampler
        self.n_generations = n_generations
        self.pred_proc_fn = pred_proc_fn
        self.save_results = save_results
        self.gt_algo = gt_algo
        self.use_spatial_norm = use_spatial_norm
        self.spatial_norm_gamma = spatial_norm_gamma
        self.precompute_teacher = precompute_teacher and use_repa
        self.teacher_cache_dir = teacher_cache_dir
        self.free_teacher_after_cache = free_teacher_after_cache

        # Precomputed teacher feature cache (disk-backed)
        self._teacher_cache = None
        self._teacher_freed = False
        if self.precompute_teacher:
            from ..teacher.cache import TeacherFeatureCache
            self._teacher_cache = TeacherFeatureCache(teacher_cache_dir)

        # Load checkpoints
        if encoder_ckpt is not None:
            load_checkpoint_for_module(
                self.feature_extractor,
                encoder_ckpt,
                keys_to_substitute={"feature_extractor.": ""},
                strict=False,
            )
        if flow_model_ckpt is not None:
            load_checkpoint_for_module(
                self.flow_model,
                flow_model_ckpt,
                prefix_to_remove="flow_model.",
                strict=False,
            )

        # Initialize evaluator and meter
        self.evaluator = Evaluator(self)
        self.meter = MetricsMeter(self)
        self._freeze_encoder()

    # ------------------------------------------------------------------
    # Encoder freezing
    # ------------------------------------------------------------------

    def _freeze_encoder(self, eval_mode: bool = False):
        if self.frozen_encoder or eval_mode:
            self.feature_extractor.eval()
            for module in self.feature_extractor.modules():
                module.eval()
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        else:
            self.feature_extractor.train()
            for module in self.feature_extractor.modules():
                module.train()
            for param in self.feature_extractor.parameters():
                param.requires_grad = True

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_fit_start(self):
        super().on_fit_start()
        if self.use_repa and self.teacher is not None and hasattr(self.teacher, "teacher_type"):
            if getattr(self.teacher, "teacher_type", None) == "sra":
                self.teacher.ema_update(self.flow_model, decay=0)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._freeze_encoder()

        # Free teacher GPU memory once the cache is fully populated
        if (
            self._teacher_cache is not None
            and not self._teacher_freed
            and self.free_teacher_after_cache
            and self.teacher is not None
            and len(self._teacher_cache) > 0
            and self.current_epoch >= 1
        ):
            print(f"[TORA] Freeing teacher encoder from GPU (cache has {len(self._teacher_cache)} entries)")
            if hasattr(self.teacher, "encoder") and self.teacher.encoder is not None:
                del self.teacher.encoder
                self.teacher.encoder = None
            self.teacher.cpu()
            torch.cuda.empty_cache()
            self._teacher_freed = True

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._freeze_encoder(eval_mode=True)

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self._freeze_encoder(eval_mode=True)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _encode(self, data_dict: dict):
        """Extract features from input data. Autocast disabled for SpConv compatibility."""
        with torch.inference_mode(self.frozen_encoder):
            with torch.autocast(device_type=self.device.type, enabled=False):
                out_dict = self.feature_extractor(data_dict)
        points = out_dict["point"]
        points["batch"] = points["batch_level1"].clone()
        return points

    # ------------------------------------------------------------------
    # Spatial normalization helper
    # ------------------------------------------------------------------

    @staticmethod
    def _spatial_normalize(x: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
        """Normalize features along the point dimension."""
        x = x - gamma * x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + 1e-6)
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, data_dict: dict, timesteps: torch.Tensor = None) -> dict:
        """Forward pass for training using rectified flow."""
        x_0 = data_dict["pointclouds_gt"]          # (B, N, 3)
        scales = data_dict["scales"]                # (B,)
        anchor_indices = data_dict["anchor_indices"]  # (B, N)
        B, N, _ = x_0.shape

        # Encode point clouds
        latent = self._encode(data_dict)

        # Sample timesteps
        if timesteps is not None:
            assert timesteps.shape[0] == B, "Timesteps batch size mismatch."
        else:
            timesteps = sample_timesteps(
                batch_size=B,
                device=self.device,
                strategy=self.timestep_sampling,
            )

        # Sample noise and compute flow target
        x_1 = torch.randn_like(x_0)                           # (B, N, 3)
        x_t, v_t = compute_flow_target(x_0, x_1, timesteps)   # (B, N, 3) each

        # Apply anchor part constraints (only in anchor-fixed mode)
        if not self.anchor_free:
            x_t[anchor_indices] = x_0[anchor_indices]
            v_t[anchor_indices] = 0.0

        # Predict velocity field
        out, interm_repr = self.flow_model(
            x=x_t,
            timesteps=timesteps,
            latent=latent,
            scales=scales,
            anchor_indices=anchor_indices,
        )
        v_pred = compute_v_pred(out, x_t, timesteps, kind=self.pred_type)

        # Representation alignment
        repr_pred, repr_t = None, None
        if self.use_repa and self.teacher is not None and self.projector is not None:
            # Extract teacher features (from cache or online)
            if self._teacher_cache is not None and self._teacher_cache.has_batch(data_dict["name"]):
                repr_t = self._teacher_cache.get_batch(data_dict["name"], device=self.device)
            else:
                repr_t = self.teacher.extract_features(data_dict)[0]
                if self._teacher_cache is not None:
                    self._teacher_cache.put_batch(data_dict["name"], repr_t)

            # Project student features
            repr_pred, subsample_idx = self.projector(interm_repr, data_dict)

            # Subsample teacher features to match projector output
            if subsample_idx is not None:
                gather_idx = subsample_idx.unsqueeze(-1).expand(-1, -1, repr_t.shape[-1])
                repr_t = torch.gather(repr_t, 1, gather_idx)

            # Optional spatial normalization
            if self.use_spatial_norm:
                repr_t = self._spatial_normalize(repr_t, gamma=self.spatial_norm_gamma)

        output_dict = {
            "t": timesteps,
            "v_pred": v_pred,
            "v_t": v_t,
            "x_0": x_0,
            "x_1": x_1,
            "x_t": x_t,
            "latent": latent,
            "repr_pred": repr_pred,
            "repr_t": repr_t,
        }

        if self.pred_proc_fn is not None:
            output_dict = self.pred_proc_fn(output_dict)
        return output_dict

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self, output_dict: dict) -> dict:
        """Compute rectified flow loss and optional alignment loss."""
        v_pred = output_dict["v_pred"]
        v_t = output_dict["v_t"]
        repr_pred = output_dict["repr_pred"]
        repr_t = output_dict["repr_t"]

        if self.loss_type == "mse":
            flow_loss = F.mse_loss(v_pred, v_t, reduction="mean")
        elif self.loss_type == "l1":
            flow_loss = F.l1_loss(v_pred, v_t, reduction="mean")
        elif self.loss_type == "huber":
            flow_loss = F.huber_loss(v_pred, v_t, reduction="mean")
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

        # Compute alignment loss
        if self.use_repa and self.alignment_loss is not None:
            if repr_pred is None or repr_t is None:
                warnings.warn(
                    f"REPA enabled but repr_pred or repr_t is None. "
                    f"repr_pred={repr_pred}, repr_t={repr_t}. "
                    f"Setting alignment_loss to 0."
                )
                align_loss = torch.tensor(0.0, device=flow_loss.device)
            else:
                align_loss = self.alignment_loss(repr_pred, repr_t, pointclouds_gt=output_dict.get("x_0"))
        else:
            align_loss = torch.tensor(0.0, device=flow_loss.device)

        loss = flow_loss + align_loss

        if not torch.isfinite(loss):
            torch.cuda.synchronize()
            print(f"\n[RANK {self.global_rank}] FATAL: NaN/Inf loss at step {self.global_step}")
            print(f"  flow_loss: {flow_loss.item()}, alignment_loss: {align_loss.item()}")
            print(f"  v_pred has NaN: {torch.isnan(v_pred).any()}, v_t has NaN: {torch.isnan(v_t).any()}")
            if repr_pred is not None:
                print(f"  repr_pred has NaN: {torch.isnan(repr_pred).any()}")
            if repr_t is not None:
                print(f"  repr_t has NaN: {torch.isnan(repr_t).any()}")
            raise ValueError("NaN loss encountered.")

        loss_dict = {
            "loss": loss,
            "norm_v_pred": v_pred.norm(dim=-1).mean().detach(),
            "norm_v_t": v_t.norm(dim=-1).mean().detach(),
            "flow_loss": flow_loss,
            "alignment_loss": align_loss,
        }

        if self.use_repa and repr_pred is not None and repr_t is not None:
            loss_dict["norm_repr_pred"] = repr_pred.norm(dim=-1).mean().detach()
            loss_dict["norm_repr_t"] = repr_t.norm(dim=-1).mean().detach()

        return loss_dict

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, data_dict: dict, batch_idx: int, dataloader_idx: int = 0):
        """Training step."""
        output_dict = self.forward(data_dict)
        loss_dict = self.loss(output_dict)
        log_metrics_on_step(self, loss_dict, prefix="train")
        return loss_dict["loss"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # SRA EMA update
        if self.use_repa and self.teacher is not None and hasattr(self.teacher, "teacher_type"):
            if getattr(self.teacher, "teacher_type", None) == "sra":
                self.teacher.ema_update(self.flow_model)
        # Optional REPA stopping
        if self.use_repa and self.repa_stop is not None and self.current_epoch >= self.repa_stop:
            self.use_repa = False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, data_dict: dict, batch_idx: int, dataloader_idx: int = 0):
        """Validation step."""
        output_dict = self.forward(data_dict)
        loss_dict = self.loss(output_dict)
        if "sampling_time" in self.extra_metrics:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        pointclouds_pred = self.sample_rectified_flow(data_dict, output_dict["latent"])
        if "sampling_time" in self.extra_metrics:
            torch.cuda.synchronize()
            self.log("sampling_time", time.perf_counter() - t0, prog_bar=False)

        # Fit per-part rotations and translations for evaluation
        rotations_pred, translations_pred = fit_transformations(
            data_dict["pointclouds"], pointclouds_pred, data_dict["points_per_part"]
        )

        # Evaluate
        eval_results = self.evaluator.run(data_dict, pointclouds_pred, rotations_pred, translations_pred)
        self.meter.add_metrics(dataset_names=data_dict["dataset_name"], **eval_results)
        return loss_dict["loss"]

    # ------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------

    def test_step(self, data_dict: dict, batch_idx: int, dataloader_idx: int = 0):
        """Test step with support for multiple generations (best-of-N)."""
        B = data_dict["points_per_part"].size(0)
        total = self.trainer.num_test_batches[dataloader_idx] if self.trainer.num_test_batches else "?"
        if self.global_rank == 0:
            print(f"[test] batch {batch_idx+1}/{total} (B={B})", flush=True)
        latent = self._encode(data_dict)
        n_trajectories = []
        n_rotations_pred = []
        n_translations_pred = []
        n_eval_results = []

        pointclouds_cond = data_dict["pointclouds"]
        points_per_part = data_dict["points_per_part"]

        for gen_idx in range(self.n_generations):
            trajs = self.sample_rectified_flow(data_dict, latent, return_trajectory=True)
            pointclouds_pred = trajs[-1]
            rotations_pred, translations_pred = fit_transformations(
                pointclouds_cond, pointclouds_pred, points_per_part
            )
            eval_results = self.evaluator.run(
                data_dict,
                pointclouds_pred,
                rotations_pred,
                translations_pred,
                save_results=self.save_results,
                generation_idx=gen_idx,
            )
            n_trajectories.append(trajs)
            n_rotations_pred.append(rotations_pred)
            n_translations_pred.append(translations_pred)
            n_eval_results.append(eval_results)

        # Accumulate average metrics via meter (properly reduced across GPUs)
        avg_metrics = {}
        for key in n_eval_results[0].keys():
            avg_metrics[f"avg/{key}"] = sum(result[key] for result in n_eval_results) / len(n_eval_results)
        self.meter.add_metrics(dataset_names=data_dict["dataset_name"], **avg_metrics)

        # Accumulate best-of-N (BoN) metrics
        if self.n_generations > 1:
            bon_metrics = {}
            for key in n_eval_results[0].keys():
                stacked = torch.stack([result[key] for result in n_eval_results])
                if "acc" in key or "recall" in key:
                    bon_metrics[f"best_of_n/{key}"] = stacked.max(dim=0).values
                else:
                    bon_metrics[f"best_of_n/{key}"] = stacked.min(dim=0).values
            self.meter.add_metrics(dataset_names=data_dict["dataset_name"], **bon_metrics)

        return {
            "trajectories": n_trajectories,
            "rotations_pred": n_rotations_pred,
            "translations_pred": n_translations_pred,
        }

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def sample_rectified_flow(
        self,
        data_dict: dict,
        latent: dict,
        x_1: torch.Tensor | None = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Sample from rectified flow using configurable integration methods.

        Args:
            data_dict: Input data dictionary.
            latent: Feature latent dictionary.
            x_1: Optional initial noise. If None, generates random Gaussian noise.
            return_trajectory: Whether to return the full trajectory.

        Returns:
            (B, N, 3) if return_trajectory is False, else (num_steps, B, N, 3).
        """
        anchor_indices = data_dict["anchor_indices"]
        scales = data_dict["scales"]

        def _flow_model_fn(x: torch.Tensor, t: float) -> torch.Tensor:
            B = x.shape[0]
            timesteps = torch.full((B,), t, device=x.device)
            out = self.flow_model(
                x=x,
                timesteps=timesteps,
                latent=latent,
                scales=scales,
                anchor_indices=anchor_indices,
            )[0]
            return compute_v_pred(out, x, timesteps, self.pred_type)

        x_0 = data_dict["pointclouds_gt"]
        x_1 = torch.randn_like(x_0) if x_1 is None else x_1

        result = get_sampler(self.inference_sampler)(
            flow_model_fn=_flow_model_fn,
            x_1=x_1,
            x_0=x_0,
            anchor_indices=anchor_indices if not self.anchor_free else None,
            num_steps=self.inference_sampling_steps,
            return_trajectory=return_trajectory,
        )
        return result

    # ------------------------------------------------------------------
    # Epoch end hooks
    # ------------------------------------------------------------------

    def on_validation_epoch_end(self):
        metrics = self.meter.compute_average()
        log_metrics_on_epoch(self, metrics, prefix="val")
        return metrics

    def on_test_epoch_end(self):
        metrics = self.meter.compute_average()
        log_metrics_on_epoch(self, metrics, prefix="test")
        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint):
        checkpoint["rng_state"] = get_rng_state()
        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        if "rng_state" in checkpoint:
            set_rng_state(checkpoint["rng_state"])
        else:
            print("No RNG state found in checkpoint.")
        super().on_load_checkpoint(checkpoint)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())

        if self.lr_scheduler is None:
            return {"optimizer": optimizer}

        lr_scheduler = self.lr_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }
