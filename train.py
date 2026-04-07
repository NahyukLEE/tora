"""Training script for TORA."""

import logging
import os
import warnings

import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from tora.utils.training import (
    setup_loggers,
    setup_wandb_resume,
    log_config_to_wandb,
    log_code_to_wandb,
)

logger = logging.getLogger("Train")
warnings.filterwarnings("ignore", module="lightning")

# Optimize for performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def setup_training(cfg: DictConfig):
    """Setup training components."""
    os.makedirs(cfg.log_dir, exist_ok=True)
    loggers = setup_loggers(cfg)
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    logger.info(f"Loading datasets: {datamodule.dataset_names}")

    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers)
    return model, datamodule, trainer, loggers


@hydra.main(version_base="1.3", config_path="./config", config_name="train")
def main(cfg: DictConfig):
    """Entry point for training the model."""

    ckpt_path = cfg.get("ckpt_path")
    is_fresh_run = not (ckpt_path and os.path.exists(ckpt_path))

    if is_fresh_run:
        seed = cfg.get("seed", 0)
        L.seed_everything(seed, workers=True, verbose=False)
        logger.info(f"Fresh run with random seed {seed}, checkpoint {ckpt_path}")
    else:
        logger.info(f"Resume training from checkpoint {ckpt_path}, no random seed set.")
        setup_wandb_resume(cfg)

    assert cfg.model.anchor_free == cfg.data.anchor_free, (
        f"model.anchor_free ({cfg.model.anchor_free}) != data.anchor_free ({cfg.data.anchor_free}). "
        "These must match — set both to True or both to False."
    )

    model, datamodule, trainer, loggers = setup_training(cfg)
    log_config_to_wandb(loggers, cfg)
    log_code_to_wandb(loggers)
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=ckpt_path if not is_fresh_run else None
    )


if __name__ == "__main__":
    main()
