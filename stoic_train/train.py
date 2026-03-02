import warnings
import os

import torch
from lightning.pytorch.cli import LightningCLI
from loguru import logger

warnings.filterwarnings("ignore")


def main():
    """
    Run with: python -m stoic_train.train fit --config configs/config_sap.yaml
    Or in sbatch: srun python -m stoic_train.train fit --config configs/config_sap.yaml
    """
    logger.info("Starting training")
    torch.set_float32_matmul_precision("medium")
    LightningCLI(save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
