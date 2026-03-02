"""Lightning callbacks for Stoic training."""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
import wandb
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from loguru import logger


class SetupWandB(Callback):
    """Attach W&B model watcher at the start of training."""

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        if trainer.is_global_zero:
            wandb.watch(pl_module, log_graph=False)


class StoichiometryModelClassWeights(Callback):
    """Compute and assign class weights to the loss function before training.

    Weights are derived from the training set's class distribution using the
    chosen weighting strategy. They are set once at the start of training and
    placed on the same device/dtype as the model parameters.

    Args:
        method: Weighting strategy. One of ``"effective_samples"``,
            ``"inverse"``, ``"inverse_sqrt"``, ``"inverse_log"``.
        beta: Smoothing parameter for the effective-samples method
            (Cui et al., 2019).
    """

    def __init__(
        self,
        method: str = "effective_samples",
        beta: float = 0.999,
    ):
        super().__init__()
        self.method = method
        self.beta = beta

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule, **kwargs) -> None:
        if (pl_module.node_class_weights is None) or (
            len(pl_module.node_class_weights)
            != len(pl_module.stoichiometry_classes_to_use)
        ):
            logger.info("Calculating stoichiometry class weights:")
            stoichiometry_class_weights = self._calculate_stoichiometry_weights(
                trainer, pl_module
            )
            pl_module.node_class_weights = torch.tensor(stoichiometry_class_weights).to(
                pl_module.dtype
            )
            logger.info(
                f"Stoichiometry classes: {pl_module.stoichiometry_classes_to_use}"
            )
            logger.info(f"Stoichiometry class weights: {pl_module.node_class_weights}")

    def _calculate_stoichiometry_weights(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> torch.Tensor:
        """Compute per-class weights and assign them to the loss module. Higher weights are assigned to rarer classes.

        Returns:
            Ordered weight tensor aligned with ``pl_module.stoichiometry_classes_to_use``.
        """
        stoichiometry_classes_counts = (
            trainer.train_dataloader.dataset.stoichiometry_classes_counts
        )

        weight_map = self._calculate_class_weights(
            stoichiometry_classes_counts,
            method=self.method,
            classes_to_use=pl_module.stoichiometry_classes_to_use,
            beta=self.beta,
        )
        stoichiometry_class_weights = torch.tensor(
            [weight_map[cl.item()] for cl in pl_module.stoichiometry_classes_to_use]
        )

        model_device = next(pl_module.model.parameters()).device
        model_dtype = next(pl_module.model.parameters()).dtype
        pl_module.loss.weight = stoichiometry_class_weights.to(model_device, model_dtype)

        return stoichiometry_class_weights

    @staticmethod
    def _calculate_class_weights(
        class_counts: Dict[str, np.ndarray],
        method: str = "effective_samples",
        smooth_factor: float = 0.0,
        beta: float = 0.999,
        classes_to_use: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> Dict[int, float]:
        """Compute normalised class weights from label counts. Higher weights are assigned to rarer classes.

        Args:
            class_counts: Dict with ``"stoichiometry_class"`` (array of class
                labels) and ``"count"`` (array of counts).
            method: Weighting strategy.
            smooth_factor: Additive smoothing applied to raw counts.
            beta: Smoothing parameter for effective-samples.
            classes_to_use: Subset of classes to keep. Accepts a list or tensor.

        Returns:
            Mapping from class label to its normalised weight.
        """
        classes = np.array(class_counts["stoichiometry_class"])
        counts = np.array(class_counts["count"]) + smooth_factor

        if classes_to_use is not None:
            mask = np.isin(classes, classes_to_use)
            classes = classes[mask]
            counts = counts[mask]
            if len(classes) == 0:
                raise ValueError("No classes left after filtering with classes_to_use")

        total_samples = counts.sum()
        frequencies = counts / total_samples

        if method == "inverse":
            weights = 1.0 / frequencies
        elif method == "inverse_sqrt":
            weights = 1.0 / np.sqrt(frequencies)
        elif method == "inverse_log":
            weights = 1.0 / np.log(1.0 + frequencies)
        elif method == "effective_samples":
            weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
        else:
            raise ValueError(f"Unknown weighting method: {method}")

        n_classes = len(classes)
        weights = weights * (n_classes / np.sum(weights))
        return {cl: float(weight) for cl, weight in zip(classes, weights)}


class ResamplingCallback(Callback):
    """Re-sample the training subset at the end of each epoch.

    Delegates to ``StoichiometryDataModule._resample_training_data`` which
    draws a new stratified sample based on stoichiometry classes from the full training pool, ensuring
    diversity across epochs. The new sample is then used for training in the next epoch.
    """

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        trainer.datamodule._resample_training_data()
