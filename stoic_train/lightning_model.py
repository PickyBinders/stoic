"""PyTorch Lightning module for stoichiometry prediction training."""

import gc
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from loguru import logger
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    AveragePrecision,
    ConfusionMatrix,
    Precision,
    Recall,
)

from stoic.model import Stoic
from stoic.utils import print_init_args
from stoic_train.losses import (
    ComplexLoss,
    ComplexProductLoss,
    ResidueWeightFocalLoss,
    SparsityLoss,
)
from stoic_train.metrics import (
    WeightDistributionMetric,
    WeightOverlapMetric,
    log_confusion_matrix_advanced,
)


class StoichiometryModelLightning(LightningModule):
    """Lightning module wrapping :class:`Stoic` for training and evaluation.

    Handles loss computation, metric tracking, optional auxiliary losses
    (sparsity, residue-weight supervision), and stoichiometry class filtering
    across train / val / test splits.

    Args:
        stoichiometry_classes_to_use: Integer stoichiometry labels to classify.
        seq_embed_model_name: ESM-2 identifier (short name or HuggingFace path).
        finetune_seq_embed_model: Allow gradient flow into the PLM.
        max_seq_len: Maximum padded sequence length (including special tokens).
        seq_feature_encoder: Short name of the sequence encoder layer (either a GCN or Identity).
        feature_pooling_strategy: Short name of the pooling strategy.
        predict_unknown_classes: Map unseen classes to a catch-all class ``0``.
        load_in_4bit: Quantise PLM weights to 4-bit via bitsandbytes.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout inside LoRA layers.
        seq_feature_encoder_dropout: Dropout in the sequence feature encoder.
        seq_embed_model_chunk_size: Chunk size for sequence embeddings generation.
        use_precomputed_embeddings: Bypass PLM and use pre-cached embeddings.
        use_contacting_res_weight: Weight pooling by known contacting residues.
        loss: Loss *class* to instantiate (e.g. ``CrossEntropyLoss``).
        use_focal: Pass ``use_focal=True`` when instantiating complex losses.
        fps_output_dim: Override output dim of the pooling layer. It will increase the model size significantly due to additional MLP projection.
        fps_return_weights: Return per-residue attention weights from pooling.
        fps_reduction_factor: Reduction factor inside pooling.
        use_sparsity_loss: Add a sparsity regulariser on residue weights.
        use_residue_weight_loss: Add supervised residue-weight loss.
        use_pretrained_res_pred_model: Load pre-trained pooling weights (frozen).
        pretrained_res_pred_model_path: Checkpoint path for pre-trained pooling.
        residue_weight_loss_type: Class name in ``stoic_train.losses``.
        residue_lambda: Scalar weight for the residue-weight loss term.
        sparsity_lambda: Scalar weight for the sparsity loss term.
        fps_use_soft_pooling: Enable soft (attention-based) pooling. If False, the threshold of 0.5 will be used to define the residues to pool across.
        fps_num_heads: Attention heads in the pooling layer if applicable.
        seq_feature_encoder_num_heads: Attention heads in the encoder if applicable.
    """

    UNKNOWN_CLASS: int = 0

    @print_init_args
    def __init__(
        self,
        stoichiometry_classes_to_use: List[int],
        seq_embed_model_name: str = "esm2_t33_650M_UR50D",
        finetune_seq_embed_model: bool = False,
        max_seq_len: int = 514,
        seq_feature_encoder: str = "Identity",
        feature_pooling_strategy: str = "AveragePooling",
        predict_unknown_classes: bool = False,
        load_in_4bit: bool = True,
        lora_r: int = 32,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        seq_feature_encoder_dropout: float = 0.2,
        seq_embed_model_chunk_size: int = 256,
        use_precomputed_embeddings: bool = False,
        use_contacting_res_weight: bool = False,
        loss: Type[nn.Module] = torch.nn.CrossEntropyLoss,
        use_focal: bool = False,
        fps_output_dim: Optional[int] = None,
        fps_return_weights: bool = False,
        fps_reduction_factor: int = 1,
        use_sparsity_loss: bool = False,
        use_residue_weight_loss: bool = False,
        use_pretrained_res_pred_model: bool = False,
        pretrained_res_pred_model_path: Optional[str] = None,
        residue_weight_loss_type: str = "ResidueWeightFocalLoss",
        residue_lambda: float = 10.0,
        sparsity_lambda: float = 10.0,
        fps_use_soft_pooling: bool = True,
        fps_num_heads: int = 4,
        fps_max_context_neighbors: Optional[int] = None,
        seq_feature_encoder_num_heads: int = 4,
    ):
        super().__init__()
        self.predict_unknown_classes = predict_unknown_classes

        if self.predict_unknown_classes:
            stoichiometry_classes_to_use.append(self.UNKNOWN_CLASS)
        self.stoichiometry_classes_to_use = torch.tensor(stoichiometry_classes_to_use)

        self.classes_to_use_mapper = {
            cl.item(): i for i, cl in enumerate(self.stoichiometry_classes_to_use)
        }
        self.classes_to_use_reverse_mapper = {
            i: cl.item() for i, cl in enumerate(self.stoichiometry_classes_to_use)
        }
        self.node_class_weights = None
        self.num_stoichiometry_classes = len(self.stoichiometry_classes_to_use)

        if "esm2" in seq_embed_model_name.lower() and "/" not in seq_embed_model_name:
            hf_model_name = f"facebook/{seq_embed_model_name}"
        else:
            hf_model_name = seq_embed_model_name
        self.seq_embed_model_name = seq_embed_model_name
        self.hf_model_name = hf_model_name

        self.seq_feature_encoder = self._resolve_class_name(seq_feature_encoder)
        self.feature_pooling_strategy = self._resolve_class_name(
            feature_pooling_strategy
        )

        self.finetune_seq_embed_model = finetune_seq_embed_model
        self.max_seq_len = max_seq_len
        self.seq_embed_model_chunk_size = seq_embed_model_chunk_size
        self.use_precomputed_embeddings = use_precomputed_embeddings
        self.use_contacting_res_weight = use_contacting_res_weight
        self.fps_use_soft_pooling = fps_use_soft_pooling
        self.fps_num_heads = fps_num_heads
        self.fps_max_context_neighbors = fps_max_context_neighbors
        self.seq_feature_encoder_num_heads = seq_feature_encoder_num_heads
        self.fps_return_weights = fps_return_weights
        self.fps_reduction_factor = fps_reduction_factor
        self.fps_output_dim = fps_output_dim
        self.use_sparsity_loss = use_sparsity_loss
        self.use_residue_weight_loss = use_residue_weight_loss
        self.use_pretrained_res_pred_model = use_pretrained_res_pred_model
        self.pretrained_res_pred_model_path = pretrained_res_pred_model_path
        self.residue_weight_loss_type = residue_weight_loss_type

        self.loss = loss
        self.use_focal = use_focal
        self.stoichiometry_lambda = 1.0
        if self.use_residue_weight_loss:
            self.residue_lambda = residue_lambda
        if self.use_sparsity_loss:
            self.sparsity_lambda = sparsity_lambda
        self._init_losses()
        self._init_metrics()

        stoic_kwargs = {
            "finetune_seq_embed_model": finetune_seq_embed_model,
            "load_in_4bit": load_in_4bit,
            "max_seq_len": max_seq_len,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "seq_feature_encoder_dropout": seq_feature_encoder_dropout,
            "seq_embed_model_chunk_size": seq_embed_model_chunk_size,
            "fps_use_soft_pooling": fps_use_soft_pooling,
            "fps_num_heads": fps_num_heads,
            "fps_return_weights": fps_return_weights,
            "fps_reduction_factor": fps_reduction_factor,
            "fps_max_context_neighbors": fps_max_context_neighbors,
            "seq_feature_encoder_num_heads": seq_feature_encoder_num_heads,
        }
        if fps_output_dim is not None:
            stoic_kwargs["fps_output_dim"] = fps_output_dim

        self.model = Stoic(
            stoichiometry_classes_to_use=stoichiometry_classes_to_use,
            seq_embed_model_name=self.hf_model_name,
            seq_feature_encoder=self.seq_feature_encoder,
            feature_pooling_strategy=self.feature_pooling_strategy,
            **stoic_kwargs,
        )
        if self.use_pretrained_res_pred_model:
            self._load_pretrained_pooling(pretrained_res_pred_model_path)

        self.save_hyperparameters()

    @staticmethod
    def _resolve_class_name(name: str) -> str:
        """``"stoic.layers.GCNConv"`` -> ``"GCNConv"``."""
        return name.rsplit(".", 1)[-1] if "." in name else name

    def _init_losses(self) -> None:
        """Instantiate the classification loss and optional auxiliary losses."""
        if self.loss.__name__ in ("ComplexLoss", "ComplexProductLoss"):
            self.loss = self.loss(use_focal=self.use_focal)
        else:
            self.loss = self.loss()

        if self.use_sparsity_loss:
            self.sparsity_loss = SparsityLoss()

        if self.use_residue_weight_loss:
            import stoic_train.losses as losses_mod

            loss_class = getattr(
                losses_mod, self.residue_weight_loss_type, ResidueWeightFocalLoss
            )
            self.residue_weight_loss = loss_class()

    def _init_metrics(self) -> None:
        """Create metric collections for train / val / test splits."""
        base_set1 = MetricCollection(
            [
                AveragePrecision(
                    task="multiclass",
                    num_classes=self.num_stoichiometry_classes,
                    average="none",
                ),
            ],
        )
        base_set2 = MetricCollection(
            [
                Precision(
                    task="multiclass",
                    num_classes=self.num_stoichiometry_classes,
                    average="none",
                ),
                Recall(
                    task="multiclass",
                    num_classes=self.num_stoichiometry_classes,
                    average="none",
                ),
                ConfusionMatrix(
                    task="multiclass",
                    num_classes=self.num_stoichiometry_classes,
                    normalize="true",
                ),
            ],
        )

        for split in ("train", "val", "test"):
            prefix = "" if split == "train" else f"{split}_"
            setattr(self, f"{prefix}node_metrics_set1", base_set1.clone(prefix=f"{split}_node_metrics_set1"))
            setattr(self, f"{prefix}node_metrics_set2", base_set2.clone(prefix=f"{split}_node_metrics_set2"))

        if self.fps_return_weights:
            base_wd = MetricCollection([WeightDistributionMetric()])
            base_wo = MetricCollection([WeightOverlapMetric()])
            for split in ("train", "val", "test"):
                prefix = "" if split == "train" else f"{split}_"
                setattr(self, f"{prefix}weight_distribution_metric", base_wd.clone(prefix=f"{split}_weight_distribution_metric_"))
                setattr(self, f"{prefix}weight_overlap_metric", base_wo.clone(prefix=f"{split}_weight_overlap_metric_"))

    def _load_pretrained_pooling(self, path: str) -> None:
        """
        Load and freeze pre-trained feature-pooling weights. Sometimes it's easier to train the stoichiometry 
        prediction model on top of a pretrained feature-pooling model, so weak classifier does not affect the residue prediction model.
        """
        logger.info(f"Loading pretrained res pred model from {path}")
        checkpoint = torch.load(path, map_location="cuda")
        pooling_state = {
            ".".join(k.split(".")[2:]): v
            for k, v in checkpoint["module"].items()
            if "feature_pooling" in k
        }
        self.model.feature_pooling_strategy.load_state_dict(pooling_state)
        for param in self.model.feature_pooling_strategy.parameters():
            param.requires_grad = False

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Make checkpoint loading robust to loss-module shape/key differences.

        Some historical checkpoints contain ``loss.weight`` (e.g. when using
        ``CrossEntropyLoss`` with dynamically assigned class weights). If the
        current loss implementation does not register this key, strict loading
        fails with ``Unexpected key(s) in state_dict: "loss.weight"``.
        """
        state_dict = checkpoint.get("state_dict", {})
        if (
            "loss.weight" in state_dict
            and "loss.weight" not in self.state_dict()
        ):
            state_dict.pop("loss.weight")
            logger.warning(
                "Dropped checkpoint key 'loss.weight' for compatibility with "
                "current loss implementation."
            )

    def forward(
        self,
        sequences: List[str],
        edge_index: torch.Tensor,
        contacting_res_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.model(sequences, edge_index, contacting_res_weight)

    def _get_split_metrics(
        self, split: str
    ) -> Tuple[
        MetricCollection,
        MetricCollection,
        Optional[MetricCollection],
        Optional[MetricCollection],
    ]:
        """
        Return ``(set1, set2, weight_dist | None, weight_overlap | None)`` for *split*.
        """
        prefix = "" if split == "train" else f"{split}_"
        set1 = getattr(self, f"{prefix}node_metrics_set1")
        set2 = getattr(self, f"{prefix}node_metrics_set2")
        wd = getattr(self, f"{prefix}weight_distribution_metric", None) if self.fps_return_weights else None
        wo = getattr(self, f"{prefix}weight_overlap_metric", None) if self.fps_return_weights else None
        return set1, set2, wd, wo

    def calculate_loss(
        self,
        node_scores: torch.Tensor,
        node_labels: torch.Tensor,
        complex_id: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the primary classification loss for the given node scores and labels, and complex id if applicable."""
        if isinstance(self.loss, (ComplexLoss, ComplexProductLoss)):
            return self.loss(node_scores, node_labels, complex_id)
        return self.loss(node_scores, node_labels)

    def _filter_classes(self, node_labels: torch.Tensor) -> torch.Tensor:
        """Boolean mask selecting nodes whose labels are in the target stoichiometry classes to use."""
        return torch.isin(node_labels.cpu(), self.stoichiometry_classes_to_use)

    def _process_batch(
        self,
        batch: Any,
        batch_idx: int,
        split: Literal["train", "val", "test"],
    ) -> Dict[str, Any]:
        """Run a single batch through the model and apply class filtering.

        Returns a dict with keys ``node_scores``, ``node_labels``,
        ``complex_id`` (optional), ``residue_weights`` (optional), ``target_residue_weights`` (optional),
        and ``attention_mask`` (optional).
        """
        sequences = np.concatenate(batch.sequence).tolist()
        node_labels = batch.quantity
        interacting_res = np.concatenate(batch.interacting_res).tolist()
        complex_id = torch.tensor(np.concatenate(batch.complex_id).tolist())
        edge_index = batch.edge_index

        contacting_res_weight = None
        if self.use_contacting_res_weight:
            contacting_res_weight = self._calculate_contacting_res_weight(
                sequences=sequences,
                interacting_res=interacting_res,
                seq_dim=self._get_seq_dim(sequences),
            )

        if self.use_precomputed_embeddings:
            output = self.model(
                batch.sequence_embedding.to(
                    next(self.model.node_classifier.parameters()).dtype
                ),
                edge_index,
                contacting_res_weight,
            )
            attention_mask = None
            residue_weights = None
        else:
            output = self.model(sequences, edge_index, contacting_res_weight)
            attention_mask = output["attention_mask"]
            residue_weights = output.get("residue_weights")

        node_scores = output["node_scores"]
        classes_mask = self._filter_classes(node_labels)
        if self.predict_unknown_classes:
            node_labels[~classes_mask] = self.UNKNOWN_CLASS
        else:
            node_scores = node_scores[classes_mask]
            node_labels = node_labels[classes_mask]
            complex_id = complex_id[classes_mask]
            if attention_mask is not None:
                attention_mask = attention_mask[classes_mask]
            if contacting_res_weight is not None:
                contacting_res_weight = contacting_res_weight[classes_mask]
            if residue_weights is not None:
                residue_weights = residue_weights[classes_mask]

        node_labels = torch.tensor(
            [self.classes_to_use_mapper[label.item()] for label in node_labels],
            device=node_scores.device,
            dtype=torch.long,
        )

        result: Dict[str, Any] = {
            "node_scores": node_scores,
            "node_labels": node_labels,
            "complex_id": complex_id,
            "residue_weights": residue_weights,
            "target_residue_weights": contacting_res_weight,
        }
        if attention_mask is not None:
            result["attention_mask"] = attention_mask
        return result

    def _compute_losses(self, step_output: Dict[str, Any]) -> Dict[str, Any]:
        """Compute main + auxiliary losses from a processed batch.
        
        Args:
            step_output: Output from :meth:`_process_batch`.
        Returns:
            Dictionary containing the loss, node scores, node labels, residue weights, target residue weights, and attention mask.
            "target_residue_weights": step_output["target_residue_weights"],
            "attention_mask": step_output["attention_mask"],
        }
        """
        loss = self.calculate_loss(
            step_output["node_scores"],
            step_output["node_labels"],
            step_output["complex_id"],
        )
        output: Dict[str, Any] = {
            "loss": loss,
            "node_scores": step_output["node_scores"],
            "node_labels": step_output["node_labels"],
            "residue_weights": step_output["residue_weights"],
            "target_residue_weights": step_output["target_residue_weights"],
        }
        if "attention_mask" in step_output:
            output["attention_mask"] = step_output["attention_mask"]

        if self.use_sparsity_loss:
            sparsity_loss, sparsity_dict = self.sparsity_loss(
                attention_weights=step_output["residue_weights"],
                mask=step_output.get("attention_mask"),
            )
            output["sparsity_loss"] = sparsity_loss
            output["sparsity_loss_dict"] = sparsity_dict

        if self.use_residue_weight_loss:
            output["residue_weight_loss"] = self.residue_weight_loss(
                residue_weights=step_output["residue_weights"],
                target_residue_weights=step_output["target_residue_weights"],
            )

        return output

    def _shared_step(
        self,
        batch: Any,
        batch_idx: int,
        split: Literal["train", "val", "test"],
    ) -> torch.Tensor:
        """Forward pass, loss aggregation, logging, and metric updates for *split*."""
        step_output = self._process_batch(batch, batch_idx, split)
        output = self._compute_losses(step_output)

        is_train = split == "train"
        log_kw: Dict[str, Any] = {"on_step": is_train, "on_epoch": True}

        node_loss = output["loss"]
        self.log(f"{split}_node_loss", node_loss.item(), **log_kw)
        total_loss = node_loss * self.stoichiometry_lambda

        if self.use_sparsity_loss:
            total_loss = total_loss + output["sparsity_loss"] * self.sparsity_lambda
            for key, val in output["sparsity_loss_dict"].items():
                self.log(f"{split}_{key}", val.item(), **log_kw)

        if self.use_residue_weight_loss:
            total_loss = total_loss + output["residue_weight_loss"] * self.residue_lambda
            self.log(
                f"{split}_residue_weight_loss",
                output["residue_weight_loss"].item(),
                **log_kw,
            )

        self.log(
            f"{split}_loss",
            total_loss.item(),
            prog_bar=True,
            sync_dist=(not is_train),
            **log_kw,
        )

        scores, labels = output["node_scores"], output["node_labels"]
        m1, m2, wd, wo = self._get_split_metrics(split)
        m1(scores, labels.int())
        m2(torch.argmax(scores, dim=1), labels.int())

        if wd is not None:
            wd(
                attention_weights=output["residue_weights"],
                mask=output.get("attention_mask"),
            )
        if wo is not None:
            wo(
                predicted_probs=output["residue_weights"],
                target_weights=output["target_residue_weights"],
            )

        return total_loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, batch_idx, "train")
        torch.cuda.empty_cache()
        gc.collect()
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, "test")

    def _log_epoch_metrics(self, split: str) -> None:
        """Compute, log, and reset all metrics for *split*."""
        m1, m2, wd, wo = self._get_split_metrics(split)
        sync = split != "train"

        metrics_dict = m1.compute()
        metrics_dict.update(m2.compute())
        if wd is not None:
            metrics_dict.update(wd.compute())
        if wo is not None:
            metrics_dict.update(wo.compute())

        class_names = list(self.classes_to_use_mapper.keys())
        for metric_name, value in metrics_dict.items():
            if "ConfusionMatrix" in metric_name:
                logger.info(
                    f"{split.capitalize()} ConfusionMatrix diagonal: "
                    f"{value.diagonal()}"
                )
                if self.global_rank == 0:
                    log_confusion_matrix_advanced(
                        value,
                        class_names=class_names,
                        step=None,
                        split=split,
                    )
                self.log(
                    f"{split}_{metric_name}",
                    value.diagonal().nanmean(),
                    sync_dist=sync,
                )
            elif "top" in metric_name:
                logger.info(f"{split.capitalize()} {metric_name}: {value}")
                self.log(f"{split}_{metric_name}", value, sync_dist=sync)
            elif "weight_overlap" in metric_name:
                logger.info(f"{split.capitalize()} {metric_name}: {value}")
                self.log(metric_name, value, sync_dist=sync)
            else:
                per_class = [
                    f"{cl}: {v:.2f}" for cl, v in zip(class_names, value)
                ]
                logger.info(f"{split.capitalize()} {metric_name}: {per_class}")
                self.log(
                    f"{split}_{metric_name}", value.nanmean(), sync_dist=sync
                )

        m1.reset()
        m2.reset()
        if wd is not None:
            wd.reset()
        if wo is not None:
            wo.reset()

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics("test")

    def enable_full_length_inference(
        self, max_inference_seq_len: Optional[int] = None
    ) -> None:
        """Switch the PLM to full-length embedding mode."""
        self.model.seq_embed_model.full_length_inference = True
        if max_inference_seq_len is not None:
            self.model.seq_embed_model.max_inference_seq_len = max_inference_seq_len

    def disable_full_length_inference(self) -> None:
        """Revert to padded embedding mode."""
        self.model.seq_embed_model.full_length_inference = False

    def _get_seq_dim(self, sequences: List[str]) -> int:
        """Sequence feature dimension: actual max length (inference) or padded length (training / validation)."""
        if (
            hasattr(self.model, "seq_embed_model")
            and getattr(self.model.seq_embed_model, "full_length_inference", False)
            and not self.training
        ):
            return max(len(seq) for seq in sequences)
        return self.max_seq_len - 2

    @staticmethod
    def _calculate_contacting_res_weight(
        sequences: List[str],
        interacting_res: List[Dict[str, int]],
        seq_dim: int,
    ) -> torch.Tensor:
        """Build a ``(N, seq_dim, 1)`` weight tensor from per-residue interface annotations.

        Positions in ``interacting_res`` get weight 1; if no contacts are
        annotated the entire sequence is weighted 1.0 
        (usually happens only for very long sequences where none of max_len residues are on the interface, or for monomers).
        """
        weights = torch.zeros(len(sequences), seq_dim)
        for i, seq in enumerate(sequences):
            if len(interacting_res[i]) == 0:
                weights[i, : len(seq)] = 1.0
                continue
            positions: List[int] = []
            for pos in interacting_res[i].keys():
                try:
                    idx = int(pos[3:]) - 1
                    if 0 <= idx < seq_dim:
                        positions.append(idx)
                except ValueError:
                    continue
            if positions:
                weights[i, positions] = 1.0
            else:
                weights[i, : len(seq)] = 1.0

        return weights.unsqueeze(2)

    def configure_optimizers(self) -> Dict[str, Any]:
        """AdamW with per-group LR and OneCycleLR (one step per epoch)."""
        pooling_params: List[torch.nn.Parameter] = []
        non_pooling_params: List[torch.nn.Parameter] = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logger.info(f"Trainable parameter: {name}, {param.shape}")
                if "feature_pooling_strategy" in name:
                    pooling_params.append(param)
                else:
                    non_pooling_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": non_pooling_params, "lr": 5e-4},
                {"params": pooling_params, "lr": 5e-4},
            ],
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=True,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[5e-3, 5e-4],
            total_steps=self.trainer.max_epochs,
            pct_start=0.065,
            div_factor=10.0,
            final_div_factor=100.0,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
