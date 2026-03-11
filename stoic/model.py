from collections import Counter
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin, login
from loguru import logger

from stoic.feature_pooling import (
    AveragePooling,
    LinearPooling,
    NeighborContextSelfAttentionPooling,
    SelfAttentionPooling,
)
from stoic.utils import (
    top_n_stoichiometry_combinations,
)
from stoic.layers import Identity
from stoic import layers, feature_pooling
from stoic.seq_emb_models import Esm2


class Stoic(nn.Module, PyTorchModelHubMixin, repo_url="stoic", license="mit"):
    """Model for protein complex stoichiometry prediction.

    The model pipeline is:
    1) sequence encoder (e.g. ESM-2),
    2) feature pooling over residues,
    3) graph-based sequence feature encoder, if specified otherwise, it will be an identity layer,
    4) per-node stoichiometry classifier.
    """

    def __init__(
        self,
        stoichiometry_classes_to_use: List[int],
        seq_embed_model_name: str = "facebook/esm2_t33_650M_UR50D",
        seq_feature_encoder: str = "Identity",
        feature_pooling_strategy: str = "AveragePooling",
        **kwargs: Any,
    ) -> None:
        """Initialize model components and configuration.

        Args:
            stoichiometry_classes_to_use: Output class labels used at `predict_stoichiometry`.
            seq_embed_model_name: Hugging Face model id/path for sequence embeddings.
            seq_feature_encoder: Name of the sequence feature encoder class (from stoic.layers).
            feature_pooling_strategy: Name of the pooling strategy class (from stoic.feature_pooling).
            **kwargs: Optional component-specific hyperparameters.
        """
        super().__init__()
        self.stoichiometry_classes_to_use = stoichiometry_classes_to_use
        self.num_stoichiometry_classes = len(stoichiometry_classes_to_use)
        self.seq_embed_model_name = seq_embed_model_name
        if "esm2" in seq_embed_model_name.lower():
            self._seq_embed_model_cls = Esm2
        else:
            raise ValueError(f"Invalid sequence embedding model: {seq_embed_model_name}")
        self._seq_feature_encoder_name = seq_feature_encoder
        self._feature_pooling_strategy_name = feature_pooling_strategy
        self.seq_embed_model_chunk_size = kwargs.get("seq_embed_model_chunk_size", 1000)
        self.prediction_head_reduction_factor = kwargs.get(
            "prediction_head_reduction_factor", 4
        )
        self._assign_extra_args(**kwargs)
        self.configure_model()

    def _assign_extra_args(self, **kwargs: Any) -> None:
        """Split free-form kwargs into per-component argument dictionaries."""
        feature_pooling_strategy_args = [
            "fps_num_heads",
            "fps_dropout",
            "fps_output_dim",
            "fps_return_weights",
            "fps_reduction_factor",
            "fps_use_soft_pooling",
            "fps_max_context_neighbors",
        ]
        seq_embed_model_args = [
            "finetune_seq_embed_model",
            "load_in_4bit",
            "max_seq_len",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
        ]
        seq_feature_encoder_args = [
            "seq_feature_encoder_dropout",
            "seq_feature_encoder_num_heads",
        ]
        prediction_head_args = [
            "prediction_head_reduction_factor",
            "prediction_head_dropout",
        ]
        self.feature_pooling_strategy_args = {
            arg: kwargs[arg] for arg in feature_pooling_strategy_args if arg in kwargs
        }
        self.seq_embed_model_args = {
            arg: kwargs[arg] for arg in seq_embed_model_args if arg in kwargs
        }
        self.seq_feature_encoder_args = {
            arg: kwargs[arg] for arg in seq_feature_encoder_args if arg in kwargs
        }
        self.prediction_head_args = {
            arg: kwargs[arg] for arg in prediction_head_args if arg in kwargs
        }
    def configure_model(self) -> None:
        """Instantiate all model submodules from stored configuration."""
        self._configure_seq_embed_model()
        self._configure_feature_pooling_strategy()
        self._configure_seq_feature_encoder()
        self._configure_classification_head()

    def _configure_seq_embed_model(self) -> None:
        """Instantiate the sequence embedding backbone."""
        self.seq_embed_model = self._seq_embed_model_cls(
            model_name=self.seq_embed_model_name,
            max_seq_len=self.seq_embed_model_args.get("max_seq_len", 512),
            full_length_inference=self.seq_embed_model_args.get("full_length_inference", False),
            max_inference_seq_len=self.seq_embed_model_args.get("max_inference_seq_len", None),
            finetune=self.seq_embed_model_args.get("finetune_seq_embed_model", False),
            load_in_4bit=self.seq_embed_model_args.get("load_in_4bit", False),
            lora_r=self.seq_embed_model_args.get("lora_r", 32),
            lora_alpha=self.seq_embed_model_args.get("lora_alpha", 16),
            lora_dropout=self.seq_embed_model_args.get("lora_dropout", 0.05),
        )

    def _configure_feature_pooling_strategy(self) -> None:
        """Instantiate the configured residue feature pooling strategy."""
        if self._feature_pooling_strategy_name == "AveragePooling":
            self.feature_pooling_strategy = AveragePooling(
                emb_dim=self.seq_embed_model.seq_embed_size,
                output_dim=self.feature_pooling_strategy_args.get(
                    "fps_output_dim", self.seq_embed_model.seq_embed_size
                ),
            )
        elif self._feature_pooling_strategy_name in [
            "LinearPooling",
            "SelfAttentionPooling",
            "NeighborContextSelfAttentionPooling",
        ]:
            pooling_cls = getattr(feature_pooling, self._feature_pooling_strategy_name)
            self.feature_pooling_strategy = pooling_cls(
                emb_dim=self.seq_embed_model.seq_embed_size,
                output_dim=self.feature_pooling_strategy_args.get(
                    "fps_output_dim", self.seq_embed_model.seq_embed_size
                ),
                reduction_factor=self.feature_pooling_strategy_args.get(
                    "fps_reduction_factor", 1
                ),
                return_weights=self.feature_pooling_strategy_args.get(
                    "fps_return_weights", False
                ),
                use_soft_pooling=self.feature_pooling_strategy_args.get(
                    "fps_use_soft_pooling", False
                ),
                num_heads=self.feature_pooling_strategy_args.get("fps_num_heads", 1),
                max_context_neighbors=self.feature_pooling_strategy_args.get(
                    "fps_max_context_neighbors", None
                ),
            )
        else:
            raise ValueError(
                f"Invalid feature pooling strategy: {self._feature_pooling_strategy_name}. "
                f"Available: AveragePooling, LinearPooling, SelfAttentionPooling, "
                f"NeighborContextSelfAttentionPooling"
            )

    def _configure_seq_feature_encoder(self) -> None:
        """Instantiate the sequence feature encoder block."""
        seq_feature_encoder_cls = getattr(layers, self._seq_feature_encoder_name)
        self.seq_feature_encoder = seq_feature_encoder_cls(
            in_channels=self.feature_pooling_strategy.output_dim,
            out_channels=self.feature_pooling_strategy.output_dim,
            dropout=self.seq_feature_encoder_args.get(
                "seq_feature_encoder_dropout", 0.2,
            ),
            num_heads=self.seq_feature_encoder_args.get(
                "seq_feature_encoder_num_heads", 1
            ),
        )

    def _configure_classification_head(self) -> None:
        """Build the node-level classification head."""
        reduction_factor = self.prediction_head_args.get("prediction_head_reduction_factor", 4)
        self.node_classifier = nn.Sequential(
            nn.Linear(
                self.seq_feature_encoder.out_channels,
                self.seq_feature_encoder.out_channels // reduction_factor,
            ),
            nn.LayerNorm(self.seq_feature_encoder.out_channels // reduction_factor),
            nn.GELU(),
            nn.Dropout(self.prediction_head_args.get("prediction_head_dropout", 0.2)),
            nn.Linear(
                self.seq_feature_encoder.out_channels // reduction_factor,
                self.num_stoichiometry_classes,
            ),
        )

    def get_sequence_embeddings(
        self, sequences: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode sequences in chunks and return embeddings plus padding masks."""
        batch_size = len(sequences)
        device = next(iter(self.node_classifier.parameters())).device
        dtype = next(iter(self.node_classifier.parameters())).dtype
        use_full_length = (
            self.seq_embed_model.full_length_inference and not self.training
        )

        if use_full_length:
            all_embeddings: List[torch.Tensor] = []
            all_masks: List[torch.Tensor] = []
            for i in range(0, batch_size, self.seq_embed_model_chunk_size):
                chunk = sequences[i : i + self.seq_embed_model_chunk_size]
                chunk_emb, chunk_mask = self.seq_embed_model(chunk)
                all_embeddings.append(chunk_emb)
                all_masks.append(chunk_mask)

            max_len = max(emb.size(1) for emb in all_embeddings)
            padded_embeddings: List[torch.Tensor] = []
            padded_masks: List[torch.Tensor] = []
            for emb, mask in zip(all_embeddings, all_masks):
                pad_len = max_len - emb.size(1)
                if pad_len > 0:
                    emb = F.pad(emb, (0, 0, 0, pad_len))
                    mask = F.pad(mask, (0, pad_len), value=True)
                padded_embeddings.append(emb)
                padded_masks.append(mask)

            sequence_embeddings = torch.cat(padded_embeddings, dim=0)
            attention_masks = torch.cat(padded_masks, dim=0)
        else:
            seq_len = self.seq_embed_model.max_seq_len - 2
            sequence_embeddings = torch.empty(
                (batch_size, seq_len, self.seq_embed_model.seq_embed_size),
                device=device, dtype=dtype,
            )
            attention_masks = torch.empty(
                (batch_size, seq_len), device=device, dtype=torch.bool,
            )
            for i in range(0, batch_size, self.seq_embed_model_chunk_size):
                chunk = sequences[i : i + self.seq_embed_model_chunk_size]
                chunk_emb, chunk_mask = self.seq_embed_model(chunk)
                sequence_embeddings[i : i + self.seq_embed_model_chunk_size] = chunk_emb
                attention_masks[i : i + self.seq_embed_model_chunk_size] = chunk_mask

        return sequence_embeddings, attention_masks


    def forward(
        self,
        sequences: List[str],
        edge_index: torch.Tensor,
        contacting_res_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run a forward pass and return prediction artifacts.

        Args:
            sequences: Protein sequences in the input graph.
            edge_index: Graph edges in COO format.
            contacting_res_weight: Optional per-residue weights for pooling.

        Returns:
            Dictionary containing at least:
            - ``attention_mask``: residue padding mask.
            - ``node_scores``: per-node class logits.
            Optionally includes ``residue_weights`` for weighted pooling modes.
        """
        output: Dict[str, torch.Tensor] = {}
        sequence_embeddings, sequence_attention_masks = (
            self.get_sequence_embeddings(sequences)
        )
        output["attention_mask"] = sequence_attention_masks
        if contacting_res_weight is not None:
            contacting_res_weight = contacting_res_weight.to(
                sequence_embeddings.dtype
            ).to(sequence_embeddings.device)

        if getattr(self.feature_pooling_strategy, "return_weights", False):
            pooled_node_features, residue_weights = (
                self.feature_pooling_strategy.pool_node_features(
                    node_features=sequence_embeddings,
                    attention_mask=sequence_attention_masks,
                    edge_index=edge_index,
                    contacting_res_weight=contacting_res_weight,
                )
            )
        else:
            pooled_node_features = self.feature_pooling_strategy.pool_node_features(
                node_features=sequence_embeddings,
                attention_mask=sequence_attention_masks,
                edge_index=edge_index,
                contacting_res_weight=contacting_res_weight,
            )

        node_embeddings_updated = self.seq_feature_encoder(
            node_features=pooled_node_features,
            edge_index=edge_index,
        )
        output["node_scores"] = self.node_classifier(
            node_embeddings_updated
        )
        if getattr(self.feature_pooling_strategy, "return_weights", False):
            output["residue_weights"] = residue_weights

        return output
    
    def get_edge_index(
        self, 
        sequences: List[str]
    ) -> torch.Tensor:
        """Construct a fully connected edge index for the input sequences."""
        edge_i: List[int] = []
        edge_j: List[int] = []
        for i in range(len(sequences)):
            for j in range(len(sequences)):
                edge_i.append(i)
                edge_j.append(j)
        edge_index = torch.tensor(
            [edge_i, edge_j], dtype=torch.long, device=next(self.parameters()).device
        )
        return edge_index

    def enable_full_length_inference(self, max_inference_seq_len: Optional[int] = None) -> None:
        """Switch the sequence embedding model to full-length embedding mode."""
        self.seq_embed_model.full_length_inference = True
        if max_inference_seq_len is not None:
            self.seq_embed_model.max_inference_seq_len = max_inference_seq_len
    
    def disable_full_length_inference(self) -> None:
        self.seq_embed_model.full_length_inference = False

    def predict_stoichiometry(
        self,
        sequences: List[str],
        top_n: int = 3,
        return_residue_weights: bool = False,
    ) -> List[Dict[str, int]]:
        """Predict top-N stoichiometry assignments for input sequences."""
        duplicate_counts = {
            seq: count for seq, count in Counter(sequences).items() if count > 1
        }
        if duplicate_counts:
            warning_message = (
                "Non-unique input sequences detected; duplicated sequences will be "
                "treated as the same sequence. Duplicated sequences: "
                + ", ".join(
                    f"{seq}: {count}" for seq, count in duplicate_counts.items()
                )
            )
            logger.warning(warning_message)

        sequences = list(set(sequences))
        edge_index = self.get_edge_index(sequences)
        output = self.forward(sequences, edge_index)
        node_scores = output["node_scores"]
        top_combinations = top_n_stoichiometry_combinations(
            node_scores.detach().cpu(),
            n=top_n,
            class_labels=self.stoichiometry_classes_to_use,
        )
        result: List[Dict[str, int]] = []
        for combination, _, _ in top_combinations:
            result_item: Dict[str, int] = {}
            for seq, copy_n in zip(sequences, combination):
                result_item[seq] = copy_n
            result.append(result_item)
        if return_residue_weights:
            residue_predictions = {
                "sequences": sequences,
                "pred_residues": output["residue_weights"].detach().cpu().numpy(),
                "attention_mask": output["attention_mask"].detach().cpu().numpy(),
            }
            return result, residue_predictions
        else:
            return result
