"""Loss functions for stoichiometry model training."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance (Lin et al., 2017).

    Down-weights well-classified examples so the model focuses on hard,
    misclassified ones.

    Args:
        weight: Per-class weights for cross-entropy, shape ``(C,)``.
        gamma: Focusing parameter. Higher values down-weight easy examples more.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape ``(N, C)``.
            targets: Ground-truth class indices of shape ``(N,)``.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.weight)
        if self.weight is not None:
            ce_loss = ce_loss / self.weight.mean()
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class ComplexLoss(nn.Module):
    """Complex-level loss combining per-node mean and max losses.

    Groups node predictions by ``complex_id`` and computes a weighted
    combination of the average and maximum per-node loss within each complex.

    Args:
        alpha: Interpolation weight between mean (``alpha``) and max
            (``1 - alpha``) losses.
        gamma: Focusing parameter when ``use_focal=True``.
        use_focal: Apply focal weighting to per-node cross-entropy.
        weight: Per-class weights for cross-entropy, shape ``(C,)``.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 2.0,
        use_focal: bool = False,
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.use_focal = use_focal
        self.weight = weight

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, complex_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: Node logits of shape ``(N, C)``.
            targets: Ground-truth class indices of shape ``(N,)``.
            complex_id: Integer tensor mapping each node to its complex.
        """
        if not isinstance(complex_id, torch.Tensor):
            complex_id = torch.tensor(complex_id)

        node_losses = self._per_node_loss(logits, targets)
        unique_complexes = torch.unique(complex_id)

        complex_losses = []
        for cid in unique_complexes:
            mask = complex_id == cid
            losses_c = node_losses[mask]
            if len(losses_c) > 0:
                avg_loss = torch.mean(losses_c) * len(losses_c)
                max_loss = torch.max(losses_c)
                complex_losses.append(self.alpha * avg_loss + (1 - self.alpha) * max_loss)

        if complex_losses:
            return torch.mean(torch.stack(complex_losses))
        return torch.mean(node_losses)

    def _per_node_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute (optionally focal-weighted) per-node cross-entropy."""
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.weight)
        if self.weight is not None:
            ce_loss = ce_loss / self.weight.mean()
        if self.use_focal:
            pt = torch.exp(-ce_loss)
            ce_loss = (1 - pt) ** self.gamma * ce_loss
        return ce_loss


class ComplexProductLoss(nn.Module):
    """Complex-level product loss using log-sum aggregation.

    For each complex, per-node losses are shifted by a margin, log-transformed,
    and summed. This approximates a product of individual loss terms in
    log-space, encouraging ALL nodes in a complex to be classified correctly.

    Args:
        gamma: Focusing parameter when ``use_focal=True``.
        use_focal: Apply focal weighting to per-node cross-entropy.
        weight: Per-class weights for cross-entropy, shape ``(C,)``.
        temperature: Divisor applied to the log-sum before clamping.
        margin: Additive shift before taking the log (ensures positivity).
        min_value: Lower clamp for the tempered log-sum.
        max_value: Upper clamp for the tempered log-sum.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        use_focal: bool = False,
        weight: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        margin: float = 1.0,
        min_value: float = 1e-6,
        max_value: float = 30.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.use_focal = use_focal
        self.weight = weight
        self.temperature = temperature
        self.margin = margin
        self.min_value = min_value
        self.max_value = max_value

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, complex_id: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: Node logits of shape ``(N, C)``.
            targets: Ground-truth class indices of shape ``(N,)``.
            complex_id: Integer tensor mapping each node to its complex.
        """
        if not isinstance(complex_id, torch.Tensor):
            complex_id = torch.tensor(complex_id)

        node_losses = self._per_node_loss(logits, targets)
        unique_complexes = torch.unique(complex_id)

        complex_losses = []
        for cid in unique_complexes:
            mask = complex_id == cid
            losses_c = node_losses[mask]
            if len(losses_c) > 0:
                log_sum = torch.log(losses_c + self.margin).sum()
                log_sum = (log_sum / self.temperature).clamp(self.min_value, self.max_value)
                complex_losses.append(log_sum)

        if complex_losses:
            return torch.mean(torch.stack(complex_losses))
        return torch.mean(node_losses)

    def _per_node_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute (optionally focal-weighted) per-node cross-entropy."""
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.weight)
        if self.weight is not None:
            ce_loss = ce_loss / self.weight.mean()
        if self.use_focal:
            pt = torch.exp(-ce_loss)
            ce_loss = (1 - pt) ** self.gamma * ce_loss
        return ce_loss


class SparsityLoss(nn.Module):
    """Encourage attention weights to concentrate on a small fraction of residues.

    Penalises when the sum of the top-k weights (where k is a percentage of
    the sequence length) falls below a target concentration.

    Args:
        topk_percent: Fraction of residues considered "top-k".
        target_concentration: Desired minimum mass in the top-k positions.
    """

    def __init__(self, topk_percent: float = 0.2, target_concentration: float = 0.95):
        super().__init__()
        self.topk_percent = topk_percent
        self.target_concentration = target_concentration

    def forward(
        self,
        attention_weights: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            attention_weights: Shape ``(B, L)``.
            mask: Boolean mask where ``True`` = padding (ignore).

        Returns:
            Tuple of (scalar loss, dict with ``"topk"`` and ``"concentration"``).
        """
        if mask is not None:
            valid_weights = attention_weights.masked_fill(mask, 0.0)
            seq_lengths = (~mask).sum(dim=1)
        else:
            valid_weights = attention_weights
            seq_lengths = torch.full(
                (attention_weights.shape[0],),
                attention_weights.shape[1],
                device=attention_weights.device,
            )
        topk_loss, concentration = self._compute_topk_loss(valid_weights, seq_lengths)
        loss_dict = {
            "topk": topk_loss.mean(),
            "concentration": concentration.mean(),
        }
        return loss_dict["topk"], loss_dict

    def _compute_topk_loss(
        self, weights: torch.Tensor, seq_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k = torch.maximum(
            torch.ones_like(seq_lengths), (seq_lengths * self.topk_percent).int()
        )
        top_k_values, _ = torch.topk(weights, k.max().item(), dim=1)
        k_mask = (
            torch.arange(k.max().item(), device=weights.device)[None, :] < k[:, None]
        )
        concentration = torch.sum(top_k_values * k_mask, dim=1)
        loss = F.relu(self.target_concentration - concentration)
        return loss, concentration


class ResidueWeightL1Loss(nn.Module):
    """L1 loss between predicted residue-level weights and binary contact labels."""

    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()

    def forward(
        self, residue_weights: torch.Tensor, target_residue_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            residue_weights: Predicted weights, shape ``(B, L)``.
            target_residue_weights: Binary targets, shape ``(B, L, 1)``.
        """
        return self.l1_loss(residue_weights, target_residue_weights.squeeze(2))


class ResidueWeightKLLoss(nn.Module):
    """KL-divergence loss between predicted residue weights and contact labels."""

    def forward(
        self, residue_weights: torch.Tensor, target_residue_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            residue_weights: Predicted weights, shape ``(B, L)``.
            target_residue_weights: Target distribution, shape ``(B, L, 1)``.
        """
        return F.kl_div(
            torch.log(residue_weights + 1e-9),
            target_residue_weights.squeeze(2),
            reduction="batchmean",
        )


class ResidueWeightFocalLoss(nn.Module):
    """Focal binary cross-entropy between predicted residue weights and contact labels.

    Args:
        alpha: Balancing factor for the focal term.
        gamma: Focusing parameter.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self, residue_weights: torch.Tensor, target_residue_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            residue_weights: Predicted weights in ``[0, 1]``, shape ``(B, L)``.
            target_residue_weights: Binary targets, shape ``(B, L, 1)``.
        """
        target_residue_weights = target_residue_weights.squeeze(2).to(
            residue_weights.device
        )
        bce = F.binary_cross_entropy(
            residue_weights,
            target_residue_weights,
            reduction="none",
        )
        pt = torch.exp(-bce)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()
