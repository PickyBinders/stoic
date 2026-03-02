from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger

from stoic.layers import Identity


class MaskedInstanceNorm1d(nn.Module):
    """InstanceNorm1d that ignores padding positions when computing statistics.

    Standard InstanceNorm1d normalizes over the full length dimension including
    padding, which causes different outputs depending on padding length. This
    version computes mean/variance only over valid (non-masked) positions,
    making the output invariant to padding strategy.

    Uses the same parameter names (weight, bias) as nn.InstanceNorm1d.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, C, L) tensor.
            mask: (B, L) boolean tensor where True = padding (ignore).
                  When None, falls back to standard InstanceNorm behaviour.
        """
        if mask is None:
            mean = x.mean(dim=2, keepdim=True)
            var = x.var(dim=2, keepdim=True, unbiased=False)
        else:
            # valid_mask: (B, 1, L) — 1.0 for real tokens, 0.0 for padding
            valid_mask = (~mask).unsqueeze(1).to(x.dtype)
            count = valid_mask.sum(dim=2, keepdim=True).clamp(min=1)  # (B, 1, 1)
            mean = (x * valid_mask).sum(dim=2, keepdim=True) / count
            var = ((x - mean) ** 2 * valid_mask).sum(dim=2, keepdim=True) / count

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if mask is not None:
            x_norm = x_norm * (~mask).unsqueeze(1).to(x.dtype)

        if self.affine:
            x_norm = x_norm * self.weight[None, :, None] + self.bias[None, :, None]

        return x_norm

    def extra_repr(self) -> str:
        return f"{self.num_features}, eps={self.eps}, affine={self.affine}"


class FeaturePoolingStrategy(ABC, torch.nn.Module):
    """Abstract base class defining interface for feature pooling strategies"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def pool_node_features(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor = None,
        edge_index: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Pool node features according to the implemented strategy.

        Args:
            features (torch.Tensor): Node features to be pooled, shape [num_nodes, seq_len, hidden_dim]
            attention_mask (torch.Tensor, optional): Boolean mask indicating which positions to ignore (True = masked),
                                                   shape [num_nodes, seq_len]
            edge_index (torch.Tensor, optional): Graph connectivity in COO format, shape [2, num_edges]

        Returns:
            torch.Tensor: Pooled node features, shape [num_nodes, hidden_dim]
        """
        pass


class AveragePooling(FeaturePoolingStrategy):
    """Average pooling strategy"""

    def __init__(
        self,
        emb_dim: int,
        output_dim: int,
        **kwargs,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.norm = MaskedInstanceNorm1d(num_features=emb_dim, affine=True)

        self.mlp = (
            nn.Sequential(
                nn.Linear(emb_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim, output_dim),
            )
            if emb_dim != output_dim
            else Identity(in_channels=emb_dim, out_channels=emb_dim)
        )

    def pool_node_features(
        self,
        node_features: torch.Tensor,
        contacting_res_weight: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        edge_index: torch.Tensor = None,
    ) -> torch.Tensor:
        if contacting_res_weight is None:
            if attention_mask is not None:
                node_features = node_features.masked_fill(
                    attention_mask.unsqueeze(-1), 0
                )
                return node_features.sum(dim=1) / (~attention_mask).sum(
                    dim=1, keepdim=True
                )
            return node_features.mean(dim=1)

        node_features = self.norm(
            node_features.transpose(1, 2), mask=attention_mask
        ).transpose(1, 2)
        num_active = contacting_res_weight.sum(dim=1, keepdim=True)
        normalized_mask = contacting_res_weight / num_active
        node_features = node_features * normalized_mask
        node_features = self.mlp(node_features)

        return node_features.sum(dim=1)


class LinearPooling(FeaturePoolingStrategy):
    def __init__(
        self,
        emb_dim,
        output_dim,
        num_heads=1,
        return_weights=False,
        reduction_factor=4,
        threshold=0.5,
        use_soft_pooling=False,
        **kwargs,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.reduction_factor = reduction_factor
        self.use_soft_pooling = use_soft_pooling
        self.threshold = threshold
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // self.reduction_factor),
            nn.LayerNorm(emb_dim // self.reduction_factor),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim // self.reduction_factor, 1),
        )
        self.norm = MaskedInstanceNorm1d(num_features=emb_dim, affine=True)
        self.return_weights = return_weights
        self.mlp = (
            nn.Sequential(
                nn.Linear(emb_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim, output_dim),
            )
            if emb_dim != output_dim
            else nn.Identity()
        )

    def pool_node_features(
        self,
        node_features: torch.Tensor,
        contacting_res_weight: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        edge_index: torch.Tensor = None,
    ):
        node_features = self.norm(
            node_features.transpose(1, 2), mask=attention_mask
        ).transpose(1, 2)
        average_weights_logits = self.linear(node_features)
        average_weights_logits = torch.squeeze(average_weights_logits, dim=-1)

        if attention_mask is not None:
            average_weights_logits = average_weights_logits.masked_fill(
                attention_mask, -1e9
            )
        average_weights = torch.sigmoid(average_weights_logits)

        if self.use_soft_pooling:
            weighted_features = node_features * average_weights.unsqueeze(-1)
            pooled_features = weighted_features.sum(dim=1) / average_weights.sum(
                dim=1, keepdim=True
            )
        else:
            contacting_res = (average_weights > self.threshold).float()
            num_active = contacting_res.sum(dim=1, keepdim=True)
            num_active = torch.clamp(num_active, min=1.0)
            normalized_contacting_res = contacting_res / num_active

            pooled_features = node_features * normalized_contacting_res.unsqueeze(-1)
            pooled_features = pooled_features.sum(dim=1)

        pooled_features = self.mlp(pooled_features)
        if self.return_weights:
            return pooled_features, average_weights
        else:
            return pooled_features


class SelfAttentionPooling(FeaturePoolingStrategy):
    def __init__(
        self,
        emb_dim,
        output_dim,
        hidden_dim=None,
        num_heads=4,
        return_weights=False,
        reduction_factor=4,
        threshold=0.5,
        use_soft_pooling=False,
        **kwargs,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else emb_dim
        self.layer_norm = MaskedInstanceNorm1d(num_features=emb_dim, affine=True)
        self.norm = MaskedInstanceNorm1d(num_features=emb_dim, affine=True)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            dropout=0.35,
            batch_first=True,
        )
        self.reduction_factor = reduction_factor
        self.linear = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // self.reduction_factor),
            nn.LayerNorm(emb_dim // self.reduction_factor),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(emb_dim // self.reduction_factor, 1),
        )
        self.return_weights = return_weights
        self.mlp = (
            nn.Sequential(
                nn.Linear(emb_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim, output_dim),
            )
            if emb_dim != output_dim
            else nn.Identity()
        )
        self.threshold = threshold
        self.use_soft_pooling = use_soft_pooling

    def pool_node_features(
        self,
        node_features: torch.Tensor,
        contacting_res_weight: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        edge_index: torch.Tensor = None,
    ) -> torch.Tensor:
        node_features = self.layer_norm(
            node_features.transpose(1, 2), mask=attention_mask
        ).transpose(1, 2)
        attn_output, attn_weights = self.self_attention(
            query=node_features,
            key=node_features,
            value=node_features,
            key_padding_mask=attention_mask,
            need_weights=self.return_weights,
            average_attn_weights=True,
        )
        attn_output = attn_output + node_features
        attn_output = self.norm(
            attn_output.transpose(1, 2), mask=attention_mask
        ).transpose(1, 2)
        average_weights_logits = self.linear(attn_output)
        average_weights_logits = torch.squeeze(average_weights_logits, dim=-1)

        if attention_mask is not None:
            average_weights_logits = average_weights_logits.masked_fill(
                attention_mask, -1e9
            )

        average_weights = torch.sigmoid(average_weights_logits)

        if self.use_soft_pooling:
            weighted_features = node_features * average_weights.unsqueeze(-1)
            pooled_features = weighted_features.sum(dim=1) / average_weights.sum(
                dim=1, keepdim=True
            )
        else:
            contacting_res = (average_weights > self.threshold).float()
            num_active = contacting_res.sum(dim=1, keepdim=True)
            num_active = torch.clamp(num_active, min=1.0)
            normalized_contacting_res = contacting_res / num_active

            pooled_features = node_features * normalized_contacting_res.unsqueeze(-1)
            pooled_features = pooled_features.sum(dim=1)

        pooled_features = self.mlp(pooled_features)
        if self.return_weights:
            return pooled_features, average_weights
        else:
            return pooled_features


class NeighborContextSelfAttentionPooling(SelfAttentionPooling):
    """Self-attention pooling conditioned on graph-neighbor residue context.

    Query residues always come from the node itself. Keys/values are built from
    the node plus its graph neighbors (from ``edge_index``), allowing interface
    residue scoring to depend on partner context while preserving output shape.
    """

    def __init__(
        self,
        emb_dim,
        output_dim,
        hidden_dim=None,
        num_heads=4,
        return_weights=False,
        reduction_factor=4,
        threshold=0.5,
        use_soft_pooling=False,
        max_context_neighbors: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            emb_dim=emb_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            return_weights=return_weights,
            reduction_factor=reduction_factor,
            threshold=threshold,
            use_soft_pooling=use_soft_pooling,
        )
        self.max_context_neighbors = max_context_neighbors

    def _build_neighbor_context(
        self,
        node_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        edge_index: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build batched K/V context tensors with per-node neighbor padding.

        Returns:
            context_features: (B, (1 + max_neighbors) * L, D)
            context_mask: (B, (1 + max_neighbors) * L), True = ignore token
        """
        batch_size, seq_len, emb_dim = node_features.shape
        device = node_features.device

        if attention_mask is None:
            attention_mask = torch.zeros(
                (batch_size, seq_len),
                dtype=torch.bool,
                device=device,
            )

        if edge_index is None or edge_index.numel() == 0:
            return node_features, attention_mask

        neighbor_sets: List[set] = [set() for _ in range(batch_size)]
        edges = edge_index.detach().cpu().t().tolist()
        for src, dst in edges:
            if 0 <= src < batch_size and 0 <= dst < batch_size and src != dst:
                neighbor_sets[src].add(dst)
                neighbor_sets[dst].add(src)

        sampled_neighbors: List[List[int]] = []
        for neighbors in neighbor_sets:
            neighbors_list = sorted(neighbors)
            if (
                self.max_context_neighbors is not None
                and len(neighbors_list) > self.max_context_neighbors
            ):
                perm = torch.randperm(len(neighbors_list))
                sampled_idx = perm[: self.max_context_neighbors].tolist()
                neighbors_list = [neighbors_list[i] for i in sampled_idx]
            sampled_neighbors.append(neighbors_list)

        max_neighbors = max((len(neighbors) for neighbors in sampled_neighbors), default=0)
        if max_neighbors == 0:
            return node_features, attention_mask

        context_len = (1 + max_neighbors) * seq_len
        context_features = node_features.new_zeros((batch_size, context_len, emb_dim))
        context_mask = torch.ones(
            (batch_size, context_len),
            dtype=torch.bool,
            device=device,
        )

        for node_idx in range(batch_size):
            # Slot 0: own residues
            context_features[node_idx, :seq_len] = node_features[node_idx]
            context_mask[node_idx, :seq_len] = attention_mask[node_idx]

            # Slots 1..N: neighbors (padded to max_neighbors by masked slots)
            for slot_idx, neighbor_idx in enumerate(sampled_neighbors[node_idx]):
                start = (slot_idx + 1) * seq_len
                end = start + seq_len
                context_features[node_idx, start:end] = node_features[neighbor_idx]
                context_mask[node_idx, start:end] = attention_mask[neighbor_idx]

        return context_features, context_mask

    def pool_node_features(
        self,
        node_features: torch.Tensor,
        contacting_res_weight: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        edge_index: torch.Tensor = None,
    ) -> torch.Tensor:
        node_features = self.layer_norm(
            node_features.transpose(1, 2), mask=attention_mask
        ).transpose(1, 2)
        context_features, context_mask = self._build_neighbor_context(
            node_features=node_features,
            attention_mask=attention_mask,
            edge_index=edge_index,
        )
        attn_output, attn_weights = self.self_attention(
            query=node_features,
            key=context_features,
            value=context_features,
            key_padding_mask=context_mask,
            need_weights=self.return_weights,
            average_attn_weights=True,
        )
        attn_output = attn_output + node_features
        attn_output = self.norm(
            attn_output.transpose(1, 2), mask=attention_mask
        ).transpose(1, 2)
        average_weights_logits = self.linear(attn_output)
        average_weights_logits = torch.squeeze(average_weights_logits, dim=-1)

        if attention_mask is not None:
            average_weights_logits = average_weights_logits.masked_fill(
                attention_mask, -1e9
            )

        average_weights = torch.sigmoid(average_weights_logits)

        if self.use_soft_pooling:
            weighted_features = node_features * average_weights.unsqueeze(-1)
            pooled_features = weighted_features.sum(dim=1) / average_weights.sum(
                dim=1, keepdim=True
            )
        else:
            contacting_res = (average_weights > self.threshold).float()
            num_active = contacting_res.sum(dim=1, keepdim=True)
            num_active = torch.clamp(num_active, min=1.0)
            normalized_contacting_res = contacting_res / num_active

            pooled_features = node_features * normalized_contacting_res.unsqueeze(-1)
            pooled_features = pooled_features.sum(dim=1)

        pooled_features = self.mlp(pooled_features)
        if self.return_weights:
            return pooled_features, average_weights
        else:
            return pooled_features


if __name__ == "__main__":
    torch.manual_seed(0)
    emb_dim, out_dim = 32, 32
    batch_size, seq_len = 4, 16

    sap = SelfAttentionPooling(
        emb_dim=emb_dim,
        output_dim=out_dim,
        return_weights=True,
        use_soft_pooling=True,
        num_heads=4,
    )
    nsap = NeighborContextSelfAttentionPooling(
        emb_dim=emb_dim,
        output_dim=out_dim,
        return_weights=True,
        use_soft_pooling=True,
        num_heads=4,
    )
    # Match weights so we can compare behavior in no-neighbor cases.
    nsap.load_state_dict(sap.state_dict())
    sap.eval()
    nsap.eval()

    x = torch.randn(batch_size, seq_len, emb_dim)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    mask[0, -3:] = True
    mask[1, -1:] = True

    # Test 1: no edges -> NeighborContextSelfAttentionPooling == SelfAttentionPooling
    out_sap, w_sap = sap.pool_node_features(x, attention_mask=mask, edge_index=None)
    out_nsap, w_nsap = nsap.pool_node_features(x, attention_mask=mask, edge_index=None)
    assert out_sap.shape == (batch_size, out_dim)
    assert w_sap.shape == (batch_size, seq_len)
    assert torch.allclose(out_sap, out_nsap, atol=1e-6), "No-edge outputs should match."
    assert torch.allclose(w_sap, w_nsap, atol=1e-6), "No-edge residue weights should match."

    # Test 2: context builder handles variable neighbor counts and padding
    edge_index = torch.tensor(
        [
            [0, 0, 1, 3],  # src
            [1, 2, 2, 2],  # dst
        ],
        dtype=torch.long,
    )
    context_features, context_mask = nsap._build_neighbor_context(x, mask, edge_index)
    # Undirected conversion inside builder gives neighbors:
    # node0:{1,2} node1:{0,2} node2:{0,1,3} node3:{2} -> max_neighbors=3
    expected_context_len = (1 + 3) * seq_len
    assert context_features.shape == (batch_size, expected_context_len, emb_dim)
    assert context_mask.shape == (batch_size, expected_context_len)
    assert context_mask.dtype == torch.bool

    # Test 3: with neighbors, outputs remain finite and correctly shaped
    out_ctx, w_ctx = nsap.pool_node_features(x, attention_mask=mask, edge_index=edge_index)
    assert out_ctx.shape == (batch_size, out_dim)
    assert w_ctx.shape == (batch_size, seq_len)
    assert torch.isfinite(out_ctx).all(), "Pooled features contain NaN/Inf."
    assert torch.isfinite(w_ctx).all(), "Residue weights contain NaN/Inf."

    # Test 3b: max_context_neighbors caps context length
    nsap_capped = NeighborContextSelfAttentionPooling(
        emb_dim=emb_dim,
        output_dim=out_dim,
        return_weights=True,
        use_soft_pooling=True,
        num_heads=4,
        max_context_neighbors=2,
    )
    nsap_capped.load_state_dict(sap.state_dict(), strict=False)
    capped_context_features, _ = nsap_capped._build_neighbor_context(x, mask, edge_index)
    assert capped_context_features.shape[1] == (1 + 2) * seq_len

    # Test 4: single-sequence graph should degrade to self-only behavior
    x1 = torch.randn(1, seq_len, emb_dim)
    m1 = torch.zeros(1, seq_len, dtype=torch.bool)
    e1 = torch.tensor([[0], [0]], dtype=torch.long)  # self-loop only
    out_s1, w_s1 = sap.pool_node_features(x1, attention_mask=m1, edge_index=e1)
    out_n1, w_n1 = nsap.pool_node_features(x1, attention_mask=m1, edge_index=e1)
    assert torch.allclose(out_s1, out_n1, atol=1e-6), "Single-node outputs should match."
    assert torch.allclose(w_s1, w_n1, atol=1e-6), "Single-node weights should match."

    print("All NeighborContextSelfAttentionPooling smoke tests passed.")
