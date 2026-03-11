import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.nn.models import GCN


class GATLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 1,
        edge_dim: Optional[int] = None,
        concat: bool = False,
        gr_layer: nn.Module = GATConv,
        dropout: float = 0.0,
        use_residual: bool = True,
    ):
        """
        GATLayer.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.edge_dim = edge_dim
        self.concat = concat
        self.dropout = dropout
        self.use_residual = use_residual
        self.gat = gr_layer(
            in_channels,
            out_channels,
            num_heads,
            edge_dim=edge_dim,
            concat=concat,
            dropout=dropout,
            residual=use_residual,
        )
        if edge_dim is not None:
            self.edge_dim = edge_dim
            self.edge_norm = nn.LayerNorm(edge_dim)
            self.edge_importance = nn.Parameter(torch.ones(1))
        else:
            self.edge_dim = None
            self.edge_norm = None
            self.edge_importance = None

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ):
        if (edge_features is not None) and (self.edge_dim is None):
            warnings.warn(
                "Edge features were provided, but no edge dimension was specified. Edge features will be ignored."
            )
        elif edge_features is not None:
            edge_features = self.edge_importance * edge_features

        # GAT layer
        if return_attention_weights:
            node_features, (edge_index, edge_attr) = self.gat(
                node_features,
                edge_index,
                edge_features,
                return_attention_weights=return_attention_weights,
            )
            return node_features, (edge_index, edge_attr)

        else:
            node_features = self.gat(node_features, edge_index, edge_features)
            return node_features


class GCNConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=0.1,
        activation=nn.GELU(),
        use_residual: bool = True,
        num_heads: int = 1,
    ):
        """
        GCNConv layer.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.use_residual = use_residual
        self.num_heads = num_heads  # Unused, only for consistency
        if self.use_residual:
            self.alpha = torch.nn.Parameter(torch.tensor(0.5))
        self.gcn = GCN(
            in_channels=in_channels,
            hidden_channels=in_channels // 4,
            out_channels=out_channels,
            num_layers=1,
            dropout=dropout,
            act=activation,
            norm=nn.LayerNorm(out_channels),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
    ):
        node_features_updated = self.gcn(
            x=node_features, edge_index=edge_index, edge_attr=edge_features
        )
        if self.use_residual:
            node_features = self.alpha * node_features + (
                1 - self.alpha
            ) * self.dropout(node_features_updated)
        else:
            node_features = self.dropout(node_features_updated)

        return node_features


class Identity(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_heads: int = 1,
    ):
        """
        Identity layer to test baseline without any information about the neighbors.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.num_heads = num_heads

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
    ):
        return node_features
