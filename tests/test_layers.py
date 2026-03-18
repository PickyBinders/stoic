import pytest
import torch

from stoic.layers import GATLayer, GCNConv, Identity

torch_geometric = pytest.importorskip("torch_geometric")


def _dummy_gat_layer_class():
    class DummyGAT(torch.nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            heads,
            edge_dim=None,
            concat=False,
            dropout=0.0,
            residual=False,
        ):
            super().__init__()
            self.last_edge_features = None

        def forward(
            self,
            node_features,
            edge_index,
            edge_features=None,
            return_attention_weights=False,
        ):
            self.last_edge_features = edge_features
            if return_attention_weights:
                edge_attr = torch.ones(edge_index.size(1), device=node_features.device)
                return node_features, (edge_index, edge_attr)
            return node_features

    return DummyGAT


def test_identity_layer_returns_input() -> None:
    layer = Identity(in_channels=8, out_channels=8)
    x = torch.randn(4, 8)
    out = layer(x)
    assert torch.allclose(out, x)


def test_gat_layer_output_shape() -> None:
    layer = GATLayer(
        in_channels=8,
        out_channels=8,
        num_heads=1,
        concat=False,
        gr_layer=_dummy_gat_layer_class(),
    )
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    out = layer(x, edge_index)
    assert out.shape == (3, 8)


def test_gat_layer_warns_if_edge_features_without_edge_dim() -> None:
    layer = GATLayer(
        in_channels=8,
        out_channels=8,
        num_heads=1,
        concat=False,
        edge_dim=None,
        gr_layer=_dummy_gat_layer_class(),
    )
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_features = torch.randn(edge_index.size(1), 4)
    with pytest.warns(UserWarning, match="Edge features were provided"):
        out = layer(x, edge_index, edge_features=edge_features)
    assert out.shape == (3, 8)


def test_gat_layer_returns_attention_weights() -> None:
    layer = GATLayer(
        in_channels=8,
        out_channels=8,
        num_heads=1,
        concat=False,
        gr_layer=_dummy_gat_layer_class(),
    )
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    node_out, (ei, edge_attr) = layer(
        x, edge_index, return_attention_weights=True
    )
    assert node_out.shape == (3, 8)
    assert ei.shape[0] == 2
    assert edge_attr.shape[0] == ei.shape[1]


def test_gat_layer_scales_edge_features_with_edge_importance(monkeypatch) -> None:
    layer = GATLayer(
        in_channels=8,
        out_channels=8,
        num_heads=1,
        concat=False,
        edge_dim=2,
        gr_layer=_dummy_gat_layer_class(),
    )
    layer.edge_importance.data.fill_(2.0)
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_features = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    _ = layer(x, edge_index, edge_features=edge_features)
    torch.testing.assert_close(layer.gat.last_edge_features, edge_features * 2.0)


def test_gcnconv_residual_path_formula(monkeypatch) -> None:
    layer = GCNConv(in_channels=8, out_channels=8, dropout=0.0, use_residual=True)
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    updated = torch.randn_like(x)
    layer.alpha.data.fill_(0.25)

    def fake_gcn(**kwargs):
        return updated

    monkeypatch.setattr(layer.gcn, "forward", fake_gcn)
    out = layer(x, edge_index)
    expected = 0.25 * x + 0.75 * updated
    torch.testing.assert_close(out, expected)


def test_gcnconv_non_residual_returns_updated_with_dropout_off(monkeypatch) -> None:
    layer = GCNConv(in_channels=8, out_channels=8, dropout=0.0, use_residual=False)
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    updated = torch.randn_like(x)

    def fake_gcn(**kwargs):
        return updated

    monkeypatch.setattr(layer.gcn, "forward", fake_gcn)
    out = layer(x, edge_index)
    torch.testing.assert_close(out, updated)
