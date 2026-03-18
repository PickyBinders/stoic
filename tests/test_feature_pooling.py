import torch

from stoic.feature_pooling import (
    AveragePooling,
    MaskedInstanceNorm1d,
    SelfAttentionPooling,
)


def test_masked_instance_norm_preserves_shape() -> None:
    norm = MaskedInstanceNorm1d(num_features=4, affine=True)
    x = torch.randn(2, 4, 6)
    mask = torch.tensor(
        [[False, False, False, True, True, True], [False, False, True, True, True, True]]
    )
    out = norm(x, mask=mask)
    assert out.shape == x.shape


def test_masked_instance_norm_is_invariant_to_extra_masked_padding() -> None:
    norm = MaskedInstanceNorm1d(num_features=3, affine=True)
    x_short = torch.randn(2, 3, 4)
    mask_short = torch.tensor(
        [[False, False, False, False], [False, False, False, True]],
        dtype=torch.bool,
    )
    x_long = torch.cat([x_short, torch.randn(2, 3, 3)], dim=2)
    mask_long = torch.tensor(
        [
            [False, False, False, False, True, True, True],
            [False, False, False, True, True, True, True],
        ],
        dtype=torch.bool,
    )
    out_short = norm(x_short, mask=mask_short)
    out_long = norm(x_long, mask=mask_long)
    torch.testing.assert_close(out_short, out_long[:, :, :4], rtol=0, atol=1e-6)


def test_average_pooling_without_mask_computes_mean_over_sequence() -> None:
    pool = AveragePooling(emb_dim=2, output_dim=2)
    node_features = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
        ]
    )
    out = pool.pool_node_features(node_features=node_features, attention_mask=None)
    expected = node_features.mean(dim=1)
    torch.testing.assert_close(out, expected)


def test_average_pooling_with_mask_changes_average_correctly() -> None:
    pool = AveragePooling(emb_dim=2, output_dim=2)
    node_features = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]],
            [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
        ]
    )
    attention_mask = torch.tensor(
        [[False, False, True], [False, True, True]], dtype=torch.bool
    )
    out = pool.pool_node_features(node_features=node_features, attention_mask=attention_mask)
    expected = torch.tensor([[2.0, 3.0], [2.0, 4.0]])
    torch.testing.assert_close(out, expected)


def test_average_pooling_all_contacting_matches_plain_average() -> None:
    pool = AveragePooling(emb_dim=2, output_dim=2)
    node_features = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
        ]
    )
    attention_mask = None
    contacting_res_weight = torch.ones((2, 3, 1))

    out_plain = pool.pool_node_features(
        node_features=pool.norm(node_features.transpose(1, 2), mask=attention_mask).transpose(1, 2),
        attention_mask=attention_mask,
        contacting_res_weight=None,
    )
    out_contact = pool.pool_node_features(
        node_features=node_features,
        attention_mask=attention_mask,
        contacting_res_weight=contacting_res_weight,
    )
    torch.testing.assert_close(out_plain, out_contact)


def test_self_attention_pooling_output_contract_without_weights() -> None:
    pool = SelfAttentionPooling(
        emb_dim=8,
        output_dim=8,
        num_heads=2,
        return_weights=False,
        use_soft_pooling=True,
    )
    node_features = torch.randn(3, 5, 8)
    attention_mask = torch.tensor(
        [
            [False, False, False, True, True],
            [False, False, False, False, True],
            [False, False, False, False, False],
        ],
        dtype=torch.bool,
    )
    out = pool.pool_node_features(node_features=node_features, attention_mask=attention_mask)
    assert out.shape == (3, 8)


def test_self_attention_pooling_output_contract_with_weights_and_masked_positions() -> None:
    pool = SelfAttentionPooling(
        emb_dim=8,
        output_dim=8,
        num_heads=2,
        return_weights=True,
        use_soft_pooling=True,
    )
    node_features = torch.randn(2, 6, 8)
    attention_mask = torch.tensor(
        [[False, False, False, True, True, True], [False, False, False, False, True, True]],
        dtype=torch.bool,
    )
    pooled, average_weights = pool.pool_node_features(
        node_features=node_features,
        attention_mask=attention_mask,
    )
    assert pooled.shape == (2, 8)
    assert average_weights.shape == (2, 6)
    assert torch.max(average_weights[attention_mask]) < 1e-6


def test_self_attention_pooling_soft_and_hard_paths_are_finite() -> None:
    node_features = torch.randn(2, 5, 8)
    attention_mask = torch.zeros((2, 5), dtype=torch.bool)

    soft_pool = SelfAttentionPooling(
        emb_dim=8,
        output_dim=8,
        num_heads=2,
        return_weights=False,
        use_soft_pooling=True,
    )
    hard_pool = SelfAttentionPooling(
        emb_dim=8,
        output_dim=8,
        num_heads=2,
        return_weights=False,
        use_soft_pooling=False,
        threshold=1.0,
    )

    out_soft = soft_pool.pool_node_features(node_features=node_features, attention_mask=attention_mask)
    out_hard = hard_pool.pool_node_features(node_features=node_features, attention_mask=attention_mask)

    assert out_soft.shape == (2, 8)
    assert out_hard.shape == (2, 8)
    assert torch.isfinite(out_soft).all()
    assert torch.isfinite(out_hard).all()


def test_neighbor_context_build_no_edges_returns_identity_context() -> None:
    from stoic.feature_pooling import NeighborContextSelfAttentionPooling

    pool = NeighborContextSelfAttentionPooling(
        emb_dim=8,
        output_dim=8,
        num_heads=2,
    )
    node_features = torch.randn(3, 4, 8)
    attention_mask = torch.tensor(
        [[False, False, False, True], [False, False, True, True], [False, False, False, False]],
        dtype=torch.bool,
    )
    context_features, context_mask = pool._build_neighbor_context(
        node_features=node_features,
        attention_mask=attention_mask,
        edge_index=None,
    )
    torch.testing.assert_close(context_features, node_features)
    torch.testing.assert_close(context_mask, attention_mask)


def test_neighbor_context_build_with_edges_and_neighbor_cap() -> None:
    from stoic.feature_pooling import NeighborContextSelfAttentionPooling

    pool = NeighborContextSelfAttentionPooling(
        emb_dim=6,
        output_dim=6,
        num_heads=2,
        max_context_neighbors=1,
    )
    node_features = torch.randn(4, 5, 6)
    attention_mask = torch.tensor(
        [
            [False, False, False, False, True],
            [False, False, False, True, True],
            [False, False, False, False, False],
            [False, True, True, True, True],
        ],
        dtype=torch.bool,
    )

    edge_index = torch.tensor(
        [[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]],
        dtype=torch.long,
    )
    context_features, context_mask = pool._build_neighbor_context(
        node_features=node_features,
        attention_mask=attention_mask,
        edge_index=edge_index,
    )
    assert context_features.shape == (4, 10, 6)
    assert context_mask.shape == (4, 10)
    torch.testing.assert_close(context_features[:, :5, :], node_features)


def test_neighbor_context_pool_output_contract_with_weights() -> None:
    from stoic.feature_pooling import NeighborContextSelfAttentionPooling

    pool = NeighborContextSelfAttentionPooling(
        emb_dim=8,
        output_dim=8,
        num_heads=2,
        return_weights=True,
        use_soft_pooling=True,
    )
    node_features = torch.randn(3, 6, 8)
    attention_mask = torch.tensor(
        [
            [False, False, False, True, True, True],
            [False, False, False, False, True, True],
            [False, False, False, False, False, False],
        ],
        dtype=torch.bool,
    )
    edge_index = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1]],
        dtype=torch.long,
    )
    pooled, average_weights = pool.pool_node_features(
        node_features=node_features,
        attention_mask=attention_mask,
        edge_index=edge_index,
    )
    assert pooled.shape == (3, 8)
    assert average_weights.shape == (3, 6)
    assert torch.max(average_weights[attention_mask]) < 1e-6


def test_neighbor_context_pool_hard_path_is_finite() -> None:
    from stoic.feature_pooling import NeighborContextSelfAttentionPooling

    pool = NeighborContextSelfAttentionPooling(
        emb_dim=8,
        output_dim=8,
        num_heads=2,
        return_weights=False,
        use_soft_pooling=False,
        threshold=1.0,
    )
    node_features = torch.randn(2, 5, 8)
    attention_mask = torch.zeros((2, 5), dtype=torch.bool)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    out = pool.pool_node_features(
        node_features=node_features,
        attention_mask=attention_mask,
        edge_index=edge_index,
    )
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()
