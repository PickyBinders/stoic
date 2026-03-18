import torch
import torch.nn.functional as F

from stoic_train.losses import (
    ComplexProductLoss,
    FocalLoss,
    ResidueWeightFocalLoss,
    ResidueWeightKLLoss,
    ResidueWeightL1Loss,
    SparsityLoss,
)


def test_focal_loss_reduction_none_returns_per_sample_vector() -> None:
    inputs = torch.tensor([[2.0, -1.0], [-1.0, 2.0], [0.2, 0.1]], dtype=torch.float32)
    targets = torch.tensor([0, 1, 0], dtype=torch.long)
    loss_fn = FocalLoss(gamma=2.0, reduction="none")
    loss = loss_fn(inputs, targets)
    assert loss.shape == (3,)
    assert torch.isfinite(loss).all()


def test_focal_loss_reduction_mean_and_sum_return_scalars() -> None:
    inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)
    mean_loss = FocalLoss(gamma=2.0, reduction="mean")(inputs, targets)
    sum_loss = FocalLoss(gamma=2.0, reduction="sum")(inputs, targets)
    assert mean_loss.ndim == 0
    assert sum_loss.ndim == 0
    assert torch.isfinite(mean_loss)
    assert torch.isfinite(sum_loss)


def test_focal_loss_gamma_zero_matches_cross_entropy() -> None:
    inputs = torch.tensor([[2.5, -0.5], [0.3, 1.7], [1.0, -1.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1, 0], dtype=torch.long)
    focal = FocalLoss(gamma=0.0, reduction="mean")(inputs, targets)
    ce = torch.nn.functional.cross_entropy(inputs, targets, reduction="mean")
    torch.testing.assert_close(focal, ce)


def test_focal_loss_downweights_easy_examples_more_than_hard_ones() -> None:
    inputs = torch.tensor([[8.0, -8.0], [0.2, -0.2]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)
    ce_none = torch.nn.functional.cross_entropy(inputs, targets, reduction="none")
    focal_none = FocalLoss(gamma=2.0, reduction="none")(inputs, targets)

    ratio_easy = focal_none[0] / ce_none[0]
    ratio_hard = focal_none[1] / ce_none[1]
    assert ratio_easy < ratio_hard


def test_focal_loss_with_class_weights_changes_loss_value() -> None:
    inputs = torch.tensor([[0.4, 0.6], [0.7, 0.3], [0.3, 0.7]], dtype=torch.float32)
    targets = torch.tensor([1, 0, 1], dtype=torch.long)
    unweighted = FocalLoss(gamma=2.0, reduction="mean")(inputs, targets)
    class_weight = torch.tensor([1.0, 5.0], dtype=torch.float32)
    weighted = FocalLoss(weight=class_weight, gamma=2.0, reduction="mean")(inputs, targets)
    assert torch.isfinite(weighted)
    assert weighted.item() != unweighted.item()


def test_focal_loss_backward_produces_finite_gradients() -> None:
    inputs = torch.tensor(
        [[0.2, -0.1, 0.0], [1.4, -0.7, 0.1], [0.0, 0.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor([0, 2, 1], dtype=torch.long)
    loss = FocalLoss(gamma=2.0, reduction="mean")(inputs, targets)
    loss.backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_complex_product_loss_returns_scalar_and_is_finite() -> None:
    logits = torch.tensor(
        [[2.0, -1.0], [0.5, -0.2], [-0.1, 0.3], [0.0, 1.1]],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    complex_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    loss = ComplexProductLoss()(logits, targets, complex_id)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_complex_product_loss_matches_manual_grouped_logsum_formula() -> None:
    logits = torch.tensor(
        [[1.2, -0.3], [0.6, -0.1], [-0.4, 0.7], [-0.2, 1.5]],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    complex_id = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    loss_fn = ComplexProductLoss(temperature=2.0, margin=1.0, min_value=1e-6, max_value=30.0)

    got = loss_fn(logits, targets, complex_id)
    node_losses = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    c0 = torch.log(node_losses[complex_id == 0] + 1.0).sum() / 2.0
    c1 = torch.log(node_losses[complex_id == 1] + 1.0).sum() / 2.0
    expected = torch.stack([c0.clamp(1e-6, 30.0), c1.clamp(1e-6, 30.0)]).mean()
    torch.testing.assert_close(got, expected)


def test_complex_product_loss_accepts_non_tensor_complex_id() -> None:
    logits = torch.tensor([[1.0, -0.1], [0.2, 0.8], [0.3, 0.1]], dtype=torch.float32)
    targets = torch.tensor([0, 1, 0], dtype=torch.long)
    complex_id_list = [0, 1, 1]
    loss = ComplexProductLoss()(logits, targets, complex_id_list)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_complex_product_loss_clamps_to_configured_bounds() -> None:
    logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)
    complex_id = torch.tensor([0, 0], dtype=torch.long)

    low = ComplexProductLoss(min_value=0.5, max_value=10.0, margin=1e-9)
    high = ComplexProductLoss(min_value=1e-9, max_value=0.1, margin=1000.0)

    low_loss = low(logits, targets, complex_id)
    high_loss = high(logits, targets, complex_id)
    torch.testing.assert_close(low_loss, torch.tensor(0.5))
    torch.testing.assert_close(high_loss, torch.tensor(0.1))


def test_complex_product_loss_use_focal_with_gamma_zero_matches_plain() -> None:
    logits = torch.tensor([[1.2, -0.6], [-0.5, 0.3], [0.9, -0.1]], dtype=torch.float32)
    targets = torch.tensor([0, 1, 0], dtype=torch.long)
    complex_id = torch.tensor([0, 0, 1], dtype=torch.long)

    plain = ComplexProductLoss(use_focal=False, gamma=2.0)(logits, targets, complex_id)
    focal_gamma_zero = ComplexProductLoss(use_focal=True, gamma=0.0)(
        logits, targets, complex_id
    )
    torch.testing.assert_close(plain, focal_gamma_zero)


def test_sparsity_loss_without_mask_returns_expected_zero_loss() -> None:
    loss_fn = SparsityLoss(topk_percent=0.5, target_concentration=0.9)
    attention_weights = torch.tensor([[0.6, 0.4, 0.0, 0.0]], dtype=torch.float32)

    loss, stats = loss_fn(attention_weights, mask=None)

    torch.testing.assert_close(stats["concentration"], torch.tensor(1.0))
    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(stats["topk"], torch.tensor(0.0))


def test_sparsity_loss_mask_ignores_padded_positions() -> None:
    loss_fn = SparsityLoss(topk_percent=0.5, target_concentration=0.5)
    attention_weights = torch.tensor([[0.1, 0.2, 0.3, 0.9, 0.8]], dtype=torch.float32)
    mask = torch.tensor([[False, False, False, True, True]])

    masked_loss, masked_stats = loss_fn(attention_weights, mask=mask)
    unmasked_loss, _ = loss_fn(attention_weights, mask=None)

    torch.testing.assert_close(masked_stats["concentration"], torch.tensor(0.3))
    torch.testing.assert_close(masked_loss, torch.tensor(0.2))
    assert masked_loss > unmasked_loss


def test_sparsity_loss_uses_at_least_one_topk_element() -> None:
    loss_fn = SparsityLoss(topk_percent=0.01, target_concentration=0.4)
    attention_weights = torch.tensor(
        [[0.1, 0.3, 0.2], [0.5, 0.1, 0.4]],
        dtype=torch.float32,
    )

    loss, stats = loss_fn(attention_weights, mask=None)

    expected_concentration_mean = torch.tensor((0.3 + 0.5) / 2)
    torch.testing.assert_close(stats["concentration"], expected_concentration_mean)
    assert torch.isfinite(loss)


def test_residue_weight_l1_loss_matches_manual_l1_with_squeezed_targets() -> None:
    pred = torch.tensor([[0.2, 0.8], [0.5, 0.5]], dtype=torch.float32)
    target = torch.tensor([[[0.0], [1.0]], [[1.0], [0.0]]], dtype=torch.float32)
    loss = ResidueWeightL1Loss()(pred, target)
    expected = F.l1_loss(pred, target.squeeze(2))
    torch.testing.assert_close(loss, expected)


def test_residue_weight_kl_loss_matches_manual_formula() -> None:
    pred = torch.tensor([[0.6, 0.4], [0.3, 0.7]], dtype=torch.float32)
    target = torch.tensor([[[0.5], [0.5]], [[0.2], [0.8]]], dtype=torch.float32)
    loss = ResidueWeightKLLoss()(pred, target)
    expected = F.kl_div(torch.log(pred + 1e-9), target.squeeze(2), reduction="batchmean")
    torch.testing.assert_close(loss, expected)


def test_residue_weight_focal_loss_returns_scalar_and_is_finite() -> None:
    pred = torch.tensor([[0.9, 0.2, 0.7], [0.1, 0.8, 0.3]], dtype=torch.float32)
    target = torch.tensor(
        [[[1.0], [0.0], [1.0]], [[0.0], [1.0], [0.0]]],
        dtype=torch.float32,
    )
    loss = ResidueWeightFocalLoss(alpha=0.25, gamma=2.0)(pred, target)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_residue_weight_focal_loss_gamma_zero_matches_alpha_scaled_bce() -> None:
    pred = torch.tensor([[0.8, 0.3], [0.4, 0.9]], dtype=torch.float32)
    target = torch.tensor([[[1.0], [0.0]], [[0.0], [1.0]]], dtype=torch.float32)
    alpha = 0.7
    loss = ResidueWeightFocalLoss(alpha=alpha, gamma=0.0)(pred, target)
    bce = F.binary_cross_entropy(pred, target.squeeze(2), reduction="none").mean()
    torch.testing.assert_close(loss, alpha * bce)


