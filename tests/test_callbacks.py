import math
import types

import numpy as np
import pytest
import torch

import stoic_train.callbacks as cb
from stoic_train.callbacks import (
    ResamplingCallback,
    SetupWandB,
    StoichiometryModelClassWeights,
)


class DummyLoss:
    def __init__(self):
        self.weight = None


class DummyInnerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))


class DummyPLModule:
    def __init__(self, classes_to_use):
        self.stoichiometry_classes_to_use = torch.tensor(classes_to_use, dtype=torch.long)
        self.node_class_weights = None
        self.loss = DummyLoss()
        self.model = DummyInnerModel()
        self.dtype = torch.float32


class DummyTrainLoaderDataset:
    def __init__(self, class_counts):
        self.stoichiometry_classes_counts = class_counts


class DummyTrainLoader:
    def __init__(self, class_counts):
        self.dataset = DummyTrainLoaderDataset(class_counts)


def _make_trainer(class_counts, is_global_zero=True):
    return types.SimpleNamespace(
        is_global_zero=is_global_zero,
        train_dataloader=DummyTrainLoader(class_counts),
    )


def test_setup_wandb_calls_watch_only_on_global_zero(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(cb.wandb, "watch", lambda *args, **kwargs: calls.append((args, kwargs)))
    callback = SetupWandB()
    module = DummyPLModule([1, 2, 3])

    callback.on_train_start(_make_trainer({"stoichiometry_class": np.array([1]), "count": np.array([1])}, is_global_zero=True), module)
    callback.on_train_start(_make_trainer({"stoichiometry_class": np.array([1]), "count": np.array([1])}, is_global_zero=False), module)

    assert len(calls) == 1


def test_calculate_class_weights_supports_all_methods_and_normalization() -> None:
    class_counts = {
        "stoichiometry_class": np.array([1, 2, 4]),
        "count": np.array([100, 10, 1]),
    }
    methods = ["inverse", "inverse_sqrt", "inverse_log", "effective_samples"]
    for method in methods:
        weights = StoichiometryModelClassWeights._calculate_class_weights(
            class_counts=class_counts,
            method=method,
        )
        assert set(weights.keys()) == {1, 2, 4}
        assert all(v > 0 for v in weights.values())
        assert math.isclose(sum(weights.values()), 3.0, rel_tol=1e-6, abs_tol=1e-6)


def test_calculate_class_weights_filters_classes_to_use() -> None:
    class_counts = {
        "stoichiometry_class": np.array([1, 2, 4]),
        "count": np.array([100, 10, 1]),
    }
    weights = StoichiometryModelClassWeights._calculate_class_weights(
        class_counts=class_counts,
        method="inverse",
        classes_to_use=[2, 4],
    )
    assert set(weights.keys()) == {2, 4}


def test_calculate_class_weights_raises_on_unknown_method() -> None:
    class_counts = {
        "stoichiometry_class": np.array([1, 2]),
        "count": np.array([10, 20]),
    }
    with pytest.raises(ValueError, match="Unknown weighting method"):
        StoichiometryModelClassWeights._calculate_class_weights(
            class_counts=class_counts,
            method="bad_method",
        )


def test_calculate_class_weights_raises_when_filter_removes_all_classes() -> None:
    class_counts = {
        "stoichiometry_class": np.array([1, 2]),
        "count": np.array([10, 20]),
    }
    with pytest.raises(ValueError, match="No classes left"):
        StoichiometryModelClassWeights._calculate_class_weights(
            class_counts=class_counts,
            method="inverse",
            classes_to_use=[999],
        )


def test_class_weights_on_train_start_sets_node_and_loss_weights() -> None:
    class_counts = {
        "stoichiometry_class": np.array([1, 2, 4]),
        "count": np.array([100, 10, 1]),
    }
    trainer = _make_trainer(class_counts)
    module = DummyPLModule([1, 2, 4])
    callback = StoichiometryModelClassWeights(method="inverse")

    callback.on_train_start(trainer, module)

    assert module.node_class_weights is not None
    assert module.node_class_weights.dtype == module.dtype
    assert module.loss.weight is not None
    assert module.loss.weight.dtype == next(module.model.parameters()).dtype
    assert module.loss.weight.device == next(module.model.parameters()).device
    assert module.loss.weight.shape[0] == 3


def test_class_weights_on_train_start_skips_when_already_matching_length(monkeypatch) -> None:
    class_counts = {
        "stoichiometry_class": np.array([1, 2, 4]),
        "count": np.array([100, 10, 1]),
    }
    trainer = _make_trainer(class_counts)
    module = DummyPLModule([1, 2, 4])
    module.node_class_weights = torch.tensor([1.0, 1.0, 1.0])
    callback = StoichiometryModelClassWeights(method="inverse")

    calls = {"n": 0}

    def fake_calc(*args, **kwargs):
        calls["n"] += 1
        return torch.tensor([1.0, 1.0, 1.0])

    monkeypatch.setattr(callback, "_calculate_stoichiometry_weights", fake_calc)
    callback.on_train_start(trainer, module)

    assert calls["n"] == 0


def test_resampling_callback_calls_datamodule_resample_once() -> None:
    called = {"n": 0}

    class DummyDataModule:
        def _resample_training_data(self):
            called["n"] += 1

    trainer = types.SimpleNamespace(datamodule=DummyDataModule())
    callback = ResamplingCallback()
    callback.on_train_epoch_end(trainer, pl_module=object())
    assert called["n"] == 1
