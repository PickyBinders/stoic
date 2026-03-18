from types import SimpleNamespace

import numpy as np
import torch

import stoic_train.lightning_model as lm
from stoic_train.losses import ComplexProductLoss, ResidueWeightFocalLoss, SparsityLoss
from stoic_train.lightning_model import StoichiometryModelLightning


class DummyStoic(torch.nn.Module):
    def __init__(
        self,
        stoichiometry_classes_to_use,
        seq_embed_model_name,
        seq_feature_encoder,
        feature_pooling_strategy,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = len(stoichiometry_classes_to_use)
        self.return_weights = kwargs.get("fps_return_weights", False)
        self.node_classifier = torch.nn.Linear(4, self.num_classes)
        self.feature_pooling_strategy = torch.nn.Linear(4, 4)
        self.seq_embed_model = SimpleNamespace(full_length_inference=False)

    def forward(self, inputs, edge_index, contacting_res_weight=None):
        if isinstance(inputs, list):
            batch = len(inputs)
            seq_len = max(len(s) for s in inputs) if inputs else 1
        else:
            batch = inputs.size(0)
            seq_len = inputs.size(1) if inputs.ndim >= 2 else 1

        node_scores = torch.randn(batch, self.num_classes, dtype=torch.float32)
        out = {
            "node_scores": node_scores,
            "attention_mask": torch.zeros(batch, seq_len, dtype=torch.bool),
        }
        if self.return_weights:
            out["residue_weights"] = torch.full((batch, seq_len), 0.5, dtype=torch.float32)
        return out


class DummyMetric:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class DummyMetricCollectionLifecycle:
    def __init__(self, compute_dict):
        self.compute_dict = compute_dict
        self.reset_calls = 0

    def compute(self):
        return self.compute_dict

    def reset(self):
        self.reset_calls += 1


def _build_batch():
    return SimpleNamespace(
        sequence=[["AAAA"], ["BB"]],
        quantity=torch.tensor([1, 99], dtype=torch.long),
        interacting_res=[[{"RES1": 1, "RES3": 1}], [{}]],
        complex_id=[[0], [1]],
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        sequence_embedding=torch.randn(2, 4, 4),
    )


def _build_model(monkeypatch, **kwargs) -> StoichiometryModelLightning:
    monkeypatch.setattr(lm, "Stoic", DummyStoic)
    monkeypatch.setattr(StoichiometryModelLightning, "_init_metrics", lambda self: None)
    return StoichiometryModelLightning(
        stoichiometry_classes_to_use=[1, 2, 3],
        seq_embed_model_name="esm2_t33_650M_UR50D",
        seq_feature_encoder="stoic.layers.Identity",
        feature_pooling_strategy="stoic.feature_pooling.AveragePooling",
        **kwargs,
    )


def test_init_configuration_and_class_resolution(monkeypatch) -> None:
    model = _build_model(monkeypatch, predict_unknown_classes=True)
    assert model.hf_model_name == "facebook/esm2_t33_650M_UR50D"
    assert model.seq_feature_encoder == "Identity"
    assert model.feature_pooling_strategy == "AveragePooling"
    assert model.UNKNOWN_CLASS in model.stoichiometry_classes_to_use.tolist()


def test_init_losses_instantiates_expected_modules(monkeypatch) -> None:
    model = _build_model(
        monkeypatch,
        loss=ComplexProductLoss,
        use_focal=True,
        use_sparsity_loss=True,
        use_residue_weight_loss=True,
        residue_weight_loss_type="ResidueWeightFocalLoss",
    )
    assert isinstance(model.loss, ComplexProductLoss)
    assert model.loss.use_focal is True
    assert isinstance(model.sparsity_loss, SparsityLoss)
    assert isinstance(model.residue_weight_loss, ResidueWeightFocalLoss)


def test_filter_classes_mask(monkeypatch) -> None:
    model = _build_model(monkeypatch)
    labels = torch.tensor([1, 5, 2, 9], dtype=torch.long)
    mask = model._filter_classes(labels)
    assert mask.tolist() == [True, False, True, False]


def test_calculate_contacting_res_weight_handles_empty_and_invalid_positions() -> None:
    weights = StoichiometryModelLightning._calculate_contacting_res_weight(
        sequences=["AAAA", "BBB"],
        interacting_res=[{"RES1": 1, "BAD": 1, "RES99": 1}, {}],
        seq_dim=4,
    )
    assert weights.shape == (2, 4, 1)
    assert weights[0, :, 0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert weights[1, :, 0].tolist() == [1.0, 1.0, 1.0, 0.0]


def test_get_seq_dim_respects_full_length_eval_mode(monkeypatch) -> None:
    model = _build_model(monkeypatch, max_seq_len=10)
    sequences = ["AAAA", "BB"]
    model.train()
    assert model._get_seq_dim(sequences) == 8
    model.model.seq_embed_model.full_length_inference = True
    model.eval()
    assert model._get_seq_dim(sequences) == 4


def test_process_batch_filters_unknown_classes_when_disabled(monkeypatch) -> None:
    model = _build_model(monkeypatch, predict_unknown_classes=False, fps_return_weights=True)
    batch = _build_batch()
    out = model._process_batch(batch=batch, batch_idx=0, split="train")
    assert out["node_scores"].shape[0] == 1
    assert out["node_labels"].tolist() == [0]
    assert out["complex_id"].tolist() == [0]
    assert out["attention_mask"].shape[0] == 1
    assert out["residue_weights"].shape[0] == 1


def test_process_batch_maps_unknown_to_unknown_class_when_enabled(monkeypatch) -> None:
    model = _build_model(monkeypatch, predict_unknown_classes=True)
    batch = _build_batch()
    out = model._process_batch(batch=batch, batch_idx=0, split="train")
    unknown_idx = model.classes_to_use_mapper[model.UNKNOWN_CLASS]
    assert out["node_scores"].shape[0] == 2
    assert out["node_labels"].shape[0] == 2
    assert unknown_idx in out["node_labels"].tolist()


def test_compute_losses_includes_auxiliary_terms(monkeypatch) -> None:
    model = _build_model(
        monkeypatch,
        use_sparsity_loss=True,
        use_residue_weight_loss=True,
        fps_return_weights=True,
    )
    step_output = {
        "node_scores": torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32),
        "node_labels": torch.tensor([0, 1], dtype=torch.long),
        "complex_id": torch.tensor([0, 1], dtype=torch.long),
        "residue_weights": torch.tensor([[0.8, 0.2], [0.6, 0.4]], dtype=torch.float32),
        "target_residue_weights": torch.tensor(
            [[[1.0], [0.0]], [[1.0], [0.0]]], dtype=torch.float32
        ),
        "attention_mask": torch.tensor([[False, False], [False, False]], dtype=torch.bool),
    }
    out = model._compute_losses(step_output)
    assert "loss" in out
    assert "sparsity_loss" in out
    assert "sparsity_loss_dict" in out
    assert "residue_weight_loss" in out


def test_shared_step_aggregates_losses_logs_and_updates_metrics(monkeypatch) -> None:
    model = _build_model(
        monkeypatch,
        use_sparsity_loss=True,
        use_residue_weight_loss=True,
        fps_return_weights=True,
    )
    logged = []
    model.log = lambda name, value, **kwargs: logged.append((name, float(value)))

    m1 = DummyMetric()
    m2 = DummyMetric()
    wd = DummyMetric()
    wo = DummyMetric()
    model._get_split_metrics = lambda split: (m1, m2, wd, wo)

    model._process_batch = lambda batch, batch_idx, split: {
        "node_scores": torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32),
        "node_labels": torch.tensor([0, 1], dtype=torch.long),
        "complex_id": torch.tensor([0, 1], dtype=torch.long),
        "residue_weights": torch.tensor([[0.8, 0.2], [0.6, 0.4]], dtype=torch.float32),
        "target_residue_weights": torch.tensor(
            [[[1.0], [0.0]], [[1.0], [0.0]]], dtype=torch.float32
        ),
        "attention_mask": torch.tensor([[False, False], [False, False]], dtype=torch.bool),
    }
    model._compute_losses = lambda step_output: {
        "loss": torch.tensor(1.0),
        "node_scores": step_output["node_scores"],
        "node_labels": step_output["node_labels"],
        "residue_weights": step_output["residue_weights"],
        "target_residue_weights": step_output["target_residue_weights"],
        "attention_mask": step_output["attention_mask"],
        "sparsity_loss": torch.tensor(0.5),
        "sparsity_loss_dict": {"topk": torch.tensor(0.1), "concentration": torch.tensor(0.9)},
        "residue_weight_loss": torch.tensor(0.25),
    }

    total_loss = model._shared_step(batch=object(), batch_idx=0, split="train")
    expected = (
        1.0 * model.stoichiometry_lambda
        + 0.5 * model.sparsity_lambda
        + 0.25 * model.residue_lambda
    )
    assert torch.isclose(total_loss, torch.tensor(expected))
    assert any(name == "train_node_loss" for name, _ in logged)
    assert any(name == "train_loss" for name, _ in logged)
    assert len(m1.calls) == 1
    assert len(m2.calls) == 1
    assert len(wd.calls) == 1
    assert len(wo.calls) == 1


def test_get_split_metrics_returns_expected_objects(monkeypatch) -> None:
    model = _build_model(monkeypatch, fps_return_weights=True)
    model.node_metrics_set1 = "train_m1"
    model.node_metrics_set2 = "train_m2"
    model.weight_distribution_metric = "train_wd"
    model.weight_overlap_metric = "train_wo"
    model.val_node_metrics_set1 = "val_m1"
    model.val_node_metrics_set2 = "val_m2"
    model.val_weight_distribution_metric = "val_wd"
    model.val_weight_overlap_metric = "val_wo"

    m1, m2, wd, wo = model._get_split_metrics("train")
    assert (m1, m2, wd, wo) == ("train_m1", "train_m2", "train_wd", "train_wo")
    m1, m2, wd, wo = model._get_split_metrics("val")
    assert (m1, m2, wd, wo) == ("val_m1", "val_m2", "val_wd", "val_wo")


def test_get_split_metrics_without_weight_metrics_returns_none(monkeypatch) -> None:
    model = _build_model(monkeypatch, fps_return_weights=False)
    model.node_metrics_set1 = "train_m1"
    model.node_metrics_set2 = "train_m2"
    m1, m2, wd, wo = model._get_split_metrics("train")
    assert (m1, m2) == ("train_m1", "train_m2")
    assert wd is None
    assert wo is None


def test_log_epoch_metrics_handles_all_metric_name_branches_and_resets(monkeypatch) -> None:
    model = _build_model(monkeypatch, fps_return_weights=True)
    logged = []
    cm_logged = []
    model.log = lambda name, value, **kwargs: logged.append((name, value, kwargs))
    monkeypatch.setattr(
        lm,
        "log_confusion_matrix_advanced",
        lambda *args, **kwargs: cm_logged.append((args, kwargs)),
    )

    model.test_node_metrics_set1 = DummyMetricCollectionLifecycle(
        {
            "test_node_metrics_set1AveragePrecision": torch.tensor(
                [0.2, 0.4, 0.6], dtype=torch.float32
            )
        }
    )
    model.test_node_metrics_set2 = DummyMetricCollectionLifecycle(
        {
            "test_node_metrics_set2ConfusionMatrix": torch.tensor(
                [[0.8, 0.2, 0.0], [0.1, 0.9, 0.0], [0.0, 0.3, 0.7]],
                dtype=torch.float32,
            ),
            "test_node_metrics_set2top1": torch.tensor(0.75),
        }
    )
    model.test_weight_distribution_metric = DummyMetricCollectionLifecycle({})
    model.test_weight_overlap_metric = DummyMetricCollectionLifecycle(
        {"test_weight_overlap_metric_overlap": torch.tensor(0.55)}
    )

    model._log_epoch_metrics("test")

    assert len(cm_logged) == 1
    assert any("ConfusionMatrix" in name for name, _, _ in logged)
    assert any("top1" in name for name, _, _ in logged)
    assert any("weight_overlap" in name for name, _, _ in logged)
    assert any("AveragePrecision" in name for name, _, _ in logged)

    assert model.test_node_metrics_set1.reset_calls == 1
    assert model.test_node_metrics_set2.reset_calls == 1
    assert model.test_weight_distribution_metric.reset_calls == 1
    assert model.test_weight_overlap_metric.reset_calls == 1


def test_epoch_end_hooks_delegate_to_log_epoch_metrics(monkeypatch) -> None:
    model = _build_model(monkeypatch)
    seen = []
    model._log_epoch_metrics = lambda split: seen.append(split)

    model.on_train_epoch_end()
    model.on_validation_epoch_end()
    model.on_test_epoch_end()

    assert seen == ["train", "val", "test"]


def test_on_load_checkpoint_drops_incompatible_loss_weight(monkeypatch) -> None:
    model = _build_model(monkeypatch)
    checkpoint = {"state_dict": {"loss.weight": torch.tensor([1.0]), "x": torch.tensor([2.0])}}
    model.on_load_checkpoint(checkpoint)
    assert "loss.weight" not in checkpoint["state_dict"]
    assert "x" in checkpoint["state_dict"]


def test_on_load_checkpoint_keeps_loss_weight_when_expected(monkeypatch) -> None:
    model = _build_model(monkeypatch)
    model.state_dict = lambda: {"loss.weight": torch.tensor([1.0])}
    checkpoint = {"state_dict": {"loss.weight": torch.tensor([1.0])}}
    model.on_load_checkpoint(checkpoint)
    assert "loss.weight" in checkpoint["state_dict"]


def test_enable_disable_full_length_inference_toggle_seq_embed_flags(monkeypatch) -> None:
    model = _build_model(monkeypatch)
    assert model.model.seq_embed_model.full_length_inference is False
    model.enable_full_length_inference(max_inference_seq_len=321)
    assert model.model.seq_embed_model.full_length_inference is True
    assert model.model.seq_embed_model.max_inference_seq_len == 321
    model.disable_full_length_inference()
    assert model.model.seq_embed_model.full_length_inference is False


def test_configure_optimizers_builds_expected_groups_and_scheduler(monkeypatch) -> None:
    model = _build_model(monkeypatch)
    model.trainer = SimpleNamespace(max_epochs=7)

    first_param = next(model.model.feature_pooling_strategy.parameters())
    first_param.requires_grad = False

    captured = {}

    class FakeAdamW:
        def __init__(self, param_groups, **kwargs):
            captured["param_groups"] = param_groups
            captured["optimizer_kwargs"] = kwargs
            self.param_groups = param_groups

    class FakeOneCycleLR:
        def __init__(self, optimizer, **kwargs):
            captured["scheduler_optimizer"] = optimizer
            captured["scheduler_kwargs"] = kwargs

    monkeypatch.setattr(lm.torch.optim, "AdamW", FakeAdamW)
    monkeypatch.setattr(lm.torch.optim.lr_scheduler, "OneCycleLR", FakeOneCycleLR)

    out = model.configure_optimizers()

    assert "optimizer" in out
    assert "lr_scheduler" in out
    assert out["lr_scheduler"]["interval"] == "epoch"

    assert len(captured["param_groups"]) == 2
    non_pooling, pooling = captured["param_groups"]
    assert non_pooling["lr"] == 5e-4
    assert pooling["lr"] == 5e-4
    assert all(p.requires_grad for p in non_pooling["params"])
    assert all(p.requires_grad for p in pooling["params"])
    assert len(pooling["params"]) >= 1

    assert captured["optimizer_kwargs"]["weight_decay"] == 0.01
    assert captured["scheduler_optimizer"] is out["optimizer"]
    assert captured["scheduler_kwargs"]["total_steps"] == 7
    assert captured["scheduler_kwargs"]["max_lr"] == [5e-3, 5e-4]
