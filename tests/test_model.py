from typing import List

import torch

import stoic.model as model_module
from stoic.model import Stoic


class DummySeqEmb(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        max_seq_len: int = 8,
        full_length_inference: bool = False,
        max_inference_seq_len=None,
        finetune: bool = False,
        load_in_4bit: bool = False,
        lora_r: int = 32,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.seq_embed_size = 6
        self.full_length_inference = full_length_inference
        self.max_inference_seq_len = max_inference_seq_len
        self.finetune = finetune
        self.load_in_4bit = load_in_4bit
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

    def forward(self, sequences: List[str]):
        batch = len(sequences)
        if self.full_length_inference:
            lengths = [len(seq) for seq in sequences]
            if self.max_inference_seq_len is not None:
                lengths = [min(l, self.max_inference_seq_len) for l in lengths]
            seq_len = max(lengths) if lengths else 0
        else:
            seq_len = self.max_seq_len - 2
            lengths = [min(len(seq), seq_len) for seq in sequences]

        emb = torch.zeros(batch, seq_len, self.seq_embed_size, dtype=torch.float32)
        mask = torch.ones(batch, seq_len, dtype=torch.bool)
        for row, valid_len in enumerate(lengths):
            emb[row, :valid_len, :] = float(row + 1)
            mask[row, :valid_len] = False
        return emb, mask


class DummyReturnWeightsPooling(torch.nn.Module):
    def __init__(
        self,
        emb_dim: int,
        output_dim: int,
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
        self.return_weights = return_weights

    def pool_node_features(
        self,
        node_features: torch.Tensor,
        contacting_res_weight: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        edge_index: torch.Tensor = None,
    ):
        pooled = node_features.mean(dim=1)
        weights = torch.full(
            (node_features.size(0), node_features.size(1)),
            0.5,
            dtype=node_features.dtype,
            device=node_features.device,
        )
        return pooled, weights


def _build_model(monkeypatch, **kwargs) -> Stoic:
    monkeypatch.setattr(model_module, "Esm2", DummySeqEmb)
    kwargs.setdefault("max_seq_len", 8)
    return Stoic(
        stoichiometry_classes_to_use=[1, 2, 3],
        seq_embed_model_name="facebook/esm2_t33_650M_UR50D",
        seq_feature_encoder="Identity",
        feature_pooling_strategy="AveragePooling",
        **kwargs,
    )


def test_configuration_defaults_work(monkeypatch) -> None:
    model = _build_model(monkeypatch)
    assert model.num_stoichiometry_classes == 3
    assert model._seq_feature_encoder_name == "Identity"
    assert model._feature_pooling_strategy_name == "AveragePooling"
    assert model.feature_pooling_strategy.output_dim == model.seq_embed_model.seq_embed_size


def test_configuration_kwargs_change_components(monkeypatch) -> None:
    model = _build_model(
        monkeypatch,
        max_seq_len=12,
        fps_output_dim=10,
        seq_feature_encoder_dropout=0.3,
        seq_feature_encoder_num_heads=2,
        prediction_head_reduction_factor=2,
        prediction_head_dropout=0.1,
        finetune_seq_embed_model=True,
        load_in_4bit=True,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.2,
    )
    assert model.seq_embed_model.max_seq_len == 12
    assert model.seq_embed_model.finetune is True
    assert model.seq_embed_model.load_in_4bit is True
    assert model.seq_embed_model.lora_r == 8
    assert model.feature_pooling_strategy.output_dim == 10
    assert model.seq_feature_encoder.dropout == 0.3
    assert model.seq_feature_encoder.num_heads == 2
    assert model.node_classifier[0].out_features == 5
    assert model.node_classifier[3].p == 0.1


def test_get_sequence_embeddings_default_shapes_and_mask_values(monkeypatch) -> None:
    model = _build_model(monkeypatch, max_seq_len=8)
    sequences = ["AAAA", "AA", "A"]
    emb, mask = model.get_sequence_embeddings(sequences)
    assert emb.shape == (3, 6, 6)
    assert mask.shape == (3, 6)
    assert mask[0].tolist() == [False, False, False, False, True, True]
    assert mask[1].tolist() == [False, False, True, True, True, True]
    assert mask[2].tolist() == [False, True, True, True, True, True]


def test_get_sequence_embeddings_full_length_eval_mode_chunks_and_pads(monkeypatch) -> None:
    model = _build_model(
        monkeypatch,
        max_seq_len=10,
        seq_embed_model_chunk_size=2,
    )
    model.enable_full_length_inference(max_inference_seq_len=5)
    model.eval()
    sequences = ["AAAAA", "AAA", "AA"]
    emb, mask = model.get_sequence_embeddings(sequences)
    assert emb.shape == (3, 5, 6)
    assert mask.shape == (3, 5)
    assert mask[0].tolist() == [False, False, False, False, False]
    assert mask[1].tolist() == [False, False, False, True, True]
    assert mask[2].tolist() == [False, False, True, True, True]


def test_forward_without_return_weights_has_expected_keys(monkeypatch) -> None:
    model = _build_model(monkeypatch, max_seq_len=8)
    sequences = ["AAAA", "AA"]
    edge_index = model.get_edge_index(sequences)
    out = model.forward(sequences, edge_index)
    assert set(out.keys()) == {"attention_mask", "node_scores"}
    assert out["attention_mask"].shape == (2, 6)
    assert out["node_scores"].shape == (2, 3)


def test_forward_with_return_weights_includes_residue_weights(monkeypatch) -> None:
    monkeypatch.setattr(model_module, "Esm2", DummySeqEmb)
    monkeypatch.setattr(
        model_module.feature_pooling,
        "SelfAttentionPooling",
        DummyReturnWeightsPooling,
    )
    model = Stoic(
        stoichiometry_classes_to_use=[1, 2, 3],
        seq_embed_model_name="facebook/esm2_t33_650M_UR50D",
        seq_feature_encoder="Identity",
        feature_pooling_strategy="SelfAttentionPooling",
        max_seq_len=8,
        fps_return_weights=True,
        fps_output_dim=6,
    )
    sequences = ["AAAA", "AA"]
    edge_index = model.get_edge_index(sequences)
    out = model.forward(sequences, edge_index)
    assert set(out.keys()) == {"attention_mask", "node_scores", "residue_weights"}
    assert out["residue_weights"].shape == (2, 6)


def test_get_edge_index_returns_fully_connected_graph(monkeypatch) -> None:
    model = _build_model(monkeypatch)
    edge_index = model.get_edge_index(["AAA", "BBB"])
    expected = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.long)
    torch.testing.assert_close(edge_index.cpu(), expected)


def test_enable_and_disable_full_length_inference_toggle_flags(monkeypatch) -> None:
    model = _build_model(monkeypatch)
    assert model.seq_embed_model.full_length_inference is False
    model.enable_full_length_inference(max_inference_seq_len=17)
    assert model.seq_embed_model.full_length_inference is True
    assert model.seq_embed_model.max_inference_seq_len == 17
    model.disable_full_length_inference()
    assert model.seq_embed_model.full_length_inference is False


def test_predict_stoichiometry_contract_with_residue_weights(monkeypatch) -> None:
    monkeypatch.setattr(model_module, "Esm2", DummySeqEmb)
    monkeypatch.setattr(
        model_module.feature_pooling,
        "SelfAttentionPooling",
        DummyReturnWeightsPooling,
    )
    warning_messages = []
    monkeypatch.setattr(
        model_module.logger,
        "warning",
        lambda msg: warning_messages.append(msg),
    )

    model = Stoic(
        stoichiometry_classes_to_use=[10, 20, 30],
        seq_embed_model_name="facebook/esm2_t33_650M_UR50D",
        seq_feature_encoder="Identity",
        feature_pooling_strategy="SelfAttentionPooling",
        max_seq_len=8,
        fps_return_weights=True,
        fps_output_dim=6,
    )
    results, residue_predictions = model.predict_stoichiometry(
        ["AAA", "AAA", "BBB"],
        top_n=2,
        return_residue_weights=True,
    )

    assert len(results) == 2
    assert any("Duplicated sequences" in msg for msg in warning_messages)
    assert {"rank", "probability"}.issubset(results[0].keys())
    assert set(results[0].keys()) >= {"AAA", "BBB", "rank", "probability"}
    assert set(residue_predictions.keys()) == {
        "sequences",
        "pred_residues",
        "attention_mask",
    }
    assert len(residue_predictions["sequences"]) == 2
