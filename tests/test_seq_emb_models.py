import types

import torch

import stoic.seq_emb_models as sem
from stoic.seq_emb_models import Esm2


class DummyHFModel(torch.nn.Module):
    def __init__(self, hidden_size: int = 7):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden_size)
        self.device = torch.device("cpu")
        self.eval_called = False
        self.dummy_param = torch.nn.Parameter(torch.tensor(1.0))

    def eval(self):
        self.eval_called = True
        return super().eval()

    def forward(self, input_ids, attention_mask=None):
        batch, seq_len = input_ids.shape
        hidden = self.config.hidden_size
        last_hidden_state = torch.arange(
            batch * seq_len * hidden, dtype=torch.float32
        ).reshape(batch, seq_len, hidden)
        return types.SimpleNamespace(last_hidden_state=last_hidden_state)


class DummyTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        sequences,
        return_tensors="pt",
        max_length=None,
        truncation=True,
        padding="max_length",
    ):
        self.calls.append(
            {
                "max_length": max_length,
                "padding": padding,
                "truncation": truncation,
                "sequences": list(sequences),
            }
        )
        batch = len(sequences)
        if padding == "max_length":
            seq_len = max_length
            effective_lengths = [min(len(seq) + 2, max_length) for seq in sequences]
        else:
            effective_lengths = [min(len(seq) + 2, max_length) for seq in sequences]
            seq_len = max(effective_lengths)

        input_ids = torch.zeros(batch, seq_len, dtype=torch.long)
        attention_mask = torch.zeros(batch, seq_len, dtype=torch.long)
        for i, valid in enumerate(effective_lengths):
            attention_mask[i, :valid] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _patch_hf(monkeypatch):
    model_calls = []
    tokenizer = DummyTokenizer()
    prepare_called = {"value": False}
    peft_called = {"value": False, "config": None}

    def fake_from_pretrained(model_name, **kwargs):
        model_calls.append({"model_name": model_name, "kwargs": kwargs})
        return DummyHFModel(hidden_size=7)

    def fake_tokenizer_from_pretrained(model_name):
        return tokenizer

    def fake_prepare_model_for_kbit_training(model):
        prepare_called["value"] = True
        return model

    def fake_get_peft_model(model, config):
        peft_called["value"] = True
        peft_called["config"] = config
        return model

    monkeypatch.setattr(sem.AutoModel, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(sem.AutoTokenizer, "from_pretrained", fake_tokenizer_from_pretrained)
    monkeypatch.setattr(sem, "prepare_model_for_kbit_training", fake_prepare_model_for_kbit_training)
    monkeypatch.setattr(sem, "get_peft_model", fake_get_peft_model)

    return model_calls, tokenizer, prepare_called, peft_called


def test_esm2_default_configuration_freezes_backbone(monkeypatch) -> None:
    model_calls, _, prepare_called, peft_called = _patch_hf(monkeypatch)
    model = Esm2(model_name="dummy-esm2", max_seq_len=10, finetune=False, load_in_4bit=False)

    assert model.seq_embed_size == 7
    assert model.model.eval_called is True
    assert all(not p.requires_grad for p in model.model.parameters())
    assert prepare_called["value"] is False
    assert peft_called["value"] is False
    assert model_calls[0]["kwargs"]["device_map"] == "auto"
    assert model_calls[0]["kwargs"]["add_pooling_layer"] is False


def test_esm2_4bit_configuration_uses_quantization_and_prepare(monkeypatch) -> None:
    model_calls, _, prepare_called, _ = _patch_hf(monkeypatch)
    _ = Esm2(model_name="dummy-esm2", load_in_4bit=True, finetune=False)

    assert prepare_called["value"] is True
    kwargs = model_calls[0]["kwargs"]
    assert kwargs["device_map"] == "auto"
    assert kwargs["_fast_init"] is False
    assert kwargs["add_pooling_layer"] is False
    assert kwargs.get("quantization_config") is not None


def test_esm2_finetune_path_applies_lora(monkeypatch) -> None:
    _, _, _, peft_called = _patch_hf(monkeypatch)
    _ = Esm2(
        model_name="dummy-esm2",
        finetune=True,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.2,
    )

    assert peft_called["value"] is True
    assert peft_called["config"].r == 8
    assert peft_called["config"].lora_alpha == 32
    assert peft_called["config"].lora_dropout == 0.2


def test_get_inference_max_length_caps_by_max_inference_seq_len(monkeypatch) -> None:
    _patch_hf(monkeypatch)
    model = Esm2(model_name="dummy-esm2", max_inference_seq_len=12)
    assert model._get_inference_max_length(["A", "AAAAA"]) == 7
    assert model._get_inference_max_length(["A" * 20]) == 12


def test_forward_removes_cls_eos_and_builds_mask_from_sequence_lengths(monkeypatch) -> None:
    _, tokenizer, _, _ = _patch_hf(monkeypatch)
    model = Esm2(model_name="dummy-esm2", max_seq_len=8, finetune=False)

    embeddings, attention_mask = model(["AAAA", "AAAAAAA"])

    assert embeddings.shape == (2, 6, 7)
    assert attention_mask.shape == (2, 6)
    assert attention_mask[0].tolist() == [False, False, False, False, True, True]
    assert attention_mask[1].tolist() == [False, False, False, False, False, False]
    assert tokenizer.calls[-1]["max_length"] == 8
    assert tokenizer.calls[-1]["padding"] == "max_length"


def test_forward_full_length_eval_uses_dynamic_length_and_correct_mask(monkeypatch) -> None:
    _, tokenizer, _, _ = _patch_hf(monkeypatch)
    model = Esm2(
        model_name="dummy-esm2",
        full_length_inference=True,
        max_inference_seq_len=6,
        finetune=False,
    )
    model.eval()

    embeddings, attention_mask = model(["AA", "AAAAAAAAAA"])

    assert embeddings.shape == (2, 4, 7)
    assert attention_mask[0].tolist() == [False, False, True, True]
    assert attention_mask[1].tolist() == [False, False, False, False]
    assert tokenizer.calls[-1]["max_length"] == 6
    assert tokenizer.calls[-1]["padding"] == "longest"
