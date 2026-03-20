import json
import pickle
import sys
from pathlib import Path

import torch

import stoic.predict_stoichiometry as ps


class DummyStoicModel:
    def __init__(self):
        self.device = None
        self.eval_called = False
        self.enabled_full_length = None
        self.predict_calls = []

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def enable_full_length_inference(self, max_inference_seq_len):
        self.enabled_full_length = max_inference_seq_len

    def predict_stoichiometry(self, sequences, top_n=3, return_residue_weights=False):
        self.predict_calls.append(
            {
                "sequences": list(sequences),
                "top_n": top_n,
                "return_residue_weights": return_residue_weights,
            }
        )
        if top_n == 1:
            return [{"AAA": 2, "BBB": 1, "rank": 0.1, "probability": 0.9}]
        return [
            {"AAA": 2, "BBB": 1, "rank": 0.1, "probability": 0.9},
            {"AAA": 1, "BBB": 1, "rank": 0.2, "probability": 0.5},
        ]


def _patch_from_pretrained(monkeypatch, model):
    class DummyStoicClass:
        @staticmethod
        def from_pretrained(model_name):
            return model

    monkeypatch.setattr(ps, "Stoic", DummyStoicClass)


def test_predict_stoichiometry_uses_cpu_by_default_when_no_cuda(monkeypatch) -> None:
    model = DummyStoicModel()
    _patch_from_pretrained(monkeypatch, model)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    results = ps.predict_stoichiometry(["AAA", "BBB"], model_name="dummy", top_n=1)

    assert len(results) == 1
    assert model.device == torch.device("cpu")
    assert model.eval_called is True
    assert model.predict_calls[0]["sequences"] == ["AAA", "BBB"]
    assert model.predict_calls[0]["top_n"] == 1


def test_predict_stoichiometry_enables_full_length_when_requested(monkeypatch) -> None:
    model = DummyStoicModel()
    _patch_from_pretrained(monkeypatch, model)

    _ = ps.predict_stoichiometry(
        ["AAA", "BBB"],
        model_name="dummy",
        top_n=1,
        max_inference_seq_len=777,
    )

    assert model.enabled_full_length == 777


def test_predict_stoichiometry_accepts_single_fasta_file(monkeypatch, tmp_path: Path) -> None:
    model = DummyStoicModel()
    _patch_from_pretrained(monkeypatch, model)

    fasta_path = tmp_path / "complex.fasta"
    fasta_path.write_text(">a\nAAAA\n>b\nBBB\n")

    results = ps.predict_stoichiometry(str(fasta_path), model_name="dummy", top_n=1)

    assert len(results) == 1
    assert model.predict_calls[0]["sequences"] == ["AAAA", "BBB"]


def test_predict_stoichiometry_accepts_fasta_directory(monkeypatch, tmp_path: Path) -> None:
    model = DummyStoicModel()
    _patch_from_pretrained(monkeypatch, model)

    (tmp_path / "c1.fasta").write_text(">a\nAAAA\n>b\nBBB\n")
    (tmp_path / "c2.fa").write_text(">x\nCCCC\n>y\nDD\n")

    results = ps.predict_stoichiometry(str(tmp_path), model_name="dummy", top_n=1)

    assert isinstance(results, dict)
    assert set(results.keys()) == {"c1", "c2"}
    assert model.predict_calls[0]["sequences"] == ["AAAA", "BBB"]
    assert model.predict_calls[1]["sequences"] == ["CCCC", "DD"]


def test_main_prints_candidates_and_sequence_copy_numbers(monkeypatch, capsys) -> None:
    model = DummyStoicModel()
    _patch_from_pretrained(monkeypatch, model)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stoic_predict_stoichiometry",
            "--sequences",
            "AAA",
            "BBB",
            "--model",
            "dummy",
            "--top-n",
            "1",
            "--device",
            "cpu",
        ],
    )

    ps.main()
    out = capsys.readouterr().out
    assert "Candidate 1:" in out
    assert "AAA" in out
    assert "BBB" in out


def test_main_output_dir_saves_results_and_residue_weights(monkeypatch, tmp_path: Path) -> None:
    class DummyWithResidues(DummyStoicModel):
        def predict_stoichiometry(self, sequences, top_n=3, return_residue_weights=False):
            return (
                [{"AAA": 2, "BBB": 1, "rank": 0.1, "probability": 0.9}],
                {
                    "sequences": list(sequences),
                    "pred_residues": [[0.1, 0.9]],
                    "attention_mask": [[False, False]],
                },
            )

    model = DummyWithResidues()
    _patch_from_pretrained(monkeypatch, model)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stoic_predict_stoichiometry",
            "--sequences",
            "AAA",
            "BBB",
            "--model",
            "dummy",
            "--top-n",
            "1",
            "--return-residue-weights",
            "--output-dir",
            str(tmp_path),
            "--device",
            "cpu",
        ],
    )

    ps.main()

    results_path = tmp_path / "results.json"
    residues_path = tmp_path / "residue_predictions.pkl"
    af3_path = tmp_path / "af3_input_1.json"
    assert results_path.exists()
    assert residues_path.exists()
    assert af3_path.exists()

    with open(results_path, "r") as f:
        saved_results = json.load(f)
    assert isinstance(saved_results, list)
    assert saved_results[0]["AAA"] == 2

    with open(residues_path, "rb") as f:
        saved_residues = pickle.load(f)
    assert saved_residues["sequences"] == ["AAA", "BBB"]

    with open(af3_path, "r") as f:
        af3_json = json.load(f)
    assert af3_json["name"] == "input_sequences"
    assert "modelSeeds" in af3_json and len(af3_json["modelSeeds"]) == 1
    assert len(af3_json["sequences"]) >= 1

    results_path.unlink()
    residues_path.unlink()
    af3_path.unlink()
    assert not results_path.exists()
    assert not residues_path.exists()
    assert not af3_path.exists()


def test_main_directory_input_saves_separate_results_per_complex(
    monkeypatch, tmp_path: Path
) -> None:
    model = DummyStoicModel()
    _patch_from_pretrained(monkeypatch, model)

    fasta_dir = tmp_path / "fastas"
    fasta_dir.mkdir()
    (fasta_dir / "complex_a.fasta").write_text(">a\nAAAA\n>b\nBBB\n")
    (fasta_dir / "complex_b.fasta").write_text(">x\nCCCC\n")
    output_dir = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stoic_predict_stoichiometry",
            "--input-path",
            str(fasta_dir),
            "--model",
            "dummy",
            "--top-n",
            "1",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ],
    )
    ps.main()

    a_path = output_dir / "complex_a.json"
    b_path = output_dir / "complex_b.json"
    a_af3 = output_dir / "complex_a_af3_input_1.json"
    b_af3 = output_dir / "complex_b_af3_input_1.json"
    assert a_path.exists()
    assert b_path.exists()
    assert a_af3.exists()
    assert b_af3.exists()

    with open(a_path, "r") as f:
        a_json = json.load(f)
    assert isinstance(a_json, list)

    a_path.unlink()
    b_path.unlink()
    a_af3.unlink()
    b_af3.unlink()


def test_main_output_dir_saves_af3_json_for_each_candidate(
    monkeypatch, tmp_path: Path
) -> None:
    model = DummyStoicModel()
    _patch_from_pretrained(monkeypatch, model)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stoic_predict_stoichiometry",
            "--sequences",
            "AAA",
            "BBB",
            "--model",
            "dummy",
            "--top-n",
            "2",
            "--output-dir",
            str(tmp_path),
            "--device",
            "cpu",
        ],
    )

    ps.main()

    assert (tmp_path / "af3_input_1.json").exists()
    assert (tmp_path / "af3_input_2.json").exists()
