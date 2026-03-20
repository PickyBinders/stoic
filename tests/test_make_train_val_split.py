import sys
from pathlib import Path

import pandas as pd
import pytest

from stoic_train.make_train_val_split import _parse_ids, assign_split, main


def test_parse_ids_strips_and_drops_empty_values() -> None:
    assert _parse_ids("a; b ;; c ;") == ["a", "b", "c"]


def test_assign_split_returns_expected_labels() -> None:
    row_val = {"cluster_label": "x;y"}
    row_train = {"cluster_label": "a;b"}
    row_mixed = {"cluster_label": "a;x"}

    assert (
        assign_split(
            row=row_val,
            train_ids={"a", "b"},
            test_ids={"x", "y"},
            split_feature="cluster_label",
        )
        == "val"
    )
    assert (
        assign_split(
            row=row_train,
            train_ids={"a", "b"},
            test_ids={"x", "y"},
            split_feature="cluster_label",
        )
        == "train"
    )
    assert (
        assign_split(
            row=row_mixed,
            train_ids={"a", "b"},
            test_ids={"x", "y"},
            split_feature="cluster_label",
        )
        == "unassigned"
    )


def test_main_val_ratio_zero_keeps_train_rows_and_uses_default_split_feature(
    tmp_path: Path, monkeypatch
) -> None:
    data_file = tmp_path / "data.csv"
    df = pd.DataFrame(
        [
            {"pdb_id": "1", "split": "train", "cluster_label": "a"},
            {"pdb_id": "2", "split": "train", "cluster_label": "b;c"},
            {"pdb_id": "3", "split": "benchmark", "cluster_label": "x"},
        ]
    )
    df.to_csv(data_file, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_train_val_split",
            "--data-file",
            str(data_file),
            "--val-ratio",
            "0",
            "--seed",
            "1",
        ],
    )

    main()

    output_file = tmp_path / "data_train_val.csv"
    assert output_file.exists()
    out_df = pd.read_csv(output_file)

    assert out_df.loc[out_df["pdb_id"] == 1, "split"].item() == "train"
    assert out_df.loc[out_df["pdb_id"] == 2, "split"].item() == "train"
    assert out_df.loc[out_df["pdb_id"] == 3, "split"].item() == "benchmark"


def test_main_marks_mixed_ids_as_unassigned(tmp_path: Path, monkeypatch) -> None:
    data_file = tmp_path / "mixed.csv"
    df = pd.DataFrame(
        [
            {"pdb_id": "1", "split": "train", "cluster_label": "a"},
            {"pdb_id": "2", "split": "train", "cluster_label": "b"},
            {"pdb_id": "3", "split": "train", "cluster_label": "a;b"},
        ]
    )
    df.to_csv(data_file, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_train_val_split",
            "--data-file",
            str(data_file),
            "--val-ratio",
            "0.5",
            "--seed",
            "1",
            "--output-file",
            str(tmp_path / "out.csv"),
        ],
    )

    main()

    out_df = pd.read_csv(tmp_path / "out.csv")
    assert "unassigned" in set(out_df["split"])


def test_main_raises_for_missing_columns(tmp_path: Path, monkeypatch) -> None:
    no_split = tmp_path / "no_split.csv"
    pd.DataFrame([{"cluster_label": "a"}]).to_csv(no_split, index=False)
    monkeypatch.setattr(
        sys, "argv", ["make_train_val_split", "--data-file", str(no_split)]
    )
    with pytest.raises(KeyError, match="Missing split column"):
        main()

    no_feature = tmp_path / "no_feature.csv"
    pd.DataFrame([{"split": "train", "other": "a"}]).to_csv(no_feature, index=False)
    monkeypatch.setattr(
        sys, "argv", ["make_train_val_split", "--data-file", str(no_feature)]
    )
    with pytest.raises(KeyError, match="Missing split feature column"):
        main()


def test_main_raises_for_invalid_inputs(tmp_path: Path, monkeypatch) -> None:
    missing_file = tmp_path / "missing.csv"
    monkeypatch.setattr(
        sys, "argv", ["make_train_val_split", "--data-file", str(missing_file)]
    )
    with pytest.raises(FileNotFoundError):
        main()

    data_file = tmp_path / "data.csv"
    pd.DataFrame([{"split": "benchmark", "cluster_label": "x"}]).to_csv(
        data_file, index=False
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_train_val_split",
            "--data-file",
            str(data_file),
            "--val-ratio",
            "-0.1",
        ],
    )
    with pytest.raises(ValueError, match="--val-ratio must be in"):
        main()

    monkeypatch.setattr(
        sys, "argv", ["make_train_val_split", "--data-file", str(data_file)]
    )
    with pytest.raises(ValueError, match="No rows found"):
        main()


def test_main_raises_when_split_feature_has_no_ids(tmp_path: Path, monkeypatch) -> None:
    data_file = tmp_path / "empty_ids.csv"
    pd.DataFrame(
        [
            {"split": "train", "cluster_label": ""},
            {"split": "train", "cluster_label": " ; ; "},
        ]
    ).to_csv(data_file, index=False)

    monkeypatch.setattr(
        sys, "argv", ["make_train_val_split", "--data-file", str(data_file)]
    )
    with pytest.raises(ValueError, match="No IDs found"):
        main()
