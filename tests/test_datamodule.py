import pandas as pd
import pytest
import torch

import stoic_train.dataset as dataset_module
from stoic_train.dataset import StoichiometryDataModule


class DummyGraph:
    def __init__(self, quantity_value: int, interact_value: int):
        self.quantity = torch.tensor([quantity_value], dtype=torch.long)
        self.interact = torch.tensor([interact_value], dtype=torch.long)


class DummySubset:
    def __init__(self, graphs, data_df):
        self.graphs = graphs
        self.data_df = data_df

    def __iter__(self):
        return iter(self.graphs)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


class DummyStoichiometryDataset:
    def __init__(
        self,
        root,
        data_df,
        precomputed_embedding_path=None,
        transform=None,
        pre_transform=None,
    ):
        self.root = root
        self.data_df = data_df.reset_index(drop=True)
        self.graphs = []
        for _, row in self.data_df.iterrows():
            q = int(str(row["quantity"]).split(":")[0])
            interact = 1 if q > 1 else 0
            self.graphs.append(DummyGraph(quantity_value=q, interact_value=interact))

    def __getitem__(self, idx):
        if isinstance(idx, list):
            subset_graphs = [self.graphs[i] for i in idx]
            subset_df = self.data_df.iloc[idx].reset_index(drop=True)
            return DummySubset(subset_graphs, subset_df)
        return self.graphs[idx]


def _make_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"pdb_id": "A", "split": "train", "num_subunits": 1, "cluster_label": "c1", "quantity": "1"},
            {"pdb_id": "B", "split": "train", "num_subunits": 2, "cluster_label": "c1", "quantity": "2"},
            {"pdb_id": "C", "split": "train", "num_subunits": 3, "cluster_label": "c2", "quantity": "3"},
            {"pdb_id": "D", "split": "train", "num_subunits": 2, "cluster_label": "c2", "quantity": "2:1"},

            {"pdb_id": "E", "split": "val", "num_subunits": 2, "cluster_label": "v1", "quantity": "2"},
            {"pdb_id": "F", "split": "val", "num_subunits": 3, "cluster_label": "v2", "quantity": "3"},

            {"pdb_id": "G", "split": "benchmark", "num_subunits": 2, "cluster_label": "b1", "quantity": "2"},
            {"pdb_id": "H", "split": "benchmark", "num_subunits": 2, "cluster_label": "b1", "quantity": "2"},
            {"pdb_id": "I", "split": "benchmark", "num_subunits": 4, "cluster_label": "b2", "quantity": "4"},
        ]
    )


def _build_datamodule(monkeypatch, sample_training_data: bool = False) -> StoichiometryDataModule:
    monkeypatch.setattr(dataset_module.pd, "read_csv", lambda *args, **kwargs: _make_dataframe())
    dm = StoichiometryDataModule(
        root="/tmp/unused",
        data_file="/tmp/unused.csv",
        batch_size=4,
        max_num_nodes=10,
        max_num_edges=20,
        sample_training_data=sample_training_data,
        num_workers=0,
    )
    return dm


def test_filter_data_df_keeps_expected_rows(monkeypatch) -> None:
    dm = _build_datamodule(monkeypatch, sample_training_data=False)
    df = dm.data_df

    train_df = df.query("split == 'train'")
    assert (train_df["num_subunits"] > 1).all()
    benchmark_df = df.query("split == 'benchmark'")
    assert len(benchmark_df[["cluster_label", "quantity"]].drop_duplicates()) == len(
        benchmark_df
    )
    assert set(df["split"].unique()).issubset({"train", "val", "benchmark"})


def test_sample_training_data_branch_runs(monkeypatch) -> None:
    monkeypatch.setattr(dataset_module.pd, "read_csv", lambda *args, **kwargs: _make_dataframe())
    monkeypatch.setattr(pd.DataFrame, "sample", lambda self, frac=1.0: self)
    dm = StoichiometryDataModule(
        root="/tmp/unused",
        data_file="/tmp/unused.csv",
        sample_training_data=True,
        num_workers=0,
    )
    assert not dm.data_df.empty


def test_setup_builds_splits_and_attaches_class_counts(monkeypatch) -> None:
    dm = _build_datamodule(monkeypatch, sample_training_data=False)
    monkeypatch.setattr(dataset_module, "StoichiometryDataset", DummyStoichiometryDataset)

    dm.setup(stage="fit")

    assert hasattr(dm, "train")
    assert hasattr(dm, "val")
    assert hasattr(dm, "test")
    assert len(dm.train) >= 1
    assert len(dm.val) >= 1
    assert len(dm.test) >= 1
    assert hasattr(dm.train, "stoichiometry_classes_counts")
    assert hasattr(dm.train, "interaction_classes_counts")


def test_sample_training_data_returns_train_indices(monkeypatch) -> None:
    dm = _build_datamodule(monkeypatch, sample_training_data=False)
    train_idx = dm._sample_training_data()
    assert len(train_idx) >= 1
    train_df = dm.data_df.query("split == 'train'").reset_index(drop=True)
    assert all(isinstance(i, int) for i in train_idx)
    assert all(0 <= i < len(train_df) for i in train_idx)


def test_get_class_counts_returns_expected_structure(monkeypatch) -> None:
    dm = _build_datamodule(monkeypatch, sample_training_data=False)
    dm.train = DummySubset(
        graphs=[DummyGraph(2, 1), DummyGraph(3, 0), DummyGraph(2, 1)],
        data_df=pd.DataFrame(),
    )
    counts = dm._get_class_counts()
    assert set(counts.keys()) == {
        "stoichiometry_classes_counts",
        "interaction_classes_counts",
    }
    assert set(counts["stoichiometry_classes_counts"].keys()) == {"stoichiometry_class", "count"}
    assert set(counts["interaction_classes_counts"].keys()) == {"interaction_class", "count"}


def test_resample_training_data_refreshes_train_subset(monkeypatch) -> None:
    dm = _build_datamodule(monkeypatch, sample_training_data=False)
    monkeypatch.setattr(dataset_module, "StoichiometryDataset", DummyStoichiometryDataset)
    dm.setup(stage="fit")
    old_len = len(dm.train)

    monkeypatch.setattr(dm, "_sample_training_data", lambda: dm.train_idx[:1])
    dm._resample_training_data()

    assert len(dm.train) == 1
    assert len(dm.train) <= old_len


def test_train_val_test_dataloaders_wire_sampler_arguments(monkeypatch) -> None:
    dm = _build_datamodule(monkeypatch, sample_training_data=False)
    dm.train = DummySubset([DummyGraph(2, 1)], pd.DataFrame())
    dm.val = DummySubset([DummyGraph(2, 1)], pd.DataFrame())
    dm.test = DummySubset([DummyGraph(2, 1)], pd.DataFrame())

    sampler_calls = []
    dataloader_calls = []

    class CapturingSampler:
        def __init__(self, **kwargs):
            sampler_calls.append(kwargs)

    class CapturingLoader:
        def __init__(self, dataset, batch_sampler, num_workers):
            dataloader_calls.append(
                {
                    "dataset": dataset,
                    "batch_sampler": batch_sampler,
                    "num_workers": num_workers,
                }
            )

    monkeypatch.setattr(dataset_module, "DistributedDynamicBatchSampler", CapturingSampler)
    monkeypatch.setattr(dataset_module, "DataLoader", CapturingLoader)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")

    dm.train_dataloader()
    dm.val_dataloader()
    dm.test_dataloader()

    assert len(sampler_calls) == 3
    assert sampler_calls[0]["shuffle"] is True
    assert sampler_calls[1]["shuffle"] is False
    assert sampler_calls[2]["shuffle"] is False
    assert all(call["drop_last"] is True for call in sampler_calls)
    assert all(call["num_replicas"] == 2 for call in sampler_calls)
    assert all(call["rank"] == 1 for call in sampler_calls)
    assert all(call["batch_size"] == dm.batch_size for call in sampler_calls)
    assert all(call["max_num_nodes"] == dm.max_num_nodes for call in sampler_calls)
    assert all(call["max_num_edges"] == dm.max_num_edges for call in sampler_calls)
    assert len(dataloader_calls) == 3
    assert all(call["num_workers"] == 0 for call in dataloader_calls)
