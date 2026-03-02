import copy
import os
import pickle
from pathlib import Path
from typing import Optional

import lightning
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm

import graphein

graphein.verbose(enabled=False)

from stoic_train.samplers import DistributedDynamicBatchSampler


class StoichiometryDataset(Dataset):
    def __init__(
        self,
        root,
        data_df,
        precomputed_embedding_path: Optional[os.PathLike] = None,
        transform=None,
        pre_transform=None,
    ):
        self.convertor = from_networkx
        self.root = root
        self.data_df = data_df
        self.graph_names = self.get_graph_names()

        self.precomputed_embeddings = None
        if precomputed_embedding_path is not None:
            with open(precomputed_embedding_path, "rb") as f:
                self.precomputed_embeddings = pickle.load(f)

        super().__init__(root, transform, pre_transform)
        self.graphs = self.load_data()

    def get_graph_names(self):
        return (self.data_df["pdb_id"].str.lower() + "_graph").tolist()

    @property
    def raw_file_names(self):
        return [Path(self.raw_dir) / Path(f"{f}.pkl").name for f in self.graph_names]

    @property
    def processed_file_names(self):
        return [
            Path(self.processed_dir) / Path(f"{f}.pt").name for f in self.graph_names
        ]

    def download(self):
        pass

    def process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        for graph in tqdm(self.graph_names):
            processed_gr_path = Path(self.processed_dir) / Path(f"{graph}.pt").name
            if not processed_gr_path.exists():
                with open(Path(self.raw_dir) / Path(f"{graph}.pkl"), "rb") as f:
                    graphein_graph = pickle.load(f)
                torch_data = self.convertor(graphein_graph)
                torch.save(torch_data, processed_gr_path)

    def len(self):
        return len(self.graph_names)

    def load_data(self):
        graphs = []
        for i, graph in enumerate(tqdm(self.processed_file_names)):
            torch_data = torch.load(graph, weights_only=False)
            torch_data.complex_id = [i for _ in range(len(torch_data.sequence))]
            graphs.append(torch_data)

        if self.precomputed_embeddings is not None:
            for graph in graphs:
                sequence_embedding = torch.cat(
                    [
                        self.precomputed_embeddings[seq].unsqueeze(0)
                        for seq in graph.sequence
                    ]
                )
                graph.sequence_embedding = sequence_embedding

        return graphs

    def get(self, idx):
        data = self.graphs[idx]
        metadata = self.data_df.loc[[idx]].copy()
        metadata.loc[:, "quantity"] = (
            metadata["quantity"]
            .astype(str)
            .str.split(":")
            .apply(lambda x: [int(i) for i in x])
        )
        metadata_long = metadata.explode("quantity")
        setattr(
            data,
            "entry_oligomeric_state",
            metadata_long["entry_oligomeric_state"].tolist(),
        )
        setattr(data, "num_subunits", metadata_long["num_subunits"].tolist())

        if self.transform is not None:
            data = self.transform(data)

        return data

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            sliced_dataset = copy.copy(self)
            start, stop, step = idx.indices(len(self))
            sliced_dataset.data_df = self.data_df.iloc[start:stop:step].reset_index(
                drop=True
            )
            sliced_dataset.graph_names = self.graph_names[start:stop:step]
            sliced_dataset.graphs = self.graphs[start:stop:step]
            return sliced_dataset

        elif isinstance(idx, (list, np.ndarray)):
            sliced_dataset = copy.copy(self)
            sliced_dataset.data_df = self.data_df.iloc[idx].reset_index(drop=True)
            sliced_dataset.graph_names = [self.graph_names[i] for i in idx]
            sliced_dataset.graphs = [self.graphs[i] for i in idx]
            return sliced_dataset

        else:
            return self.get(idx)


class StoichiometryDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        root,
        data_file: str,
        precomputed_embedding_path: Optional[os.PathLike] = None,
        batch_size: int = 8,
        max_num_nodes: int = 100,
        max_num_edges: int = 401,
        max_samples_per_stoichiometry_class: int = 10000,
        sample_training_data: bool = True,
        num_workers: int = 4,
        transform=None,
        pre_transform=None,
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.data_file = data_file
        self.sample_training_data = sample_training_data
        self.data_df = pd.read_csv(data_file, dtype={"quantity": "object"})

        logger.info(f"Data file: {data_file}")
        logger.info(f"Split counts:\n{self.data_df['split'].value_counts()}")

        self.data_df = self._filter_data_df()
        self.precomputed_embedding_path = precomputed_embedding_path
        self.max_num_nodes = max_num_nodes
        self.max_num_edges = max_num_edges
        self.max_samples_per_stoichiometry_class = max_samples_per_stoichiometry_class
        self.num_workers = num_workers
        self.transform = transform
        self.pre_transform = pre_transform

        self.max_samples_per_cluster = 10

    def _filter_data_df(self):
        # To decreate the number of class 1 in training set. Also monomers don't have interface residues.
        train_data_df = self.data_df.query(
            "split == 'train' & num_subunits > 1"
        ).copy()
        val_data_df = self.data_df.query(
            "split == 'val'"
        ).copy()
        test_data_df = (
            self.data_df.query("split == 'benchmark'")
            .groupby(["cluster_label", "quantity"])
            .head(1)
            .copy()
        )

        self.data_df = (
            pd.concat([train_data_df, val_data_df, test_data_df])
            .sample(frac=1.0)
            .reset_index(drop=True)
        )

        # Used to speed up debugging
        if self.sample_training_data:
            self.data_df = self.data_df.sample(frac=0.1).reset_index(drop=True)

        return self.data_df

    def prepare_data(self):
        pass

    def _sample_training_data(self):
        train_data_df = self.data_df.query(
            "split == 'train'"
        ).copy()
        train_data_df = (
            train_data_df.groupby("cluster_label")
            .apply(lambda x: x.sample(n=min(len(x), self.max_samples_per_cluster), replace=False))
            .reset_index(drop=True)
        )
        train_data_df["quantity"] = train_data_df["quantity"].astype(str).str.split(":")
        train_data_df_long = train_data_df.explode("quantity").reset_index(drop=True)
        data_df_sampled = train_data_df_long.groupby("quantity").apply(
            lambda x: x.sample(
                n=min(len(x), self.max_samples_per_stoichiometry_class), replace=False
            )
        )
        sampled_pdb_ids = data_df_sampled["pdb_id"].unique()
        train_idx = train_data_df.query("pdb_id in @sampled_pdb_ids").index.tolist()
        return train_idx

    def setup(self, stage):
        self.dataset = StoichiometryDataset(
            root=self.root,
            transform=self.transform,
            data_df=self.data_df,
            pre_transform=self.pre_transform,
            precomputed_embedding_path=self.precomputed_embedding_path,
        )

        self.test_idx = self.data_df.query("split == 'benchmark'").index.tolist()
        self.val_idx = self.data_df.query("split == 'val'").index.tolist()
        self.train_idx = self.data_df.query(
            "split == 'train'"
        ).index.tolist()
        self.train = self.dataset[self.train_idx]
        classes_counts = self._get_class_counts()
        self.train_idx_sampled = self._sample_training_data()

        self.train, self.val, self.test = (
            self.dataset[self.train_idx_sampled],
            self.dataset[self.val_idx],
            self.dataset[self.test_idx],
        )
        self.train.stoichiometry_classes_counts = classes_counts[
            "stoichiometry_classes_counts"
        ]
        self.train.interaction_classes_counts = classes_counts[
            "interaction_classes_counts"
        ]

        logger.info(f"Train: {len(self.train)}, Val: {len(self.val)}, Test: {len(self.test)}")
        del classes_counts

    def _resample_training_data(self):
        self.train_idx_sampled = self._sample_training_data()
        self.train = self.dataset[self.train_idx_sampled]

    def _get_class_counts(self):
        stoichiometry_classes = []
        interaction_classes = []

        for graph in self.train:
            stoichiometry_classes.append(graph.quantity.numpy())
            interaction_classes.append(graph.interact.numpy())

        stoichiometry_classes = np.concatenate(stoichiometry_classes)
        interaction_classes = np.concatenate(interaction_classes)
        stoichiometry_classes_unique, stoichiometry_classes_counts = np.unique(
            stoichiometry_classes, return_counts=True
        )
        interaction_classes_unique, interaction_classes_counts = np.unique(
            interaction_classes, return_counts=True
        )

        return {
            "stoichiometry_classes_counts": dict(
                stoichiometry_class=stoichiometry_classes_unique,
                count=stoichiometry_classes_counts,
            ),
            "interaction_classes_counts": dict(
                interaction_class=interaction_classes_unique,
                count=interaction_classes_counts,
            ),
        }

    def train_dataloader(self):
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = world_size
        self.rank = rank

        sampler = DistributedDynamicBatchSampler(
            dataset=self.train,
            max_num_nodes=self.max_num_nodes,
            max_num_edges=self.max_num_edges,
            shuffle=True,
            num_replicas=self.world_size,
            rank=self.rank,
            batch_size=self.batch_size,
            drop_last=True,
        )
        return DataLoader(
            self.train,
            batch_sampler=sampler,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        sampler = DistributedDynamicBatchSampler(
            dataset=self.val,
            max_num_nodes=self.max_num_nodes,
            max_num_edges=self.max_num_edges,
            shuffle=False,
            num_replicas=self.world_size,
            rank=self.rank,
            batch_size=self.batch_size,
            drop_last=True,
        )
        return DataLoader(
            self.val,
            batch_sampler=sampler,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        sampler = DistributedDynamicBatchSampler(
            dataset=self.test,
            max_num_nodes=self.max_num_nodes,
            max_num_edges=self.max_num_edges,
            shuffle=False,
            num_replicas=self.world_size,
            rank=self.rank,
            batch_size=self.batch_size,
            drop_last=True,
        )
        return DataLoader(
            self.test,
            batch_sampler=sampler,
            num_workers=self.num_workers,
        )
