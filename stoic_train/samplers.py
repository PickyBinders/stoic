"""Distributed dynamic batch sampler for graph datasets."""

import math
from typing import Any, Dict, Iterator, List, Optional

import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, Sampler


class DistributedDynamicBatchSampler(BatchSampler):
    """Batch sampler that packs graphs until node/edge maximums are exceeded.

    Graphs are grouped into batches such that neither the total number of
    nodes nor the total number of edges exceeds configurable limits. Batches
    are further capped by a hard ``batch_size`` upper bound. For distributed
    training the batches are striped across replicas by rank.

    Args:
        sampler: Unused, kept for API compatibility with ``BatchSampler``.
        dataset: A PyTorch-Geometric–style dataset whose elements expose
            ``num_nodes`` and ``edge_index`` attributes.
        max_num_nodes: Maximum total nodes per batch (``None`` = no limit).
        max_num_edges: Maximum total edges per batch (``None`` = no limit).
        shuffle: Randomly permute sample order each epoch.
        batch_size: Hard upper bound on the number of graphs per batch.
        drop_last: Drop the final incomplete batch.
        seed: Base random seed (combined with epoch for reproducibility).
        rank: Current replica index in distributed training.
        num_replicas: Total number of replicas.
    """

    def __init__(
        self,
        sampler: Optional[Sampler] = None,
        dataset: Optional[Dataset] = None,
        max_num_nodes: Optional[int] = None,
        max_num_edges: Optional[int] = None,
        shuffle: bool = True,
        batch_size: Optional[int] = 100,
        drop_last: bool = False,
        seed: int = 42,
        rank: int = 0,
        num_replicas: int = 1,
    ):
        self.sampler = sampler
        self.dataset = dataset
        self.max_num_nodes = max_num_nodes
        self.max_num_edges = max_num_edges
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.num_replicas = num_replicas
        self.rank = rank
        self.graph_stats = self._compute_graph_stats()
        self._batch_size = batch_size

        if self.shuffle:
            self.generator = torch.Generator()
            self.generator.manual_seed(self.seed)

    def _compute_graph_stats(self) -> List[Dict[str, int]]:
        """Pre-compute node and edge counts for every graph in the dataset."""
        stats: List[Dict[str, int]] = []
        for data in tqdm(self.dataset, desc="Computing graph stats", total=len(self.dataset)):
            stats.append(
                {
                    "num_nodes": data.num_nodes,
                    "num_edges": (
                        data.edge_index.size(1) if hasattr(data, "edge_index") else 0
                    ),
                }
            )
        return stats

    def _create_batches(self, indices: List[int]) -> List[List[int]]:
        """Greedily pack ``indices`` into batches respecting budgets."""
        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_nodes = 0
        current_edges = 0

        for idx in tqdm(indices, desc="Creating batches", total=len(indices)):
            stats = self.graph_stats[idx]

            if (self.max_num_nodes and stats["num_nodes"] > self.max_num_nodes) or (
                self.max_num_edges and stats["num_edges"] > self.max_num_edges
            ):
                continue

            would_exceed = (
                (
                    self.max_num_nodes
                    and current_nodes + stats["num_nodes"] > self.max_num_nodes
                )
                or (
                    self.max_num_edges
                    and current_edges + stats["num_edges"] > self.max_num_edges
                )
                or len(current_batch) >= self._batch_size
            )

            if would_exceed:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [idx]
                current_nodes = stats["num_nodes"]
                current_edges = stats["num_edges"]
            else:
                current_batch.append(idx)
                current_nodes += stats["num_nodes"]
                current_edges += stats["num_edges"]

        if current_batch and not self.drop_last:
            batches.append(current_batch)

        return batches

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling across replicas."""
        self.epoch = epoch
        if self.shuffle:
            self.generator.manual_seed(self.seed + self.epoch)

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            indices = torch.randperm(len(indices), generator=self.generator).tolist()

        batches = self._create_batches(indices)
        num_batches_per_replica = math.ceil(len(batches) / self.num_replicas)
        total_size = num_batches_per_replica * self.num_replicas

        if len(batches) < total_size:
            batches.extend(batches[: (total_size - len(batches))])

        rank_batches = batches[self.rank : total_size : self.num_replicas]
        for batch in rank_batches:
            yield batch

    def __len__(self) -> int:
        indices = list(range(len(self.dataset)))
        batches = self._create_batches(indices)
        num_batches_per_replica = math.floor(len(batches) / self.num_replicas)

        # Subtract one for training splits to avoid an off-by-one with the
        # scheduler's total_steps (OneCycleLR counts from 0).
        if (
            hasattr(self.dataset, "data_df")
            and "split" in self.dataset.data_df.columns
            and "train" in self.dataset.data_df["split"].values
        ):
            num_batches_per_replica -= 1

        return num_batches_per_replica

    @property
    def batch_size(self) -> int:
        return self._batch_size
