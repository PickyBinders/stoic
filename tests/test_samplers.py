import torch

from stoic_train.samplers import DistributedDynamicBatchSampler


class DummyGraph:
    def __init__(self, num_nodes: int, num_edges: int):
        self.num_nodes = num_nodes
        self.edge_index = torch.zeros((2, num_edges), dtype=torch.long)


class _SplitColumn:
    def __init__(self, values):
        self.values = values


class _DummyDataFrame:
    def __init__(self, split_values):
        self.columns = ["split"]
        self._split = _SplitColumn(split_values)

    def __getitem__(self, key):
        if key == "split":
            return self._split
        raise KeyError(key)


class DummyDataset:
    def __init__(self, graphs):
        self.data = graphs

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _make_dataset(node_edge_pairs):
    return DummyDataset([DummyGraph(n, e) for n, e in node_edge_pairs])


def _flatten(batches):
    return [idx for batch in batches for idx in batch]


def test_batch_constraints_respected_for_nodes_edges_and_batch_size() -> None:
    dataset = _make_dataset([(3, 6), (4, 8), (2, 4), (5, 10), (1, 2)])
    sampler = DistributedDynamicBatchSampler(
        dataset=dataset,
        max_num_nodes=8,
        max_num_edges=14,
        batch_size=2,
        shuffle=False,
        num_replicas=1,
        rank=0,
    )

    batches = list(iter(sampler))
    assert len(batches) > 0
    for batch in batches:
        total_nodes = sum(dataset[idx].num_nodes for idx in batch)
        total_edges = sum(dataset[idx].edge_index.size(1) for idx in batch)
        assert len(batch) <= 2
        assert total_nodes <= 8
        assert total_edges <= 14


def test_none_limits_means_only_batch_size_limits_packing() -> None:
    dataset = _make_dataset([(100, 1000), (120, 1200), (90, 900), (80, 800), (70, 700)])
    sampler = DistributedDynamicBatchSampler(
        dataset=dataset,
        max_num_nodes=None,
        max_num_edges=None,
        batch_size=2,
        shuffle=False,
        num_replicas=1,
        rank=0,
    )

    batches = list(iter(sampler))
    assert batches == [[0, 1], [2, 3], [4]]


def test_oversized_graphs_are_skipped() -> None:
    dataset = _make_dataset([(3, 4), (100, 2), (2, 3), (4, 5)])
    sampler = DistributedDynamicBatchSampler(
        dataset=dataset,
        max_num_nodes=10,
        max_num_edges=10,
        batch_size=3,
        shuffle=False,
        num_replicas=1,
        rank=0,
    )

    batches = list(iter(sampler))
    assert 1 not in _flatten(batches)


def test_drop_last_controls_whether_incomplete_batch_is_emitted() -> None:
    dataset = _make_dataset([(1, 1), (1, 1), (1, 1)])
    keep_last = DistributedDynamicBatchSampler(
        dataset=dataset,
        max_num_nodes=None,
        max_num_edges=None,
        batch_size=2,
        drop_last=False,
        shuffle=False,
        num_replicas=1,
        rank=0,
    )
    drop_last = DistributedDynamicBatchSampler(
        dataset=dataset,
        max_num_nodes=None,
        max_num_edges=None,
        batch_size=2,
        drop_last=True,
        shuffle=False,
        num_replicas=1,
        rank=0,
    )

    assert list(iter(keep_last)) == [[0, 1], [2]]
    assert list(iter(drop_last)) == [[0, 1]]


def test_shuffling_and_set_epoch_are_deterministic_per_epoch() -> None:
    dataset = _make_dataset([(1, 1)] * 8)
    sampler = DistributedDynamicBatchSampler(
        dataset=dataset,
        max_num_nodes=None,
        max_num_edges=None,
        batch_size=1,
        shuffle=True,
        seed=123,
        num_replicas=1,
        rank=0,
    )

    sampler.set_epoch(0)
    order_epoch0_a = _flatten(list(iter(sampler)))
    sampler.set_epoch(0)
    order_epoch0_b = _flatten(list(iter(sampler)))
    sampler.set_epoch(1)
    order_epoch1 = _flatten(list(iter(sampler)))

    assert order_epoch0_a == order_epoch0_b
    assert order_epoch1 != order_epoch0_a


def test_distributed_iter_stripes_batches_and_pads_when_needed() -> None:
    dataset = _make_dataset([(1, 1)] * 5)
    rank0 = DistributedDynamicBatchSampler(
        dataset=dataset,
        max_num_nodes=None,
        max_num_edges=None,
        batch_size=2,
        shuffle=False,
        num_replicas=2,
        rank=0,
    )
    rank1 = DistributedDynamicBatchSampler(
        dataset=dataset,
        max_num_nodes=None,
        max_num_edges=None,
        batch_size=2,
        shuffle=False,
        num_replicas=2,
        rank=1,
    )

    rank0_batches = list(iter(rank0))
    rank1_batches = list(iter(rank1))

    assert rank0_batches == [[0, 1], [4]]
    assert rank1_batches == [[2, 3], [0, 1]]


def test_len_matches_floor_per_replica_and_train_adjustment() -> None:
    dataset = _make_dataset([(1, 1)] * 5)
    sampler = DistributedDynamicBatchSampler(
        dataset=dataset,
        max_num_nodes=None,
        max_num_edges=None,
        batch_size=2,
        shuffle=False,
        num_replicas=2,
        rank=0,
    )
    assert len(sampler) == 1

    dataset_with_train = _make_dataset([(1, 1)] * 5)
    dataset_with_train.data_df = _DummyDataFrame(["train", "val", "test"])
    sampler_with_train = DistributedDynamicBatchSampler(
        dataset=dataset_with_train,
        max_num_nodes=None,
        max_num_edges=None,
        batch_size=2,
        shuffle=False,
        num_replicas=2,
        rank=0,
    )
    assert len(sampler_with_train) == 0
