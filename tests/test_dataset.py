import os
from pathlib import Path

import pandas as pd
import pytest

from stoic_train.dataset import StoichiometryDataset

pytestmark = pytest.mark.integration

def _resolve_paths() -> tuple[Path, Path]:
    data_root_env = os.environ.get("STOIC_DATA_ROOT")
    data_file_env = os.environ.get("STOIC_DATA_FILE")
    if not data_root_env or not data_file_env:
        pytest.skip(
            "Set STOIC_DATA_ROOT and STOIC_DATA_FILE to run dataset integration tests."
        )

    data_root = Path(data_root_env)
    data_file = Path(data_file_env)
    if not data_root.exists() or not data_file.exists():
        pytest.skip(
            "Dataset paths not found. Set STOIC_DATA_ROOT and STOIC_DATA_FILE to run dataset integration tests."
        )
    return data_root, data_file


def _pick_rows_with_processed_graphs(
    data_file: Path, data_root: Path, n_graphs: int = 10
) -> pd.DataFrame:
    df = pd.read_csv(data_file, dtype={"quantity": "object"})
    processed_dir = data_root / "processed"
    chosen_idx = []
    for idx, row in df.iterrows():
        graph_name = f"{str(row['pdb_id']).lower()}_graph.pt"
        if (processed_dir / graph_name).exists():
            chosen_idx.append(idx)
            if len(chosen_idx) >= n_graphs:
                break

    if len(chosen_idx) < n_graphs:
        pytest.skip(
            f"Only found {len(chosen_idx)} processed graphs, expected at least {n_graphs}."
        )

    return df.loc[chosen_idx].reset_index(drop=True)


@pytest.fixture(scope="module")
def dataset_10():
    data_root, data_file = _resolve_paths()
    df_10 = _pick_rows_with_processed_graphs(data_file, data_root, n_graphs=10)
    return StoichiometryDataset(root=str(data_root), data_df=df_10)


def test_dataset_loads_only_selected_10_graphs(dataset_10) -> None:
    assert len(dataset_10) == 10
    assert len(dataset_10.graphs) == 10
    assert len(dataset_10.raw_file_names) == 10
    assert len(dataset_10.processed_file_names) == 10


def test_dataset_get_adds_metadata_and_parses_quantity(dataset_10) -> None:
    item = dataset_10.get(0)
    assert hasattr(item, "entry_oligomeric_state")
    assert hasattr(item, "num_subunits")
    assert isinstance(item.entry_oligomeric_state, list)
    assert isinstance(item.num_subunits, list)
    assert len(item.entry_oligomeric_state) == len(item.num_subunits)


def test_dataset_slice_and_list_indexing_return_subset_dataset(dataset_10) -> None:
    sliced = dataset_10[2:6]
    indexed = dataset_10[[1, 3, 5]]
    assert len(sliced) == 4
    assert len(indexed) == 3
    assert len(sliced.graphs) == 4
    assert len(indexed.graphs) == 3
