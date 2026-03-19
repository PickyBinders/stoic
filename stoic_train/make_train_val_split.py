"""Create train/val splits by partitioning category IDs from train rows.

This script updates only rows currently labeled as ``train`` in a CSV file.
Rows in other split groups (e.g. ``benchmark``) are left unchanged.
"""

import argparse
import random
from pathlib import Path

import pandas as pd
from loguru import logger


def assign_split(row, train_ids, test_ids, split_feature):
    values = [x.strip() for x in str(row[split_feature]).split(";") if x.strip()]
    if values and all(cat in test_ids for cat in values):
        return "val"
    elif values and all(cat in train_ids for cat in values):
        return "train"
    else:
        return "unassigned"


def _parse_ids(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(";") if x.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split current train rows into train/val based on category IDs."
    )
    parser.add_argument(
        "--data-file",
        required=True,
        help="Input CSV containing at least split and split-feature columns.",
    )
    parser.add_argument(
        "--split-feature",
        default="cluster_label",
        help=(
            "Column with category IDs separated by ';' "
            "(default: cluster_label)."
        ),
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output CSV path. Default: <input_stem>_train_val.csv next to input.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help=(
            "Fraction of unique category IDs assigned to val "
            "(default: 0.1, set 0 to keep all train rows as train)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible ID split (default: 42).",
    )
    parser.add_argument(
        "--split-column",
        default="split",
        help="Column name containing split labels (default: split).",
    )
    parser.add_argument(
        "--train-label",
        default="train",
        help="Label in split-column considered as train before reassignment.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    data_path = Path(args.data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Input file not found: {data_path}")

    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1).")

    df = pd.read_csv(data_path)

    if args.split_column not in df.columns:
        raise KeyError(f"Missing split column: {args.split_column}")
    if args.split_feature not in df.columns:
        raise KeyError(f"Missing split feature column: {args.split_feature}")

    train_mask = df[args.split_column].astype(str) == args.train_label
    train_df = df.loc[train_mask].copy()

    if train_df.empty:
        raise ValueError(
            f"No rows found where {args.split_column} == '{args.train_label}'."
        )

    all_ids = sorted(
        {
            cat
            for value in train_df[args.split_feature].fillna("").astype(str)
            for cat in _parse_ids(value)
        }
    )
    if not all_ids:
        raise ValueError(f"No IDs found in split feature column: {args.split_feature}")

    rng = random.Random(args.seed)
    shuffled = all_ids[:]
    rng.shuffle(shuffled)

    n_val = 0 if args.val_ratio == 0 else max(1, int(round(len(shuffled) * args.val_ratio)))
    val_ids = set(shuffled[:n_val])
    train_ids = set(shuffled[n_val:])

    train_df[args.split_column] = train_df.apply(
        assign_split,
        axis=1,
        train_ids=train_ids,
        test_ids=val_ids,
        split_feature=args.split_feature,
    )

    df_out = df.copy()
    df_out.loc[train_mask, args.split_column] = train_df[args.split_column].values

    out_path = (
        Path(args.output_file)
        if args.output_file
        else data_path.with_name(f"{data_path.stem}_train_val.csv")
    )
    df_out.to_csv(out_path, index=False)

    split_counts = df_out[args.split_column].value_counts(dropna=False).to_dict()
    unassigned_rows = int((df_out[args.split_column].astype(str) == "unassigned").sum())
    logger.info(f"Saved: {out_path}")
    logger.info(f"Unique IDs in '{args.split_feature}': {len(all_ids)}")
    logger.info(f"Assigned to train IDs: {len(train_ids)}")
    logger.info(f"Assigned to val IDs: {len(val_ids)}")
    logger.info(f"Rows marked as unassigned: {unassigned_rows}")
    logger.info(f"Split counts: {split_counts}")


if __name__ == "__main__":
    main()
