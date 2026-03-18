"""Command-line interface for protein stoichiometry prediction.

Usage:
    stoic_predict_stoichiometry --sequences SEQ1 SEQ2 ... [--model MODEL] [--top-n N] [--device DEVICE]

Or as a Python API:
    from stoic.predict_stoichiometry import predict_stoichiometry
    results = predict_stoichiometry(["MKTL...", "MGSS..."])
"""

import argparse
import json
import os
import pickle
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
from loguru import logger

from stoic.model import Stoic


def _read_fasta_sequences(fasta_path: Union[str, os.PathLike]) -> List[str]:
    """Parse sequences from a FASTA file."""
    sequences: List[str] = []
    current: List[str] = []
    with open(fasta_path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line)
    if current:
        sequences.append("".join(current))
    return sequences


def _is_fasta_file(path: Path) -> bool:
    return path.suffix.lower() in {".fa", ".fasta", ".faa", ".fna", ".fas"}


def _normalize_complex_inputs(
    sequences: Union[List[str], str, os.PathLike],
) -> Tuple[List[Tuple[str, List[str]]], bool]:
    """Return list of (complex_name, sequences) and whether source is directory."""
    if isinstance(sequences, list):
        return [("input_sequences", sequences)], False

    path = Path(sequences)
    if path.is_file():
        if not _is_fasta_file(path):
            raise ValueError(f"Input file is not a FASTA file: {path}")
        return [(path.stem, _read_fasta_sequences(path))], False

    if path.is_dir():
        fasta_files = sorted(p for p in path.iterdir() if p.is_file() and _is_fasta_file(p))
        if not fasta_files:
            raise ValueError(f"No FASTA files found in directory: {path}")
        return [(p.stem, _read_fasta_sequences(p)) for p in fasta_files], True

    raise ValueError(f"Input path does not exist: {path}")


def _chain_id_from_index(idx: int) -> str:
    """Convert 0-based index to chain id: A..Z, AA..AZ, BA..."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    out = ""
    idx += 1
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        out = letters[rem] + out
    return out


def _build_af3_input_json(
    complex_name: str,
    predictions: List[Dict[str, Union[int, float]]],
) -> Dict[str, object]:
    """Build AF3 input JSON from top-ranked stoichiometry candidate."""
    if not predictions:
        raise ValueError("Cannot build AF3 input from empty predictions.")

    top_candidate = predictions[0]
    sequence_counts = [
        (seq, int(copy_n))
        for seq, copy_n in top_candidate.items()
        if seq not in {"rank", "probability"}
    ]
    expanded_sequences: List[str] = []
    for seq, copy_n in sequence_counts:
        expanded_sequences.extend([seq] * max(copy_n, 0))

    af3_sequences = []
    for i, seq in enumerate(expanded_sequences):
        af3_sequences.append(
            {
                "protein": {
                    "id": _chain_id_from_index(i),
                    "sequence": seq,
                }
            }
        )

    return {
        "name": complex_name,
        "modelSeeds": [random.randint(1, 2**31 - 1)],
        "sequences": af3_sequences,
    }


def predict_stoichiometry(
    sequences: Union[List[str], str, os.PathLike],
    model_name: str = "PickyBinders/stoic",
    top_n: int = 3,
    device: Optional[torch.device] = None,
    return_residue_weights: bool = False,
    max_inference_seq_len: Optional[int] = None,
) -> Union[
    List[Dict[str, int]],
    Tuple[List[Dict[str, int]], Dict[str, object]],
    Dict[str, Union[List[Dict[str, int]], Dict[str, object]]],
]:
    """Predict stoichiometry for a list of protein sequences.

    Args:
        sequences: Either a list of sequences, a FASTA file path, or a
            directory containing FASTA files.
        model_name: HuggingFace model identifier or local path.
        top_n: Number of top stoichiometry candidates to return.
        device: Device to run inference on. Defaults to CUDA if available.

    Returns:
        For a list or single FASTA file: list of prediction candidates
        (or tuple including residue predictions if requested).
        For a FASTA directory: mapping from FASTA stem to per-complex output.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading model: {model_name}")
    model = Stoic.from_pretrained(model_name)
    model = model.to(device).eval()
    if max_inference_seq_len is not None:
        model.enable_full_length_inference(max_inference_seq_len)

    complexes, from_directory = _normalize_complex_inputs(sequences)
    outputs: Dict[str, Union[List[Dict[str, int]], Dict[str, object]]] = {}
    for complex_name, complex_sequences in complexes:
        logger.info(
            f"Running inference for {complex_name} on {len(complex_sequences)} sequences"
        )
        start = time.time()
        with torch.no_grad():
            pred = model.predict_stoichiometry(
                complex_sequences,
                top_n=top_n,
                return_residue_weights=return_residue_weights,
            )
        elapsed = time.time() - start
        logger.info(f"Inference for {complex_name} completed in {elapsed:.2f}s")
        outputs[complex_name] = pred

    if from_directory:
        return outputs
    return next(iter(outputs.values()))


def main():
    """Entry point for the stoic_predict_stoichiometry CLI."""
    parser = argparse.ArgumentParser(
        description="Predict protein complex stoichiometry from sequences."
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        required=False,
        help="Protein sequences (one per unique chain).",
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="Path to a FASTA file or a directory with FASTA files.",
    )
    parser.add_argument(
        "--model",
        default="PickyBinders/stoic",
        help="HuggingFace model name or local path (default: PickyBinders/stoic).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top stoichiometry candidates (default: 3).",
    )
    parser.add_argument(
        "--return-residue-weights",
        action="store_true",
        help="Return residue weights (default: False).",
    )
    parser.add_argument(
        "--max-inference-seq-len",
        type=int,
        default=None,
        help="Maximum sequence length for full-length inference (default: None).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory to save the results (default: None).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use, e.g. 'cuda' or 'cpu' (default: auto-detect).",
    )
    args = parser.parse_args()
    if args.sequences is None and args.input_path is None:
        parser.error("Provide either --sequences or --input-path")
    if args.sequences is not None and args.input_path is not None:
        parser.error("Use either --sequences or --input-path, not both")

    device = torch.device(args.device) if args.device else None
    input_data: Union[List[str], str] = args.sequences if args.sequences is not None else args.input_path
    results = predict_stoichiometry(
        sequences=input_data,
        model_name=args.model,
        top_n=args.top_n,
        device=device,
        return_residue_weights=args.return_residue_weights,
        max_inference_seq_len=args.max_inference_seq_len,
    )

    is_dir_input = args.input_path is not None and Path(args.input_path).is_dir()
    if is_dir_input:
        output_dir = args.output_dir or "stoic_predictions"
        os.makedirs(output_dir, exist_ok=True)
        assert isinstance(results, dict)
        for complex_name, complex_result in results.items():
            print(f"\nComplex: {complex_name}")
            if args.return_residue_weights:
                complex_predictions, residue_predictions = complex_result
            else:
                complex_predictions = complex_result
                residue_predictions = None

            with open(os.path.join(output_dir, f"{complex_name}.json"), "w") as f:
                json.dump(complex_predictions, f)
            af3_json = _build_af3_input_json(complex_name, complex_predictions)
            with open(os.path.join(output_dir, f"{complex_name}_af3_input.json"), "w") as f:
                json.dump(af3_json, f)
            if residue_predictions is not None:
                with open(
                    os.path.join(output_dir, f"{complex_name}_residue_predictions.pkl"),
                    "wb",
                ) as f:
                    pickle.dump(residue_predictions, f)

            for i, candidate in enumerate(complex_predictions, 1):
                print(f"  Candidate {i}:")
                for seq, copies in candidate.items():
                    print(f"    {seq[:40]}{'...' if len(seq) > 40 else ''}: {copies}")
        return

    if args.return_residue_weights:
        results, residue_predictions = results
    else:
        residue_predictions = None

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(results, f)
        af3_json = _build_af3_input_json("input_sequences", results)
        with open(os.path.join(args.output_dir, "af3_input.json"), "w") as f:
            json.dump(af3_json, f)
        if residue_predictions is not None:
            with open(os.path.join(args.output_dir, "residue_predictions.pkl"), "wb") as f:
                pickle.dump(residue_predictions, f)

    for i, candidate in enumerate(results, 1):
        print(f"Candidate {i}:")
        for seq, copies in candidate.items():
            print(f"  {seq[:40]}{'...' if len(seq) > 40 else ''}: {copies}")

if __name__ == "__main__":
    main()
