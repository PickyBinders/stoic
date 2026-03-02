"""Command-line interface for protein stoichiometry prediction.

Usage:
    stoic_predict_stoichiometry --sequences SEQ1 SEQ2 ... [--model MODEL] [--top-n N] [--device DEVICE]

Or as a Python API:
    from stoic.predict_stoichiometry import predict_stoichiometry
    results = predict_stoichiometry(["MKTL...", "MGSS..."])
"""

import argparse
import sys
import time
from typing import Dict, List, Optional

import torch
from loguru import logger

from stoic.model import Stoic


def predict_stoichiometry(
    sequences: List[str],
    model_name: str = "PickyBinders/stoic",
    top_n: int = 3,
    device: Optional[torch.device] = None,
) -> List[Dict[str, int]]:
    """Predict stoichiometry for a list of protein sequences.

    Args:
        sequences: Protein sequences (one per unique chain).
        model_name: HuggingFace model identifier or local path.
        top_n: Number of top stoichiometry candidates to return.
        device: Device to run inference on. Defaults to CUDA if available.

    Returns:
        List of dicts mapping each sequence to its predicted copy number,
        ordered from most to least likely.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading model: {model_name}")
    model = Stoic.from_pretrained(model_name)
    model = model.to(device).eval()

    logger.info(f"Running inference on {len(sequences)} sequences")
    start = time.time()
    with torch.no_grad():
        results = model.predict_stoichiometry(sequences, top_n=top_n)
    elapsed = time.time() - start
    logger.info(f"Inference completed in {elapsed:.2f}s")

    return results


def main():
    """Entry point for the stoic_predict_stoichiometry CLI."""
    parser = argparse.ArgumentParser(
        description="Predict protein complex stoichiometry from sequences."
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        required=True,
        help="Protein sequences (one per unique chain).",
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
        "--device",
        default=None,
        help="Device to use, e.g. 'cuda' or 'cpu' (default: auto-detect).",
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    results = predict_stoichiometry(
        sequences=args.sequences,
        model_name=args.model,
        top_n=args.top_n,
        device=device,
    )

    for i, candidate in enumerate(results, 1):
        print(f"Candidate {i}:")
        for seq, copies in candidate.items():
            print(f"  {seq[:40]}{'...' if len(seq) > 40 else ''}: {copies} copies")


if __name__ == "__main__":
    main()
