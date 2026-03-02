import inspect
from functools import wraps
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from loguru import logger


def top_n_stoichiometry_combinations(
    logits: torch.Tensor,
    n: int = 5,
    class_labels: Optional[List[int]] = None,
    beam_width: int = 10,
    use_ranks: bool = True,
    rank_weight: float = 1.0,
) -> List[Tuple[List[int], float, float]]:
    """
    Find the top-N most likely combinations of stoichiometry classes based on node logits.

    Args:
        logits: Tensor of shape [num_nodes, num_classes] containing logits
        n: Number of top combinations to return
        class_labels: List of class labels (default: range(num_classes))
        beam_width: Beam width for beam search method
        use_ranks: If True, use ranks instead of raw probabilities (better for class imbalance)
        rank_weight: Weight for rank-based scoring (higher = more emphasis on ranks)

    Returns:
        List of tuples (combination, score, probability) where:
            - combination is a list of stoichiometry classes (one per node)
            - score is the combined score of that combination (lower is better for ranks)
            - probability is the joint probability of that combination
    """
    num_nodes, num_classes = logits.shape
    if class_labels is None:
        class_labels = list(range(num_classes))

    probs = F.softmax(logits, dim=1)

    if use_ranks:
        ranks = torch.argsort(torch.argsort(logits, dim=1, descending=True), dim=1) + 1
        scores = ranks.float()

        if rank_weight < 1.0:
            prob_scores = 1.0 - probs
            scores = (
                rank_weight * scores + (1.0 - rank_weight) * prob_scores * num_classes
            )
    else:
        scores = 1.0 - probs
        scores = scores * num_classes

    return beam_search(scores, probs, n, class_labels, beam_width, use_ranks)


def beam_search(
    scores: torch.Tensor,
    probs: torch.Tensor,
    n: int,
    class_labels: List[int],
    beam_width: int,
    use_ranks: bool = True,
) -> List[Tuple[List[int], float, float]]:
    """
    Find top combinations using beam search with ranks or scores.
    """
    num_nodes, num_classes = scores.shape

    beam = [([], 0.0, 1.0)]

    for node_idx in range(num_nodes):
        candidates = []
        for combination, score, prob in beam:
            for class_idx in range(num_classes):
                new_combination = combination + [class_idx]
                new_score = score + scores[node_idx, class_idx].item()
                new_prob = prob * probs[node_idx, class_idx].item()
                candidates.append((new_combination, new_score, new_prob))

        beam = sorted(candidates, key=lambda x: x[1])[:beam_width]

    top_combinations = []
    for combination, score, prob in beam[:n]:
        labeled_combination = [class_labels[idx] for idx in combination]
        top_combinations.append((labeled_combination, score, prob))

    return top_combinations


def print_init_args(init_func):
    @wraps(init_func)
    def wrapper(self, *args, **kwargs):
        sig = inspect.signature(init_func)

        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()

        arg_dict = dict(bound_args.arguments)
        arg_dict.pop("self")

        logger.info(f"Init arguments for {self.__class__.__name__}:")
        logger.info(arg_dict)

        return init_func(self, *args, **kwargs)

    return wrapper
