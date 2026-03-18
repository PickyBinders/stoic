import torch

from stoic.utils import beam_search, top_n_stoichiometry_combinations


def test_top_n_stoichiometry_combinations_returns_n_items() -> None:
    logits = torch.tensor([[2.0, 0.1], [0.2, 1.7]])
    results = top_n_stoichiometry_combinations(logits, n=2, class_labels=[1, 2])
    assert len(results) == 2
    assert all(len(combo) == 2 for combo, _, _ in results)


def test_beam_search_preserves_label_mapping() -> None:
    scores = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    probs = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
    results = beam_search(
        scores=scores,
        probs=probs,
        n=1,
        class_labels=[10, 20],
        beam_width=2,
        use_ranks=True,
    )
    assert results[0][0][0] in {10, 20}


def test_top_n_defaults_to_range_class_labels() -> None:
    logits = torch.tensor([[0.5, 1.5, -0.2], [2.0, 0.1, -0.1]], dtype=torch.float32)
    results = top_n_stoichiometry_combinations(logits, n=3, class_labels=None)
    assert len(results) == 3
    assert all(idx in {0, 1, 2} for combo, _, _ in results for idx in combo)


def test_top_n_length_is_capped_by_beam_width() -> None:
    logits = torch.tensor([[2.0, 0.1], [0.2, 1.7]], dtype=torch.float32)
    results = top_n_stoichiometry_combinations(
        logits, n=10, class_labels=[0, 1], beam_width=2
    )
    assert len(results) == 2


def test_top_n_probability_mode_returns_consistent_score_and_probability() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    probs = torch.softmax(logits, dim=1)
    results = top_n_stoichiometry_combinations(
        logits,
        n=1,
        class_labels=[0, 1],
        beam_width=4,
        use_ranks=False,
    )
    combo, score, prob = results[0]
    expected_prob = probs[0, combo[0]].item() * probs[1, combo[1]].item()
    expected_score = (
        ((1.0 - probs[0, combo[0]]) * 2).item()
        + ((1.0 - probs[1, combo[1]]) * 2).item()
    )
    assert prob == expected_prob
    assert score == expected_score


def test_top_n_rank_weight_blending_changes_score_scale() -> None:
    logits = torch.tensor([[1.0, 0.7, 0.1], [0.9, 0.2, 0.0]], dtype=torch.float32)
    rank_only = top_n_stoichiometry_combinations(
        logits,
        n=1,
        class_labels=[0, 1, 2],
        beam_width=6,
        use_ranks=True,
        rank_weight=1.0,
    )[0]
    blended = top_n_stoichiometry_combinations(
        logits,
        n=1,
        class_labels=[0, 1, 2],
        beam_width=6,
        use_ranks=True,
        rank_weight=0.5,
    )[0]

    assert rank_only[0] == blended[0]
    assert rank_only[1] != blended[1]
