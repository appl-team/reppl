"""Evaluation logic for Tree of Thoughts."""

import re
from typing import List

from appl import gen, ppl, traceable
from loguru import logger

from .cache import get_cache
from .examples import VALUE_EXAMPLES
from .utils import Game24Candidate, display_examples

# Value evaluation constants
VALUE_MAP = {
    "sure": 20.0,
    "likely": 1.0,
    "impossible": 0.001,
}


@ppl
def evaluate(current_numbers: str, n_evaluate_sample: int = 3):
    """
    Evaluate if given numbers can reach 24.

    Returns:
        str: Evaluation result (sure/likely/impossible)
    """

    "Evaluate if given numbers can reach 24 (sure/likely/impossible)"
    display_examples(VALUE_EXAMPLES)
    f"Input: {current_numbers}"
    "Thoughts:"
    return [gen() for _ in range(n_evaluate_sample)]


@traceable
def get_all_values(
    proposals: List[Game24Candidate], n_evaluate_sample: int = 3
) -> List[Game24Candidate]:
    """
    Evaluate all proposals and assign values based on their potential.

    Args:
        proposals: List of candidate solutions

    Returns:
        List[Game24Candidate]: Proposals with assigned values
    """
    cache = get_cache()
    cached_results = [cache.get_evaluations(p) for p in proposals]
    values = [
        # only evaluate if not in cache
        cached_result or evaluate(p.left_numbers, n_evaluate_sample)
        for p, cached_result in zip(proposals, cached_results)
    ]

    def get_value(v: str) -> float:
        match = re.match(
            r".*Evaluation:.*(sure|likely|impossible).*", str(v), re.DOTALL
        )
        if match:
            if match.group(1) in VALUE_MAP:
                return VALUE_MAP[match.group(1)]
            else:
                logger.warning(f"Invalid evaluation outcome: {match.group(1)}")
        else:
            logger.warning(f"No match found for evaluation: {v}")
        return 0.0

    for proposal, cached, vs in zip(proposals, cached_results, values):
        if cached is None:  # no cache, use generated values
            cached = [get_value(v) for v in vs]
            cache.set_evaluations(proposal, cached)
        proposal.value = sum(cached) / len(cached)
    return proposals
