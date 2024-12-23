"""Proposal generation logic for Tree of Thoughts."""

import re
from typing import List

from appl import gen, ppl, traceable

from .cache import get_cache
from .examples import PROPOSE_EXAMPLES
from .utils import Game24Candidate, display_examples


def get_current_numbers(raw: str) -> str:
    """Extract current numbers from a step description."""
    # example: (left: 1 2 3)
    match = re.match(r".*\(left: (.*)\).*", raw, re.DOTALL)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"No match found for {raw}")


@ppl
def propose(current_numbers: str, n_propose_sample: int = 8):
    """Generate possible next steps for given numbers."""

    f"Propose at most {n_propose_sample} possible next steps without explanation. Strictly follow the format as in the examples."
    display_examples(PROPOSE_EXAMPLES)
    f"Input: {current_numbers}"
    "Possible next steps:"
    return gen()


@traceable
def get_all_proposals(
    candidates: List[Game24Candidate], n_propose_sample: int = 8
) -> List[Game24Candidate]:
    """
    Generate all possible next steps for given candidates.

    Args:
        candidates: List of current candidate solutions

    Returns:
        List[Game24Candidate]: New candidate solutions
    """
    cache = get_cache()
    all_proposals = []
    cached_results = [cache.get_proposals(c) for c in candidates]
    proposals = [
        # only propose if not in cache
        cached_result or propose(c.left_numbers, n_propose_sample)
        for c, cached_result in zip(candidates, cached_results)
    ]

    def get_proposals(response: str, candidate: Game24Candidate):
        proposals = []
        for line in str(response).strip().split("\n"):
            try:
                proposals.append(
                    Game24Candidate(
                        start_numbers=candidate.left_numbers,
                        left_numbers=get_current_numbers(line),
                        steps=candidate.steps + [line.strip()],
                    )
                )
            except ValueError:
                pass
        return proposals

    for candidate, cached, ps in zip(candidates, cached_results, proposals):
        if cached is None:
            cached = get_proposals(ps, candidate)
            cache.set_proposals(candidate, cached)
        else:
            cached = [
                Game24Candidate(
                    start_numbers=p.start_numbers,
                    left_numbers=p.left_numbers,
                    steps=candidate.steps + [p.steps[-1]],  # only get the last step
                )
                for p in cached
            ]
        all_proposals.extend(cached)
    return all_proposals
