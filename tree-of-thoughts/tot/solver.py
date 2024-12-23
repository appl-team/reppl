"""Core solving logic for Tree of Thoughts."""

from typing import List

from appl import gen, ppl, traceable
from loguru import logger

from .evaluator import get_all_values
from .examples import COT_EXAMPLES, JUDGE_EXAMPLES
from .generator import get_all_proposals
from .utils import Game24Candidate, display_examples


@ppl
def get_answer(input_numbers: str, steps: List[str]):
    """Generate final answer from steps."""

    "Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number."
    display_examples(COT_EXAMPLES)
    f"Input: {input_numbers}"
    f"Steps:\n"
    steps
    "Answer:"
    return gen(temperature=0.0)


@ppl
def judge(input: str, answer: str):
    """Validate if answer is correct."""

    "Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24."
    display_examples(JUDGE_EXAMPLES)
    f"Input: {input}"
    f"Answer: {answer}"
    "Judge:"
    return gen(temperature=0.0)


@traceable
def solve(
    input_numbers: str,
    n_propose_sample: int = 8,
    n_evaluate_sample: int = 2,  # 3 in the paper
    n_select_sample: int = 5,
    steps: int = 3,
):
    """
    Solve Game24 puzzle using Tree of Thoughts approach.

    Args:
        input_numbers (str): Space-separated numbers to use
        steps (int): Maximum steps to solve

    Returns:
        List[str]: List of valid solutions
    """
    current_candidates = [
        Game24Candidate(
            start_numbers=input_numbers, left_numbers=input_numbers, steps=[]
        )
    ]

    # Main solving loop
    for _ in range(steps):
        new_candidates = get_all_proposals(current_candidates, n_propose_sample)
        new_candidates = get_all_values(new_candidates, n_evaluate_sample)
        current_candidates = sorted(
            new_candidates, key=lambda x: x.value, reverse=True
        )[:n_select_sample]

        for c in current_candidates:
            logger.info(
                f"Start: {c.start_numbers}, Left: {c.left_numbers}, Value: {c.value}, Steps: {c.steps}"
            )

    # Validate solutions
    filtered_candidates = [c for c in current_candidates if c.left_numbers == "24"]
    answers = [get_answer(input_numbers, c.steps) for c in filtered_candidates]
    judge_results = [judge(input_numbers, str(a)) for a in answers]

    candidates = []
    for candidate, answer, judge_result in zip(
        filtered_candidates, answers, judge_results
    ):
        answer = str(answer).rsplit("=", 1)[0].strip()
        logger.info(
            f"Steps: {candidate.steps}, Answer: {answer}, Judge: {judge_result}"
        )
        if "sure" in str(judge_result).strip():
            candidates.append(answer)

    return candidates
