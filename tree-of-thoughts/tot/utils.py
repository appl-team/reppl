from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Game24Candidate:
    """Represents a candidate solution in the Game24 puzzle."""

    start_numbers: str
    left_numbers: str
    steps: list[str]
    value: float = 0.0


def display_examples(examples: List[Dict[str, Any]]) -> str:
    """Display examples in a formatted way."""
    s = ""
    for example in examples:
        for k, v in example.items():
            if isinstance(v, list):
                s += f"{k}:\n" + "\n".join(v) + "\n"
            else:
                s += f"{k}: {v}\n"
    return s
