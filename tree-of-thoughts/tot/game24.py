"""Game24 task implementation and utilities."""

import re

import pandas as pd
import sympy


class Game24Task:
    """
    Game24 puzzle task handler.

    A puzzle where the goal is to use four numbers and basic arithmetic operations
    to obtain 24.
    """

    def __init__(self, filepath="24.csv"):
        """
        Initialize Game24Task.

        Args:
            filepath (str): Path to CSV file containing puzzles
        """
        super().__init__()
        self.data = list(pd.read_csv(filepath)["Puzzles"])

    def get_input(self, idx: int) -> str:
        """Get puzzle input at specified index."""
        return self.data[idx]

    def test_output(self, idx: int, expression: str) -> bool:
        """
        Test if an expression is a valid solution for the puzzle at given index.

        Args:
            idx (int): Puzzle index
            expression (str): Mathematical expression to test

        Returns:
            bool: True if expression is valid solution, False otherwise
        """
        numbers = re.findall(r"\d+", expression)
        problem_numbers = re.findall(r"\d+", self.data[idx])

        if sorted(numbers) != sorted(problem_numbers):
            return False

        try:
            return sympy.simplify(expression) == 24
        except Exception:
            return False
