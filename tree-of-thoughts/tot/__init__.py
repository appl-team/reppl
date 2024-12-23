"""
Tree of Thoughts (ToT) implementation for solving mathematical puzzles.
Based on https://github.com/princeton-nlp/tree-of-thought-llm
"""

from .cache import get_cache
from .game24 import Game24Task
from .solver import solve

__all__ = ["Game24Task", "solve", "get_cache"]
