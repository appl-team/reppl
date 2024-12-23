"""Cache implementation for Tree of Thoughts solver."""

from typing import Dict, List, Optional

from loguru import logger

from .utils import Game24Candidate


class ToTCache:
    """Global cache for Tree of Thoughts solver."""

    def __init__(self):
        """Initialize empty cache."""
        self.proposal_cache: Dict[str, List[Game24Candidate]] = {}
        self.evaluation_cache: Dict[str, List[float]] = {}

    def _candidate_key(self, candidate: Game24Candidate) -> str:
        """Generate unique key for a candidate."""
        return f"{candidate.left_numbers}"

    def get_proposals(self, candidate: Game24Candidate) -> List[Game24Candidate] | None:
        """Get cached proposals for a candidate."""
        key = self._candidate_key(candidate)
        res = self.proposal_cache.get(key)
        if res is not None:
            logger.info(f"Found proposals for {key}: {res}")
        return res

    def set_proposals(
        self, candidate: Game24Candidate, proposals: List[Game24Candidate]
    ):
        """Cache proposals for a candidate."""
        key = self._candidate_key(candidate)
        self.proposal_cache[key] = proposals
        logger.info(f"Cached proposals for {key}: {proposals}")

    def get_evaluations(self, candidate: Game24Candidate) -> List[float] | None:
        """Get cached evaluation values for a candidate."""
        key = self._candidate_key(candidate)
        res = self.evaluation_cache.get(key)
        if res is not None:
            logger.info(f"Found evaluations for {key}: {res}")
        return res

    def set_evaluations(self, candidate: Game24Candidate, values: List[float]):
        """Cache evaluation values for a candidate."""
        key = self._candidate_key(candidate)
        self.evaluation_cache[key] = values
        logger.info(f"Cached evaluations for {key}: {values}")


# Global cache instance
_cache: Optional[ToTCache] = None


def get_cache() -> ToTCache:
    """Get global cache instance."""
    global _cache
    if _cache is None:
        _cache = ToTCache()
    return _cache
