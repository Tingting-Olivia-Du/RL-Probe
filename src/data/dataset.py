"""
MATH Dataset Loader and Processor

Handles loading, filtering, and preprocessing of MATH dataset
for KL-divergence analysis experiments.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MATHDataset:
    """
    MATH Dataset wrapper with filtering capabilities.

    Filters problems based on difficulty level and model performance
    (DPO-wrong, RLVR-right) for targeted analysis.
    """

    def __init__(
        self,
        levels: List[int] = [3, 4],
        split: str = "test",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize MATH dataset loader.

        Args:
            levels: Difficulty levels to include (1-5)
            split: Dataset split to use
            cache_dir: Directory for caching downloaded data
        """
        self.levels = levels
        self.split = split
        self.cache_dir = cache_dir
        self.dataset = None
        self.filtered_problems = []

    def load(self) -> "MATHDataset":
        """Load the MATH dataset from HuggingFace."""
        logger.info(f"Loading MATH dataset (split={self.split}, levels={self.levels})")

        self.dataset = load_dataset(
            "lighteval/MATH",
            split=self.split,
            cache_dir=self.cache_dir,
        )

        # Filter by difficulty level
        self.dataset = self.dataset.filter(
            lambda x: x["level"] in [f"Level {l}" for l in self.levels]
        )

        logger.info(f"Loaded {len(self.dataset)} problems at levels {self.levels}")
        return self

    def get_problems(self) -> List[Dict[str, Any]]:
        """Get all problems as a list of dictionaries."""
        if self.dataset is None:
            self.load()
        return [dict(problem) for problem in self.dataset]

    def get_problem_by_idx(self, idx: int) -> Dict[str, Any]:
        """Get a specific problem by index."""
        if self.dataset is None:
            self.load()
        return dict(self.dataset[idx])

    def __len__(self) -> int:
        if self.dataset is None:
            return 0
        return len(self.dataset)

    def __iter__(self):
        if self.dataset is None:
            self.load()
        for problem in self.dataset:
            yield dict(problem)


class FilteredMATHDataset:
    """
    MATH dataset filtered for DPO-wrong, RLVR-right problems.

    This is the core dataset for our analysis: problems where the
    DPO model fails but the final RLVR model succeeds.
    """

    def __init__(
        self,
        base_dataset: MATHDataset,
        dpo_results_path: Optional[str] = None,
        rlvr_results_path: Optional[str] = None,
    ):
        """
        Initialize filtered dataset.

        Args:
            base_dataset: Base MATH dataset instance
            dpo_results_path: Path to DPO model evaluation results
            rlvr_results_path: Path to RLVR model evaluation results
        """
        self.base_dataset = base_dataset
        self.dpo_results_path = dpo_results_path
        self.rlvr_results_path = rlvr_results_path
        self.filtered_problems = []

    def filter_by_model_performance(
        self,
        dpo_results: Dict[str, bool],
        rlvr_results: Dict[str, bool],
    ) -> List[Dict[str, Any]]:
        """
        Filter problems where DPO fails but RLVR succeeds.

        Args:
            dpo_results: Dict mapping problem_id to correctness (True/False)
            rlvr_results: Dict mapping problem_id to correctness (True/False)

        Returns:
            List of filtered problem dictionaries
        """
        self.filtered_problems = []

        for problem in tqdm(self.base_dataset, desc="Filtering problems"):
            problem_id = problem.get("unique_id", problem.get("problem"))

            dpo_correct = dpo_results.get(problem_id, True)  # Default to True (skip)
            rlvr_correct = rlvr_results.get(problem_id, False)  # Default to False (skip)

            # Keep problems where DPO fails and RLVR succeeds
            if not dpo_correct and rlvr_correct:
                problem["dpo_correct"] = False
                problem["rlvr_correct"] = True
                self.filtered_problems.append(problem)

        logger.info(
            f"Filtered to {len(self.filtered_problems)} problems "
            f"(DPO-wrong, RLVR-right)"
        )
        return self.filtered_problems

    def load_from_cache(self, cache_path: str) -> List[Dict[str, Any]]:
        """Load pre-filtered problems from cache."""
        with open(cache_path, "r") as f:
            self.filtered_problems = json.load(f)
        logger.info(f"Loaded {len(self.filtered_problems)} filtered problems from cache")
        return self.filtered_problems

    def save_to_cache(self, cache_path: str) -> None:
        """Save filtered problems to cache."""
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(self.filtered_problems, f, indent=2)
        logger.info(f"Saved {len(self.filtered_problems)} filtered problems to {cache_path}")

    def __len__(self) -> int:
        return len(self.filtered_problems)

    def __iter__(self):
        for problem in self.filtered_problems:
            yield problem

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.filtered_problems[idx]
