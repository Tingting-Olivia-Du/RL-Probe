"""
MATH Dataset Loader and Processor

Handles loading, filtering, and preprocessing of MATH dataset
for KL-divergence analysis experiments.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Available subjects in hendrycks_math dataset
MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


class MATHDataset:
    """
    MATH Dataset wrapper with filtering capabilities.

    Filters problems based on difficulty level and model performance
    (DPO-wrong, RLVR-right) for targeted analysis.
    """

    def __init__(
        self,
        subjects: List[str] = None,
        levels: List[int] = [3, 4],
        split: str = "test",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize MATH dataset loader.

        Args:
            subjects: List of subjects to load, or ["all"] for all subjects
            levels: Difficulty levels to include (1-5)
            split: Dataset split to use
            cache_dir: Directory for caching downloaded data
        """
        # Handle "all" subjects
        if subjects is None or subjects == ["all"]:
            self.subjects = MATH_SUBJECTS
        else:
            self.subjects = subjects

        self.levels = levels
        self.split = split
        self.cache_dir = cache_dir
        self.dataset = None
        self.filtered_problems = []

    def load(self) -> "MATHDataset":
        """Load the MATH dataset from HuggingFace."""
        logger.info(f"Loading MATH dataset (split={self.split}, subjects={self.subjects}, levels={self.levels})")

        # Load the entire dataset (MATH-500 only has 'default' config)
        logger.info("Loading MATH-500 dataset...")
        try:
            self.dataset = load_dataset(
                "HuggingFaceH4/MATH-500",
                split=self.split,
                cache_dir=self.cache_dir,
            )
        except Exception as e:
            logger.warning(f"Failed to load HuggingFaceH4/MATH-500: {e}")
            logger.info("Trying to load with 'default' config...")
            self.dataset = load_dataset(
                "HuggingFaceH4/MATH-500",
                "default",
                split=self.split,
                cache_dir=self.cache_dir,
            )

        logger.info(f"Loaded raw dataset: {len(self.dataset)} problems")
        
        # Debug: Check first example to understand structure
        if len(self.dataset) > 0:
            first_example = self.dataset[0]
            logger.info(f"Dataset features: {list(first_example.keys())}")
            logger.info(f"First example level: {first_example.get('level', 'N/A')}")
            logger.info(f"First example subject: {first_example.get('subject', 'N/A')}")

        # Filter by subject if specified (and not loading all subjects)
        if set(self.subjects) != set(MATH_SUBJECTS):  # If not loading all subjects
            logger.info(f"Filtering by subjects: {self.subjects}")
            # Normalize subject names (handle variations)
            normalized_subjects = [s.lower() for s in self.subjects]
            
            def subject_filter(x):
                subject_field = x.get("subject", "").lower()
                return subject_field in normalized_subjects
            
            self.dataset = self.dataset.filter(subject_filter)
            logger.info(f"After subject filtering: {len(self.dataset)} problems")

        # Filter by difficulty level - handle multiple possible formats
        if len(self.dataset) > 0:
            # Check level format from first example
            first_level = str(self.dataset[0].get("level", ""))
            logger.info(f"Level format example: '{first_level}'")
            
            # Try different level formats
            level_formats = [
                [f"Level {l}" for l in self.levels],  # "Level 3", "Level 4"
                [str(l) for l in self.levels],  # "3", "4"
                [f"level {l}" for l in self.levels],  # "level 3", "level 4"
            ]
            
            def level_filter(x):
                level_field = str(x.get("level", ""))
                for format_list in level_formats:
                    if level_field in format_list:
                        return True
                # Also try numeric comparison
                try:
                    level_num = int(level_field.replace("Level", "").replace("level", "").strip())
                    return level_num in self.levels
                except:
                    pass
                return False
            
            before_level_filter = len(self.dataset)
            self.dataset = self.dataset.filter(level_filter)
            logger.info(f"After level filtering ({self.levels}): {before_level_filter} -> {len(self.dataset)} problems")
        else:
            logger.warning("Dataset is empty before level filtering!")

        logger.info(f"Final loaded: {len(self.dataset)} problems at levels {self.levels}")
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
            # 使用与 01_prepare_data.py 相同的 ID 生成逻辑
            problem_id = problem.get("unique_id") or f"hash_{abs(hash(problem.get('problem', '')))}"

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
