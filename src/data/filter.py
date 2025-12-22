"""
Problem Filtering Utilities

Filter MATH problems based on model performance and other criteria.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ProblemFilter:
    """
    Filter problems based on model evaluation results.

    Identifies problems where:
    - DPO model produces incorrect answers
    - RLVR model produces correct answers
    """

    def __init__(self, answer_extraction_pattern: Optional[str] = None):
        """
        Initialize the problem filter.

        Args:
            answer_extraction_pattern: Regex pattern for extracting answers
        """
        # Default pattern matches \\boxed{...} format used in MATH
        self.answer_pattern = answer_extraction_pattern or r"\\boxed\{([^}]+)\}"

    def extract_answer(self, response: str) -> Optional[str]:
        """
        Extract the final answer from a model response.

        Args:
            response: Model's generated response

        Returns:
            Extracted answer string or None
        """
        matches = re.findall(self.answer_pattern, response)
        if matches:
            return matches[-1].strip()  # Return last match (final answer)
        return None

    def normalize_answer(self, answer: str) -> str:
        """
        Normalize answer for comparison.

        Handles common variations in mathematical notation.
        """
        if answer is None:
            return ""

        # Remove whitespace
        answer = answer.strip()

        # Normalize common mathematical expressions
        answer = answer.replace(" ", "")
        answer = answer.replace("\\frac", "frac")
        answer = answer.replace("\\sqrt", "sqrt")
        answer = answer.replace("\\pi", "pi")
        answer = answer.replace("\\cdot", "*")

        return answer.lower()

    def check_correctness(
        self,
        model_response: str,
        ground_truth: str,
    ) -> bool:
        """
        Check if model response matches ground truth.

        Args:
            model_response: Model's generated response
            ground_truth: Ground truth answer

        Returns:
            True if answers match, False otherwise
        """
        extracted = self.extract_answer(model_response)
        if extracted is None:
            return False

        norm_extracted = self.normalize_answer(extracted)
        norm_truth = self.normalize_answer(ground_truth)

        return norm_extracted == norm_truth

    def evaluate_model_responses(
        self,
        problems: List[Dict[str, Any]],
        responses: List[str],
    ) -> Dict[str, bool]:
        """
        Evaluate a batch of model responses.

        Args:
            problems: List of problem dictionaries with ground truth
            responses: List of model responses

        Returns:
            Dict mapping problem identifiers to correctness
        """
        results = {}

        for problem, response in zip(problems, responses):
            problem_id = problem.get("unique_id", str(hash(problem["problem"])))
            ground_truth = problem.get("solution", problem.get("answer", ""))

            # Extract ground truth answer
            gt_answer = self.extract_answer(ground_truth)
            if gt_answer is None:
                gt_answer = ground_truth  # Use raw if no boxed format

            is_correct = self.check_correctness(response, gt_answer)
            results[problem_id] = is_correct

        return results


class DifficultyAnalyzer:
    """Analyze problem difficulty based on various metrics."""

    @staticmethod
    def estimate_steps(solution: str) -> int:
        """
        Estimate number of reasoning steps in a solution.

        Args:
            solution: Ground truth solution text

        Returns:
            Estimated number of steps
        """
        # Count step indicators
        step_patterns = [
            r"step \d+",
            r"first|second|third|fourth|fifth",
            r"then|next|finally",
            r"therefore|thus|hence",
            r"we have|we get|we find",
        ]

        step_count = 0
        solution_lower = solution.lower()

        for pattern in step_patterns:
            step_count += len(re.findall(pattern, solution_lower))

        # Also count equation lines
        equation_count = solution.count("=")

        return max(step_count, equation_count // 2, 1)

    @staticmethod
    def categorize_problem(problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Categorize a problem by type and difficulty.

        Args:
            problem: Problem dictionary

        Returns:
            Dictionary with category information
        """
        problem_text = problem.get("problem", "")
        solution = problem.get("solution", "")

        categories = {
            "has_algebra": bool(re.search(r"solve|equation|variable", problem_text.lower())),
            "has_geometry": bool(re.search(r"triangle|circle|angle|area", problem_text.lower())),
            "has_probability": bool(re.search(r"probability|chance|random", problem_text.lower())),
            "has_combinatorics": bool(re.search(r"ways|arrange|choose|combination", problem_text.lower())),
            "has_number_theory": bool(re.search(r"divisible|prime|remainder|factor", problem_text.lower())),
            "estimated_steps": DifficultyAnalyzer.estimate_steps(solution),
            "solution_length": len(solution),
        }

        return categories
