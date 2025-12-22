"""
Token Classification

Classify tokens by type (math symbols, numbers, logical connectors, etc.)
to understand what types of tokens show the most KL divergence.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TokenCategory:
    """Definition of a token category."""

    name: str
    patterns: List[str]  # Regex patterns
    keywords: Set[str]  # Exact match keywords


@dataclass
class ClassifiedToken:
    """A token with its classification."""

    position: int
    token_string: str
    token_id: int
    category: str
    kl_value: Optional[float] = None


class TokenClassifier:
    """
    Classify tokens by semantic category.

    Categories:
    - math_symbol: Mathematical operators and symbols (+, -, *, /, =, etc.)
    - number: Numeric values and expressions
    - logical: Logical connectors (therefore, because, if, then)
    - step_marker: Step indicators (Step 1, First, Next, Finally)
    - variable: Mathematical variables (x, y, n, etc.)
    - punctuation: Punctuation marks
    - other: Everything else
    """

    DEFAULT_CATEGORIES = {
        "math_symbol": TokenCategory(
            name="math_symbol",
            patterns=[
                r"^[\+\-\*\/\=\<\>\^\(\)\[\]\{\}]+$",
                r"^\\(?:frac|sqrt|times|div|pm|mp|cdot|leq|geq|neq|approx).*$",
            ],
            keywords={"+", "-", "*", "/", "=", "<", ">", "^", "(", ")", "[", "]", "{", "}", "±", "×", "÷"},
        ),
        "number": TokenCategory(
            name="number",
            patterns=[
                r"^\d+\.?\d*$",  # Plain numbers
                r"^-?\d+\.?\d*$",  # Negative numbers
                r"^\d+/\d+$",  # Fractions
            ],
            keywords=set(),
        ),
        "logical": TokenCategory(
            name="logical",
            patterns=[],
            keywords={
                "therefore", "thus", "hence", "so", "because", "since",
                "if", "then", "when", "where", "given", "assuming",
                "implies", "follows", "means", "consequently",
                "however", "but", "although", "unless",
            },
        ),
        "step_marker": TokenCategory(
            name="step_marker",
            patterns=[
                r"^step\s*\d+:?$",
                r"^(?:first|second|third|fourth|fifth|finally|next|then|now|lastly):?$",
            ],
            keywords={
                "step", "first", "second", "third", "fourth", "fifth",
                "next", "then", "finally", "lastly", "now",
            },
        ),
        "variable": TokenCategory(
            name="variable",
            patterns=[
                r"^[a-zA-Z]$",  # Single letters
                r"^[a-zA-Z]_\d+$",  # Subscripted variables
            ],
            keywords={"x", "y", "z", "n", "m", "k", "a", "b", "c", "t", "r", "s"},
        ),
        "punctuation": TokenCategory(
            name="punctuation",
            patterns=[r"^[.,;:!?]+$"],
            keywords={".", ",", ";", ":", "!", "?", "'", '"'},
        ),
    }

    def __init__(
        self,
        categories: Optional[Dict[str, TokenCategory]] = None,
    ):
        """
        Initialize token classifier.

        Args:
            categories: Custom category definitions (uses defaults if None)
        """
        self.categories = categories or self.DEFAULT_CATEGORIES

    def classify_token(self, token_string: str) -> str:
        """
        Classify a single token.

        Args:
            token_string: The token text

        Returns:
            Category name
        """
        token_lower = token_string.strip().lower()
        token_stripped = token_string.strip()

        for cat_name, category in self.categories.items():
            # Check keywords first (exact match)
            if token_lower in category.keywords:
                return cat_name

            # Check regex patterns
            for pattern in category.patterns:
                if re.match(pattern, token_stripped, re.IGNORECASE):
                    return cat_name

        return "other"

    def classify_sequence(
        self,
        token_strings: List[str],
        token_ids: Optional[List[int]] = None,
        kl_values: Optional[np.ndarray] = None,
    ) -> List[ClassifiedToken]:
        """
        Classify all tokens in a sequence.

        Args:
            token_strings: List of token strings
            token_ids: Optional list of token IDs
            kl_values: Optional KL divergence values

        Returns:
            List of ClassifiedToken objects
        """
        classified = []

        for i, token_str in enumerate(token_strings):
            category = self.classify_token(token_str)

            classified.append(ClassifiedToken(
                position=i,
                token_string=token_str,
                token_id=token_ids[i] if token_ids else 0,
                category=category,
                kl_value=float(kl_values[i]) if kl_values is not None else None,
            ))

        return classified

    def get_category_statistics(
        self,
        classified_tokens: List[ClassifiedToken],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute statistics by category.

        Args:
            classified_tokens: List of classified tokens

        Returns:
            Statistics for each category
        """
        category_data: Dict[str, List[float]] = {cat: [] for cat in self.categories}
        category_data["other"] = []

        for token in classified_tokens:
            if token.kl_value is not None:
                category_data[token.category].append(token.kl_value)

        statistics = {}
        for category, values in category_data.items():
            if values:
                statistics[category] = {
                    "count": len(values),
                    "mean_kl": np.mean(values),
                    "std_kl": np.std(values),
                    "max_kl": np.max(values),
                    "min_kl": np.min(values),
                }
            else:
                statistics[category] = {
                    "count": 0,
                    "mean_kl": 0.0,
                    "std_kl": 0.0,
                    "max_kl": 0.0,
                    "min_kl": 0.0,
                }

        return statistics

    def analyze_spikes_by_category(
        self,
        classified_tokens: List[ClassifiedToken],
        spike_positions: List[int],
    ) -> Dict[str, int]:
        """
        Count spikes by token category.

        Args:
            classified_tokens: List of classified tokens
            spike_positions: Positions of detected spikes

        Returns:
            Dict mapping categories to spike counts
        """
        spike_set = set(spike_positions)
        category_spikes = {cat: 0 for cat in self.categories}
        category_spikes["other"] = 0

        for token in classified_tokens:
            if token.position in spike_set:
                category_spikes[token.category] += 1

        return category_spikes


class CategoryAggregator:
    """Aggregate token category statistics across multiple analyses."""

    @staticmethod
    def aggregate_statistics(
        stats_list: List[Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate category statistics across multiple sequences.

        Args:
            stats_list: List of per-sequence statistics

        Returns:
            Aggregated statistics
        """
        all_categories = set()
        for stats in stats_list:
            all_categories.update(stats.keys())

        aggregated = {}
        for category in all_categories:
            counts = []
            mean_kls = []

            for stats in stats_list:
                if category in stats:
                    counts.append(stats[category]["count"])
                    if stats[category]["count"] > 0:
                        mean_kls.append(stats[category]["mean_kl"])

            aggregated[category] = {
                "total_count": sum(counts),
                "avg_count": np.mean(counts) if counts else 0,
                "overall_mean_kl": np.mean(mean_kls) if mean_kls else 0,
                "overall_std_kl": np.std(mean_kls) if mean_kls else 0,
            }

        return aggregated

    @staticmethod
    def rank_categories_by_kl(
        aggregated_stats: Dict[str, Dict[str, Any]],
    ) -> List[Tuple[str, float]]:
        """
        Rank categories by mean KL divergence.

        Args:
            aggregated_stats: Aggregated statistics

        Returns:
            List of (category, mean_kl) tuples, sorted descending
        """
        rankings = [
            (cat, stats["overall_mean_kl"])
            for cat, stats in aggregated_stats.items()
            if stats["total_count"] > 0
        ]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
