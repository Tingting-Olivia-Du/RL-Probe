"""
Spike Detection

Identify critical tokens where KL divergence indicates significant
distribution shifts (logical correction points).
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class Spike:
    """Represents a detected KL divergence spike."""

    position: int  # Token position
    value: float  # KL divergence value
    zscore: float  # Z-score of the spike
    token_id: Optional[int] = None
    token_string: Optional[str] = None
    region: Optional[str] = None  # "early", "mid", "late"
    category: Optional[str] = None  # Token category


@dataclass
class SpikeAnalysis:
    """Container for spike detection results."""

    spikes: List[Spike]
    kl_values: np.ndarray
    threshold: float
    mean_kl: float
    std_kl: float
    spike_density: Dict[str, float]  # Density by region


class SpikeDetector:
    """
    Detects significant KL divergence spikes in token sequences.

    Identifies critical tokens where the model's distribution shifts
    significantly, indicating potential logical correction points.
    """

    def __init__(
        self,
        method: str = "zscore",
        threshold: float = 2.0,
        min_spike_distance: int = 3,
        percentile: float = 95.0,
    ):
        """
        Initialize spike detector.

        Args:
            method: Detection method ("zscore", "percentile", "adaptive")
            threshold: Threshold for spike detection (z-score or multiplier)
            min_spike_distance: Minimum tokens between spikes
            percentile: Percentile for percentile-based detection
        """
        self.method = method
        self.threshold = threshold
        self.min_spike_distance = min_spike_distance
        self.percentile = percentile

    def detect_spikes(
        self,
        kl_values: Union[torch.Tensor, np.ndarray],
        token_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
        token_strings: Optional[List[str]] = None,
        prompt_length: int = 0,
    ) -> SpikeAnalysis:
        """
        Detect spikes in KL divergence values.

        Args:
            kl_values: Token-level KL divergence values
            token_ids: Optional token IDs
            token_strings: Optional token strings
            prompt_length: Length of prompt to exclude

        Returns:
            SpikeAnalysis with detected spikes
        """
        # Convert to numpy (handle BFloat16 by converting to float32 first)
        if isinstance(kl_values, torch.Tensor):
            # Convert BFloat16 to float32 before numpy conversion
            if kl_values.dtype == torch.bfloat16:
                kl_values = kl_values.float()
            kl_values = kl_values.cpu().numpy()
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()

        # Exclude prompt region
        response_kl = kl_values[prompt_length:]
        response_length = len(response_kl)

        if response_length == 0:
            return SpikeAnalysis(
                spikes=[],
                kl_values=kl_values,
                threshold=self.threshold,
                mean_kl=0.0,
                std_kl=0.0,
                spike_density={},
            )

        # Compute statistics
        mean_kl = np.mean(response_kl)
        std_kl = np.std(response_kl)

        # Detect spikes based on method
        if self.method == "zscore":
            spike_mask = self._detect_zscore(response_kl, mean_kl, std_kl)
            effective_threshold = mean_kl + self.threshold * std_kl
        elif self.method == "percentile":
            spike_mask = self._detect_percentile(response_kl)
            effective_threshold = np.percentile(response_kl, self.percentile)
        elif self.method == "adaptive":
            spike_mask = self._detect_adaptive(response_kl)
            effective_threshold = self._compute_adaptive_threshold(response_kl)
        else:
            raise ValueError(f"Unknown detection method: {self.method}")

        # Apply minimum distance constraint
        spike_mask = self._apply_min_distance(spike_mask, response_kl)

        # Build spike objects
        spikes = []
        spike_positions = np.where(spike_mask)[0]

        for pos in spike_positions:
            abs_pos = pos + prompt_length  # Absolute position
            value = float(response_kl[pos])
            zscore = (value - mean_kl) / std_kl if std_kl > 0 else 0.0

            spike = Spike(
                position=int(abs_pos),
                value=value,
                zscore=zscore,
                token_id=int(token_ids[abs_pos]) if token_ids is not None else None,
                token_string=token_strings[abs_pos] if token_strings else None,
                region=self._get_region(pos, response_length),
            )
            spikes.append(spike)

        # Compute spike density by region
        spike_density = self._compute_spike_density(spike_positions, response_length)

        # Convert numpy types to Python native types for JSON serialization
        return SpikeAnalysis(
            spikes=spikes,
            kl_values=kl_values,
            threshold=float(effective_threshold),
            mean_kl=float(mean_kl),
            std_kl=float(std_kl),
            spike_density=spike_density,  # This is already a dict with float values
        )

    def _detect_zscore(
        self,
        values: np.ndarray,
        mean: float,
        std: float,
    ) -> np.ndarray:
        """Detect spikes using z-score threshold."""
        if std == 0:
            return np.zeros(len(values), dtype=bool)
        zscores = (values - mean) / std
        return zscores > self.threshold

    def _detect_percentile(self, values: np.ndarray) -> np.ndarray:
        """Detect spikes using percentile threshold."""
        threshold = np.percentile(values, self.percentile)
        return values > threshold

    def _detect_adaptive(self, values: np.ndarray) -> np.ndarray:
        """
        Adaptive spike detection using local statistics.

        Uses sliding window to compute local mean and std,
        detecting points that are outliers locally.
        """
        window_size = max(10, len(values) // 10)
        spike_mask = np.zeros(len(values), dtype=bool)

        for i in range(len(values)):
            # Define local window
            start = max(0, i - window_size // 2)
            end = min(len(values), i + window_size // 2)
            window = values[start:end]

            local_mean = np.mean(window)
            local_std = np.std(window)

            if local_std > 0:
                local_zscore = (values[i] - local_mean) / local_std
                if local_zscore > self.threshold:
                    spike_mask[i] = True

        return spike_mask

    def _compute_adaptive_threshold(self, values: np.ndarray) -> float:
        """Compute adaptive threshold."""
        # Use median + MAD-based threshold for robustness
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        return median + self.threshold * 1.4826 * mad

    def _apply_min_distance(
        self,
        spike_mask: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        """
        Apply minimum distance constraint between spikes.

        Keeps the highest spike when multiple are too close.
        """
        spike_positions = np.where(spike_mask)[0]

        if len(spike_positions) <= 1:
            return spike_mask

        # Sort by value (descending)
        sorted_indices = sorted(
            spike_positions,
            key=lambda x: values[x],
            reverse=True,
        )

        # Greedily select spikes with minimum distance
        selected = []
        for pos in sorted_indices:
            if all(abs(pos - s) >= self.min_spike_distance for s in selected):
                selected.append(pos)

        # Create new mask
        new_mask = np.zeros_like(spike_mask)
        new_mask[selected] = True

        return new_mask

    def _get_region(self, position: int, total_length: int) -> str:
        """Classify position into early/mid/late region."""
        relative_pos = position / total_length
        if relative_pos < 0.33:
            return "early"
        elif relative_pos < 0.67:
            return "mid"
        else:
            return "late"

    def _compute_spike_density(
        self,
        spike_positions: np.ndarray,
        total_length: int,
    ) -> Dict[str, float]:
        """Compute spike density by region."""
        regions = {"early": 0, "mid": 0, "late": 0}
        region_sizes = {
            "early": total_length // 3,
            "mid": total_length // 3,
            "late": total_length - 2 * (total_length // 3),
        }

        for pos in spike_positions:
            region = self._get_region(pos, total_length)
            regions[region] += 1

        # Convert to density (ensure Python float, not numpy float)
        density = {}
        for region, count in regions.items():
            size = region_sizes[region]
            density[region] = float(count / size if size > 0 else 0.0)

        return density


class SpikeStatistics:
    """Compute aggregate statistics across multiple spike analyses."""

    @staticmethod
    def aggregate_spike_density(
        analyses: List[SpikeAnalysis],
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate spike density statistics.

        Returns:
            Dictionary with mean and std for each region
        """
        densities = {"early": [], "mid": [], "late": []}

        for analysis in analyses:
            for region, density in analysis.spike_density.items():
                densities[region].append(density)

        result = {}
        for region, values in densities.items():
            if values:
                result[region] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "median": np.median(values),
                }
            else:
                result[region] = {"mean": 0.0, "std": 0.0, "median": 0.0}

        return result

    @staticmethod
    def compare_checkpoints(
        analyses_by_checkpoint: Dict[str, List[SpikeAnalysis]],
    ) -> Dict[str, Any]:
        """
        Compare spike statistics across checkpoints.

        Args:
            analyses_by_checkpoint: Dict mapping checkpoint names to analyses

        Returns:
            Comparison statistics
        """
        comparison = {}

        for ckpt_name, analyses in analyses_by_checkpoint.items():
            mean_kls = [a.mean_kl for a in analyses]
            spike_counts = [len(a.spikes) for a in analyses]

            comparison[ckpt_name] = {
                "mean_kl": {
                    "mean": np.mean(mean_kls),
                    "std": np.std(mean_kls),
                },
                "spike_count": {
                    "mean": np.mean(spike_counts),
                    "std": np.std(spike_counts),
                },
                "spike_density": SpikeStatistics.aggregate_spike_density(analyses),
            }

        return comparison
