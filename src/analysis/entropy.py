"""
Entropy Analysis

Analyze entropy evolution across checkpoints to understand
how RLVR reduces uncertainty in reasoning.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


@dataclass
class EntropyResult:
    """Container for entropy analysis results."""

    entropy_values: np.ndarray  # Token-level entropy
    mean_entropy: float
    std_entropy: float

    # Normalized entropy (by log(vocab_size))
    normalized_entropy: Optional[np.ndarray] = None
    mean_normalized: Optional[float] = None

    # Region-wise statistics
    region_entropy: Optional[Dict[str, float]] = None


class EntropyAnalyzer:
    """
    Analyze entropy of model predictions.

    Lower entropy indicates higher confidence in predictions.
    Track how RLVR training reduces entropy (uncertainty) in
    mathematical reasoning steps.
    """

    def __init__(
        self,
        normalize: bool = True,
        vocab_size: Optional[int] = None,
    ):
        """
        Initialize entropy analyzer.

        Args:
            normalize: Whether to normalize entropy by log(vocab_size)
            vocab_size: Vocabulary size for normalization
        """
        self.normalize = normalize
        self.vocab_size = vocab_size

    def compute_entropy(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute entropy of logit distributions.

        H(P) = -sum_x P(x) * log(P(x))

        Args:
            logits: Model logits [batch, seq_len, vocab_size]

        Returns:
            Entropy values [batch, seq_len]
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # H = -sum(p * log(p))
        entropy = -(probs * log_probs).sum(dim=-1)

        return entropy

    def analyze_sequence(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_length: int = 0,
    ) -> EntropyResult:
        """
        Analyze entropy for a sequence.

        Args:
            model: Language model
            input_ids: Input token IDs
            attention_mask: Attention mask
            prompt_length: Prompt length to exclude

        Returns:
            EntropyResult with analysis
        """
        # Get logits
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        logits = outputs.logits

        # Compute entropy
        entropy = self.compute_entropy(logits)
        # Convert BFloat16 to float32 before numpy conversion
        if entropy.dtype == torch.bfloat16:
            entropy = entropy.float()
        entropy = entropy.squeeze().cpu().numpy()

        # Focus on response region
        response_entropy = entropy[prompt_length:]

        # Compute statistics
        mean_entropy = float(np.mean(response_entropy))
        std_entropy = float(np.std(response_entropy))

        # Normalize if requested
        normalized_entropy = None
        mean_normalized = None

        if self.normalize:
            vocab_size = self.vocab_size or logits.shape[-1]
            max_entropy = np.log(vocab_size)
            normalized_entropy = response_entropy / max_entropy
            mean_normalized = float(mean_entropy / max_entropy)  # Ensure Python float

        # Compute region-wise entropy
        region_entropy = self._compute_region_entropy(response_entropy)

        return EntropyResult(
            entropy_values=entropy,
            mean_entropy=mean_entropy,
            std_entropy=std_entropy,
            normalized_entropy=normalized_entropy,
            mean_normalized=mean_normalized,
            region_entropy=region_entropy,
        )

    def _compute_region_entropy(
        self,
        entropy_values: np.ndarray,
    ) -> Dict[str, float]:
        """Compute mean entropy by region (early/mid/late)."""
        length = len(entropy_values)
        if length == 0:
            return {"early": 0.0, "mid": 0.0, "late": 0.0}

        third = length // 3

        return {
            "early": float(np.mean(entropy_values[:third])) if third > 0 else 0.0,
            "mid": float(np.mean(entropy_values[third : 2 * third])) if third > 0 else 0.0,
            "late": float(np.mean(entropy_values[2 * third :])) if third > 0 else 0.0,
        }

    def compare_checkpoints(
        self,
        models: Dict[str, PreTrainedModel],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_length: int = 0,
    ) -> Dict[str, EntropyResult]:
        """
        Compare entropy across multiple checkpoints.

        Args:
            models: Dict mapping checkpoint names to models
            input_ids: Input token IDs
            attention_mask: Attention mask
            prompt_length: Prompt length

        Returns:
            Dict mapping checkpoint names to EntropyResult
        """
        results = {}

        for name, model in models.items():
            results[name] = self.analyze_sequence(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_length=prompt_length,
            )

        return results


class EntropyReductionAnalyzer:
    """Analyze entropy reduction patterns across training."""

    @staticmethod
    def compute_reduction_rate(
        entropy_early: EntropyResult,
        entropy_late: EntropyResult,
    ) -> Dict[str, float]:
        """
        Compute entropy reduction between checkpoints.

        Args:
            entropy_early: Entropy from earlier checkpoint
            entropy_late: Entropy from later checkpoint

        Returns:
            Dict with reduction metrics
        """
        mean_reduction = entropy_early.mean_entropy - entropy_late.mean_entropy
        relative_reduction = mean_reduction / entropy_early.mean_entropy if entropy_early.mean_entropy > 0 else 0.0

        # Region-wise reduction
        region_reduction = {}
        if entropy_early.region_entropy and entropy_late.region_entropy:
            for region in ["early", "mid", "late"]:
                early_val = entropy_early.region_entropy.get(region, 0)
                late_val = entropy_late.region_entropy.get(region, 0)
                region_reduction[region] = early_val - late_val

        return {
            "absolute_reduction": mean_reduction,
            "relative_reduction": relative_reduction,
            "region_reduction": region_reduction,
        }

    @staticmethod
    def analyze_progression(
        checkpoint_entropies: List[Tuple[int, EntropyResult]],
    ) -> Dict[str, Any]:
        """
        Analyze entropy progression across training steps.

        Args:
            checkpoint_entropies: List of (step, EntropyResult) tuples

        Returns:
            Progression analysis
        """
        # Sort by step
        sorted_data = sorted(checkpoint_entropies, key=lambda x: x[0])

        steps = [s for s, _ in sorted_data]
        mean_entropies = [e.mean_entropy for _, e in sorted_data]

        # Compute reduction rates between consecutive checkpoints
        reduction_rates = []
        for i in range(1, len(sorted_data)):
            prev_entropy = sorted_data[i - 1][1].mean_entropy
            curr_entropy = sorted_data[i][1].mean_entropy
            rate = (prev_entropy - curr_entropy) / prev_entropy if prev_entropy > 0 else 0
            reduction_rates.append(rate)

        return {
            "steps": steps,
            "mean_entropies": mean_entropies,
            "reduction_rates": reduction_rates,
            "total_reduction": mean_entropies[0] - mean_entropies[-1] if mean_entropies else 0,
            "total_relative_reduction": (mean_entropies[0] - mean_entropies[-1]) / mean_entropies[0]
            if mean_entropies and mean_entropies[0] > 0
            else 0,
        }
