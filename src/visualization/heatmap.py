"""
Heatmap Visualization

Generate heatmaps showing token-level KL divergence patterns.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


class KLHeatmapPlotter:
    """
    Generate heatmaps for KL divergence visualization.

    Shows token-level KL divergence patterns across:
    - Single sequences
    - Multiple checkpoints
    - Comparative views (DPO vs RLVR stages)
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (16, 8),
        cmap: str = "RdYlBu_r",
        dpi: int = 300,
    ):
        """
        Initialize heatmap plotter.

        Args:
            figsize: Figure size (width, height)
            cmap: Colormap for heatmaps
            dpi: DPI for saved figures
        """
        self.figsize = figsize
        self.cmap = cmap
        self.dpi = dpi

    def plot_single_sequence(
        self,
        kl_values: np.ndarray,
        token_strings: Optional[List[str]] = None,
        prompt_length: int = 0,
        title: str = "Token-Level KL Divergence",
        spike_positions: Optional[List[int]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot heatmap for a single sequence.

        Args:
            kl_values: Token-level KL divergence values
            token_strings: Optional token labels
            prompt_length: Prompt length for annotation
            title: Plot title
            spike_positions: Positions to highlight as spikes
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Reshape for heatmap (1 row)
        data = kl_values[prompt_length:].reshape(1, -1)

        # Create heatmap
        sns.heatmap(
            data,
            ax=ax,
            cmap=self.cmap,
            cbar_kws={"label": "KL Divergence"},
            xticklabels=50,  # Show every 50th tick
            yticklabels=False,
        )

        # Add spike markers
        if spike_positions:
            adjusted_positions = [p - prompt_length for p in spike_positions if p >= prompt_length]
            for pos in adjusted_positions:
                ax.axvline(x=pos, color="red", linestyle="--", alpha=0.7, linewidth=1)

        ax.set_xlabel("Token Position (Response)")
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_checkpoint_comparison(
        self,
        kl_by_checkpoint: Dict[str, np.ndarray],
        prompt_length: int = 0,
        title: str = "KL Divergence Across Checkpoints",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot comparative heatmap across checkpoints.

        Args:
            kl_by_checkpoint: Dict mapping checkpoint names to KL values
            prompt_length: Prompt length
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        checkpoint_names = list(kl_by_checkpoint.keys())
        n_checkpoints = len(checkpoint_names)

        # Find max length
        max_len = max(len(v) - prompt_length for v in kl_by_checkpoint.values())

        # Create data matrix (pad shorter sequences)
        data = np.zeros((n_checkpoints, max_len))
        for i, name in enumerate(checkpoint_names):
            values = kl_by_checkpoint[name][prompt_length:]
            data[i, : len(values)] = values

        fig, ax = plt.subplots(figsize=self.figsize)

        sns.heatmap(
            data,
            ax=ax,
            cmap=self.cmap,
            cbar_kws={"label": "KL Divergence"},
            xticklabels=50,
            yticklabels=checkpoint_names,
        )

        ax.set_xlabel("Token Position (Response)")
        ax.set_ylabel("Checkpoint")
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_problem_batch(
        self,
        kl_values_list: List[np.ndarray],
        problem_ids: List[str],
        prompt_lengths: List[int],
        title: str = "KL Divergence Across Problems",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot heatmap for multiple problems.

        Args:
            kl_values_list: List of KL values for each problem
            problem_ids: Problem identifiers
            prompt_lengths: Prompt lengths for each problem
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        n_problems = len(kl_values_list)

        # Find max response length
        max_len = max(
            len(kl) - pl for kl, pl in zip(kl_values_list, prompt_lengths)
        )

        # Create data matrix
        data = np.zeros((n_problems, max_len))
        for i, (kl_values, prompt_len) in enumerate(zip(kl_values_list, prompt_lengths)):
            response_kl = kl_values[prompt_len:]
            data[i, : len(response_kl)] = response_kl

        # Truncate problem IDs for display
        display_ids = [pid[:20] + "..." if len(pid) > 20 else pid for pid in problem_ids]

        fig, ax = plt.subplots(figsize=(self.figsize[0], max(8, n_problems * 0.5)))

        sns.heatmap(
            data,
            ax=ax,
            cmap=self.cmap,
            cbar_kws={"label": "KL Divergence"},
            xticklabels=50,
            yticklabels=display_ids,
        )

        ax.set_xlabel("Token Position (Response)")
        ax.set_ylabel("Problem")
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_evolution_matrix(
        self,
        checkpoint_kl_matrix: np.ndarray,
        checkpoint_names: List[str],
        token_positions: Optional[np.ndarray] = None,
        title: str = "KL Divergence Evolution",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot evolution of KL divergence across training.

        Args:
            checkpoint_kl_matrix: Matrix [checkpoints, tokens]
            checkpoint_names: Names for each checkpoint
            token_positions: Optional specific positions to label
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        im = ax.imshow(
            checkpoint_kl_matrix,
            aspect="auto",
            cmap=self.cmap,
            interpolation="nearest",
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("KL Divergence")

        # Labels
        ax.set_yticks(range(len(checkpoint_names)))
        ax.set_yticklabels(checkpoint_names)
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Training Checkpoint")
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def _save_figure(
        self,
        fig: plt.Figure,
        save_path: str,
    ) -> None:
        """Save figure to file."""
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        logger.info(f"Saved figure to {save_path}")
