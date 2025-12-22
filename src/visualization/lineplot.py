"""
Line Plot Visualization

Generate line plots showing token-level KL divergence with spike annotations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class KLLinePlotter:
    """
    Generate line plots for KL divergence visualization.

    Shows token-by-token progression with spike highlighting.
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (14, 6),
        dpi: int = 300,
        spike_color: str = "red",
        line_alpha: float = 0.8,
    ):
        """
        Initialize line plotter.

        Args:
            figsize: Figure size (width, height)
            dpi: DPI for saved figures
            spike_color: Color for spike markers
            line_alpha: Alpha for main line
        """
        self.figsize = figsize
        self.dpi = dpi
        self.spike_color = spike_color
        self.line_alpha = line_alpha

    def plot_single_sequence(
        self,
        kl_values: np.ndarray,
        prompt_length: int = 0,
        spike_positions: Optional[List[int]] = None,
        threshold: Optional[float] = None,
        title: str = "Token-Level KL Divergence",
        xlabel: str = "Token Position",
        ylabel: str = "KL Divergence",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot line graph for a single sequence.

        Args:
            kl_values: Token-level KL divergence values
            prompt_length: Prompt length to exclude
            spike_positions: Positions to highlight as spikes
            threshold: Optional threshold line
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot response region only
        response_kl = kl_values[prompt_length:]
        positions = np.arange(len(response_kl))

        # Main line
        ax.plot(
            positions,
            response_kl,
            color="steelblue",
            alpha=self.line_alpha,
            linewidth=1.5,
            label="KL Divergence",
        )

        # Fill under curve
        ax.fill_between(
            positions,
            0,
            response_kl,
            color="steelblue",
            alpha=0.2,
        )

        # Add spike markers
        if spike_positions:
            adjusted_positions = [p - prompt_length for p in spike_positions if p >= prompt_length]
            spike_values = [response_kl[p] for p in adjusted_positions if p < len(response_kl)]

            ax.scatter(
                adjusted_positions[:len(spike_values)],
                spike_values,
                color=self.spike_color,
                s=100,
                zorder=5,
                label="Detected Spikes",
                marker="^",
            )

        # Add threshold line
        if threshold is not None:
            ax.axhline(
                y=threshold,
                color="orange",
                linestyle="--",
                alpha=0.7,
                label=f"Threshold ({threshold:.3f})",
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

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
        Plot overlaid lines for multiple checkpoints.

        Args:
            kl_by_checkpoint: Dict mapping checkpoint names to KL values
            prompt_length: Prompt length
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(kl_by_checkpoint)))

        for (name, kl_values), color in zip(kl_by_checkpoint.items(), colors):
            response_kl = kl_values[prompt_length:]
            positions = np.arange(len(response_kl))

            ax.plot(
                positions,
                response_kl,
                color=color,
                alpha=self.line_alpha,
                linewidth=1.5,
                label=name,
            )

        ax.set_xlabel("Token Position (Response)")
        ax.set_ylabel("KL Divergence")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_with_entropy(
        self,
        kl_values: np.ndarray,
        entropy_values: np.ndarray,
        prompt_length: int = 0,
        title: str = "KL Divergence and Entropy",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot KL divergence alongside entropy.

        Args:
            kl_values: KL divergence values
            entropy_values: Entropy values
            prompt_length: Prompt length
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5), sharex=True)

        response_kl = kl_values[prompt_length:]
        response_entropy = entropy_values[prompt_length:]
        positions = np.arange(len(response_kl))

        # KL plot
        ax1.plot(positions, response_kl, color="steelblue", alpha=self.line_alpha, linewidth=1.5)
        ax1.fill_between(positions, 0, response_kl, color="steelblue", alpha=0.2)
        ax1.set_ylabel("KL Divergence")
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)

        # Entropy plot
        ax2.plot(positions, response_entropy, color="forestgreen", alpha=self.line_alpha, linewidth=1.5)
        ax2.fill_between(positions, 0, response_entropy, color="forestgreen", alpha=0.2)
        ax2.set_xlabel("Token Position (Response)")
        ax2.set_ylabel("Entropy")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_region_breakdown(
        self,
        kl_values: np.ndarray,
        prompt_length: int = 0,
        title: str = "KL Divergence by Region",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot with region annotations (early/mid/late).

        Args:
            kl_values: KL divergence values
            prompt_length: Prompt length
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        response_kl = kl_values[prompt_length:]
        positions = np.arange(len(response_kl))
        length = len(response_kl)
        third = length // 3

        # Plot main line
        ax.plot(positions, response_kl, color="steelblue", alpha=self.line_alpha, linewidth=1.5)

        # Color regions
        colors = ["lightcoral", "lightyellow", "lightgreen"]
        labels = ["Early", "Mid", "Late"]
        regions = [(0, third), (third, 2 * third), (2 * third, length)]

        for (start, end), color, label in zip(regions, colors, labels):
            ax.axvspan(start, end, alpha=0.2, color=color, label=f"{label} Region")

        ax.set_xlabel("Token Position (Response)")
        ax.set_ylabel("KL Divergence")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

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
