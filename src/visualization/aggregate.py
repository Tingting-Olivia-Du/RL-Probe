"""
Aggregate Visualization

Generate visualizations for aggregate statistics across multiple problems.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


class AggregateVisualizer:
    """
    Generate aggregate visualizations across multiple analyses.

    Creates summary plots showing:
    - Mean KL by checkpoint
    - Spike density distributions
    - Category breakdown
    - Statistical comparisons
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 300,
    ):
        """
        Initialize aggregate visualizer.

        Args:
            figsize: Default figure size
            dpi: DPI for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_mean_kl_progression(
        self,
        checkpoint_stats: Dict[str, Dict[str, float]],
        title: str = "Mean KL Divergence Across Training",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot mean KL progression across checkpoints.

        Args:
            checkpoint_stats: Dict with checkpoint stats (mean, std)
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        checkpoints = list(checkpoint_stats.keys())
        means = [s["mean"] for s in checkpoint_stats.values()]
        stds = [s["std"] for s in checkpoint_stats.values()]

        x = np.arange(len(checkpoints))

        ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8, edgecolor="black")

        ax.set_xticks(x)
        ax.set_xticklabels(checkpoints, rotation=45, ha="right")
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Mean KL Divergence")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_spike_density_comparison(
        self,
        density_by_checkpoint: Dict[str, Dict[str, float]],
        title: str = "Spike Density by Region and Checkpoint",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot spike density comparison across checkpoints and regions.

        Args:
            density_by_checkpoint: Dict of checkpoint -> region -> density
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        checkpoints = list(density_by_checkpoint.keys())
        regions = ["early", "mid", "late"]
        x = np.arange(len(checkpoints))
        width = 0.25

        colors = {"early": "lightcoral", "mid": "lightyellow", "late": "lightgreen"}

        for i, region in enumerate(regions):
            densities = [density_by_checkpoint[ckpt].get(region, 0) for ckpt in checkpoints]
            ax.bar(x + i * width, densities, width, label=region.capitalize(), color=colors[region], edgecolor="black")

        ax.set_xticks(x + width)
        ax.set_xticklabels(checkpoints, rotation=45, ha="right")
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Spike Density")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_category_kl_breakdown(
        self,
        category_stats: Dict[str, Dict[str, float]],
        title: str = "Mean KL by Token Category",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot KL divergence breakdown by token category.

        Args:
            category_stats: Dict mapping categories to stats
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Sort by mean KL
        sorted_cats = sorted(
            category_stats.items(),
            key=lambda x: x[1].get("overall_mean_kl", 0),
            reverse=True,
        )

        categories = [c[0] for c in sorted_cats]
        means = [c[1].get("overall_mean_kl", 0) for c in sorted_cats]
        stds = [c[1].get("overall_std_kl", 0) for c in sorted_cats]

        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(categories)))

        ax.barh(categories, means, xerr=stds, capsize=3, color=colors, edgecolor="black")

        ax.set_xlabel("Mean KL Divergence")
        ax.set_ylabel("Token Category")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_entropy_reduction(
        self,
        entropy_progression: Dict[str, float],
        title: str = "Entropy Reduction Across Training",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot entropy reduction across checkpoints.

        Args:
            entropy_progression: Dict mapping checkpoints to mean entropy
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        checkpoints = list(entropy_progression.keys())
        entropies = list(entropy_progression.values())

        ax.plot(checkpoints, entropies, "o-", color="forestgreen", linewidth=2, markersize=10)
        ax.fill_between(checkpoints, entropies, alpha=0.2, color="forestgreen")

        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Mean Entropy")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Add percentage reduction annotation
        if len(entropies) >= 2:
            reduction = (entropies[0] - entropies[-1]) / entropies[0] * 100
            ax.annotate(
                f"{reduction:.1f}% reduction",
                xy=(checkpoints[-1], entropies[-1]),
                xytext=(10, 20),
                textcoords="offset points",
                fontsize=11,
                fontweight="bold",
                color="forestgreen",
            )

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_correlation_with_correctness(
        self,
        kl_values: List[float],
        correctness: List[bool],
        title: str = "KL Divergence vs Correctness",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot correlation between KL divergence and answer correctness.

        Args:
            kl_values: Mean KL values per problem
            correctness: Whether each problem was answered correctly
            title: Plot title
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        correct_kl = [kl for kl, c in zip(kl_values, correctness) if c]
        incorrect_kl = [kl for kl, c in zip(kl_values, correctness) if not c]

        # Box plot comparison
        bp = ax.boxplot(
            [correct_kl, incorrect_kl],
            labels=["Correct", "Incorrect"],
            patch_artist=True,
        )

        colors = ["lightgreen", "lightcoral"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax.set_ylabel("Mean KL Divergence")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

        # Add statistical test
        if len(correct_kl) > 0 and len(incorrect_kl) > 0:
            statistic, p_value = scipy_stats.mannwhitneyu(correct_kl, incorrect_kl, alternative="two-sided")
            ax.annotate(
                f"Mann-Whitney U p={p_value:.4f}",
                xy=(0.5, 0.95),
                xycoords="axes fraction",
                ha="center",
                fontsize=10,
            )

        plt.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_summary_dashboard(
        self,
        checkpoint_means: Dict[str, float],
        spike_densities: Dict[str, Dict[str, float]],
        category_stats: Dict[str, Dict[str, float]],
        title: str = "Analysis Summary Dashboard",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a summary dashboard with multiple panels.

        Args:
            checkpoint_means: Mean KL by checkpoint
            spike_densities: Spike density by checkpoint and region
            category_stats: Category statistics
            title: Overall title
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: Mean KL progression
        ax1 = axes[0, 0]
        checkpoints = list(checkpoint_means.keys())
        means = list(checkpoint_means.values())
        ax1.bar(checkpoints, means, color="steelblue", edgecolor="black")
        ax1.set_title("Mean KL by Checkpoint")
        ax1.set_ylabel("KL Divergence")
        ax1.tick_params(axis="x", rotation=45)

        # Panel 2: Spike density
        ax2 = axes[0, 1]
        regions = ["early", "mid", "late"]
        x = np.arange(len(checkpoints))
        width = 0.25
        for i, region in enumerate(regions):
            densities = [spike_densities.get(ckpt, {}).get(region, 0) for ckpt in checkpoints]
            ax2.bar(x + i * width, densities, width, label=region.capitalize())
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(checkpoints, rotation=45)
        ax2.set_title("Spike Density by Region")
        ax2.legend()

        # Panel 3: Top categories by KL
        ax3 = axes[1, 0]
        sorted_cats = sorted(category_stats.items(), key=lambda x: x[1].get("overall_mean_kl", 0), reverse=True)[:6]
        cat_names = [c[0] for c in sorted_cats]
        cat_means = [c[1].get("overall_mean_kl", 0) for c in sorted_cats]
        ax3.barh(cat_names, cat_means, color="coral", edgecolor="black")
        ax3.set_title("Top Categories by KL")
        ax3.set_xlabel("Mean KL")

        # Panel 4: Summary text
        ax4 = axes[1, 1]
        ax4.axis("off")

        total_problems = sum(c.get("total_count", 0) for c in category_stats.values())
        overall_mean = np.mean(means) if means else 0

        summary_text = f"""
Summary Statistics
{'='*30}

Total Tokens Analyzed: {total_problems:,}
Overall Mean KL: {overall_mean:.4f}

Highest KL Category: {cat_names[0] if cat_names else 'N/A'}
Lowest KL Category: {sorted_cats[-1][0] if sorted_cats else 'N/A'}

Training Progression:
  Start: {means[0]:.4f} (KL)
  End:   {means[-1]:.4f} (KL)
  Change: {(means[-1] - means[0]) / means[0] * 100:.1f}%
"""
        ax4.text(0.1, 0.9, summary_text, fontsize=11, transform=ax4.transAxes, verticalalignment="top", family="monospace")

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

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
