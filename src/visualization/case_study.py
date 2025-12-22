"""
Case Study Visualization

Generate detailed visualizations for individual problem case studies.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from src.analysis.kl_divergence import KLResult
from src.analysis.spike_detector import SpikeAnalysis

logger = logging.getLogger(__name__)


class CaseStudyVisualizer:
    """
    Generate comprehensive case study visualizations.

    Creates detailed multi-panel figures showing:
    - KL divergence progression
    - Spike locations with token context
    - Checkpoint evolution
    - Token category breakdown
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (16, 12),
        dpi: int = 300,
    ):
        """
        Initialize case study visualizer.

        Args:
            figsize: Figure size for full case study
            dpi: DPI for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi

    def create_case_study(
        self,
        problem_text: str,
        rollout_text: str,
        kl_result: KLResult,
        spike_analysis: SpikeAnalysis,
        checkpoint_kl: Optional[Dict[str, np.ndarray]] = None,
        title: str = "Case Study",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create comprehensive case study figure.

        Args:
            problem_text: The math problem
            rollout_text: The model's response
            kl_result: KL divergence analysis results
            spike_analysis: Spike detection results
            checkpoint_kl: Optional KL values across checkpoints
            title: Figure title
            save_path: Path to save figure

        Returns:
            matplotlib Figure
        """
        fig = plt.figure(figsize=self.figsize)
        gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1.5, 1])

        # Panel 1: Problem text (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_text_panel(ax1, problem_text, "Problem")

        # Panel 2: Statistics summary (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_statistics_panel(ax2, kl_result, spike_analysis)

        # Panel 3: KL divergence with spikes (middle, full width)
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_kl_with_spikes(ax3, kl_result, spike_analysis)

        # Panel 4: Spike token context (bottom left)
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_spike_context(ax4, kl_result, spike_analysis)

        # Panel 5: Checkpoint evolution or region breakdown (bottom right)
        ax5 = fig.add_subplot(gs[2, 1])
        if checkpoint_kl:
            self._plot_checkpoint_evolution(ax5, checkpoint_kl, kl_result.prompt_length)
        else:
            self._plot_region_breakdown(ax5, spike_analysis)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def _plot_text_panel(
        self,
        ax: plt.Axes,
        text: str,
        title: str,
    ) -> None:
        """Plot text in a panel with wrapping."""
        ax.axis("off")

        # Truncate if too long
        max_chars = 500
        if len(text) > max_chars:
            display_text = text[:max_chars] + "..."
        else:
            display_text = text

        ax.text(
            0.05, 0.95, title,
            fontsize=11, fontweight="bold",
            transform=ax.transAxes,
            verticalalignment="top",
        )
        ax.text(
            0.05, 0.85, display_text,
            fontsize=9,
            transform=ax.transAxes,
            verticalalignment="top",
            wrap=True,
            family="monospace",
        )

    def _plot_statistics_panel(
        self,
        ax: plt.Axes,
        kl_result: KLResult,
        spike_analysis: SpikeAnalysis,
    ) -> None:
        """Plot summary statistics."""
        ax.axis("off")

        stats_text = f"""
Statistics Summary
{'='*30}

Mean KL Divergence: {spike_analysis.mean_kl:.4f}
Std KL Divergence:  {spike_analysis.std_kl:.4f}
Detection Threshold: {spike_analysis.threshold:.4f}

Spikes Detected: {len(spike_analysis.spikes)}
Sequence Length: {len(kl_result.kl_forward)}
Prompt Length:   {kl_result.prompt_length}

Spike Density by Region:
  Early: {spike_analysis.spike_density.get('early', 0):.4f}
  Mid:   {spike_analysis.spike_density.get('mid', 0):.4f}
  Late:  {spike_analysis.spike_density.get('late', 0):.4f}
"""
        ax.text(
            0.05, 0.95, stats_text,
            fontsize=10,
            transform=ax.transAxes,
            verticalalignment="top",
            family="monospace",
        )

    def _plot_kl_with_spikes(
        self,
        ax: plt.Axes,
        kl_result: KLResult,
        spike_analysis: SpikeAnalysis,
    ) -> None:
        """Plot KL divergence with spike annotations."""
        kl_values = kl_result.kl_forward.cpu().numpy() if hasattr(kl_result.kl_forward, 'cpu') else kl_result.kl_forward
        prompt_length = kl_result.prompt_length

        response_kl = kl_values[prompt_length:]
        positions = np.arange(len(response_kl))

        # Main line
        ax.plot(positions, response_kl, color="steelblue", alpha=0.8, linewidth=1.5)
        ax.fill_between(positions, 0, response_kl, color="steelblue", alpha=0.2)

        # Spike markers
        for spike in spike_analysis.spikes:
            adj_pos = spike.position - prompt_length
            if 0 <= adj_pos < len(response_kl):
                ax.axvline(x=adj_pos, color="red", linestyle="--", alpha=0.5, linewidth=1)
                ax.scatter([adj_pos], [spike.value], color="red", s=100, zorder=5, marker="^")

                # Add token label
                if spike.token_string:
                    ax.annotate(
                        repr(spike.token_string),
                        (adj_pos, spike.value),
                        xytext=(5, 10),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.8,
                    )

        # Threshold line
        ax.axhline(y=spike_analysis.threshold, color="orange", linestyle="--", alpha=0.7, label="Threshold")

        ax.set_xlabel("Token Position (Response)")
        ax.set_ylabel("KL Divergence")
        ax.set_title("KL Divergence with Detected Spikes")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    def _plot_spike_context(
        self,
        ax: plt.Axes,
        kl_result: KLResult,
        spike_analysis: SpikeAnalysis,
    ) -> None:
        """Plot spike token context."""
        ax.axis("off")

        if not spike_analysis.spikes:
            ax.text(0.5, 0.5, "No spikes detected", ha="center", va="center", fontsize=12)
            return

        context_text = "Top Spike Tokens\n" + "=" * 30 + "\n\n"

        # Show top 5 spikes
        top_spikes = sorted(spike_analysis.spikes, key=lambda s: s.value, reverse=True)[:5]

        for i, spike in enumerate(top_spikes):
            token_display = repr(spike.token_string) if spike.token_string else "N/A"
            context_text += f"{i+1}. Position {spike.position}: {token_display}\n"
            context_text += f"   KL: {spike.value:.4f}, Z-score: {spike.zscore:.2f}\n"
            context_text += f"   Region: {spike.region}\n\n"

        ax.text(
            0.05, 0.95, context_text,
            fontsize=9,
            transform=ax.transAxes,
            verticalalignment="top",
            family="monospace",
        )

    def _plot_checkpoint_evolution(
        self,
        ax: plt.Axes,
        checkpoint_kl: Dict[str, np.ndarray],
        prompt_length: int,
    ) -> None:
        """Plot mean KL by checkpoint."""
        checkpoint_names = list(checkpoint_kl.keys())
        mean_kls = [np.mean(kl[prompt_length:]) for kl in checkpoint_kl.values()]

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(checkpoint_names)))

        ax.bar(checkpoint_names, mean_kls, color=colors)
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Mean KL Divergence")
        ax.set_title("KL Evolution Across Checkpoints")
        ax.tick_params(axis="x", rotation=45)

    def _plot_region_breakdown(
        self,
        ax: plt.Axes,
        spike_analysis: SpikeAnalysis,
    ) -> None:
        """Plot spike count by region."""
        regions = ["Early", "Mid", "Late"]
        densities = [
            spike_analysis.spike_density.get("early", 0),
            spike_analysis.spike_density.get("mid", 0),
            spike_analysis.spike_density.get("late", 0),
        ]
        colors = ["lightcoral", "lightyellow", "lightgreen"]

        ax.bar(regions, densities, color=colors, edgecolor="black")
        ax.set_xlabel("Region")
        ax.set_ylabel("Spike Density")
        ax.set_title("Spike Distribution by Region")

    def _save_figure(
        self,
        fig: plt.Figure,
        save_path: str,
    ) -> None:
        """Save figure to file."""
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        logger.info(f"Saved case study to {save_path}")
