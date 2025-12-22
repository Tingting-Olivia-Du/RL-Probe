#!/usr/bin/env python3
"""
Step 4: Visualize Results

Generate visualizations for KL divergence analysis:
- Heatmaps across checkpoints
- Line plots with spike annotations
- Case studies for selected problems
- Aggregate statistics dashboard
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import yaml

from src.visualization.heatmap import KLHeatmapPlotter
from src.visualization.lineplot import KLLinePlotter
from src.visualization.case_study import CaseStudyVisualizer
from src.visualization.aggregate import AggregateVisualizer
from src.analysis.kl_divergence import KLResult
from src.analysis.spike_detector import SpikeAnalysis, Spike

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def reconstruct_spike_analysis(data: dict) -> SpikeAnalysis:
    """Reconstruct SpikeAnalysis from serialized data."""
    spikes = [
        Spike(
            position=pos,
            value=val,
            zscore=0.0,  # Not stored
            token_string=tok,
            region=reg,
        )
        for pos, val, tok, reg in zip(
            data["spike_positions"],
            data["spike_values"],
            data["spike_tokens"],
            data["spike_regions"],
        )
    ]

    return SpikeAnalysis(
        spikes=spikes,
        kl_values=np.array([]),  # Not needed for visualization
        threshold=data["threshold"],
        mean_kl=data["mean_kl"],
        std_kl=data["std_kl"],
        spike_density=data["spike_density"],
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize KL analysis results")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--results-dir", type=str, default="outputs/results", help="Results directory")
    parser.add_argument("--output-dir", type=str, default="outputs/figures", help="Output directory")
    parser.add_argument("--num-case-studies", type=int, default=3, help="Number of case studies")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results_dir = Path(args.results_dir)
    all_results = {}

    for result_file in results_dir.glob("*_results.json"):
        ckpt_name = result_file.stem.replace("_results", "")
        with open(result_file) as f:
            all_results[ckpt_name] = json.load(f)

    if not all_results:
        logger.error("No results found. Run 03_compute_kl.py first.")
        return

    logger.info(f"Loaded results for checkpoints: {list(all_results.keys())}")

    # Initialize visualizers
    heatmap_plotter = KLHeatmapPlotter(
        figsize=tuple(config["visualization"]["heatmap"]["figsize"]),
        cmap=config["visualization"]["heatmap"]["cmap"],
        dpi=config["visualization"]["dpi"],
    )

    line_plotter = KLLinePlotter(
        figsize=tuple(config["visualization"]["lineplot"]["figsize"]),
        dpi=config["visualization"]["dpi"],
    )

    case_study_viz = CaseStudyVisualizer(dpi=config["visualization"]["dpi"])
    aggregate_viz = AggregateVisualizer(dpi=config["visualization"]["dpi"])

    # =========================================================================
    # 1. Aggregate Statistics
    # =========================================================================
    logger.info("Generating aggregate statistics plots...")

    # Mean KL by checkpoint
    checkpoint_means = {}
    checkpoint_stds = {}
    for ckpt_name, results in all_results.items():
        mean_kls = [a["mean_kl"] for a in results["spike_analyses"]]
        checkpoint_means[ckpt_name] = np.mean(mean_kls)
        checkpoint_stds[ckpt_name] = np.std(mean_kls)

    aggregate_viz.plot_mean_kl_progression(
        checkpoint_stats={k: {"mean": checkpoint_means[k], "std": checkpoint_stds[k]} for k in checkpoint_means},
        title="Mean KL Divergence Across RLVR Training",
        save_path=str(output_dir / "mean_kl_progression.png"),
    )

    # Spike density by checkpoint and region
    density_by_checkpoint = {}
    for ckpt_name, results in all_results.items():
        all_densities = {"early": [], "mid": [], "late": []}
        for analysis in results["spike_analyses"]:
            for region, density in analysis["spike_density"].items():
                all_densities[region].append(density)
        density_by_checkpoint[ckpt_name] = {
            region: np.mean(vals) for region, vals in all_densities.items()
        }

    aggregate_viz.plot_spike_density_comparison(
        density_by_checkpoint=density_by_checkpoint,
        title="Spike Density by Region Across Checkpoints",
        save_path=str(output_dir / "spike_density_comparison.png"),
    )

    # Category breakdown (from first checkpoint)
    first_ckpt = list(all_results.keys())[0]
    all_category_stats = {}
    for cat_stat in all_results[first_ckpt]["category_stats"]:
        for cat, stats in cat_stat["stats"].items():
            if cat not in all_category_stats:
                all_category_stats[cat] = {"mean_kls": [], "counts": []}
            if stats["count"] > 0:
                all_category_stats[cat]["mean_kls"].append(stats["mean_kl"])
                all_category_stats[cat]["counts"].append(stats["count"])

    category_summary = {
        cat: {
            "overall_mean_kl": np.mean(data["mean_kls"]) if data["mean_kls"] else 0,
            "overall_std_kl": np.std(data["mean_kls"]) if data["mean_kls"] else 0,
            "total_count": sum(data["counts"]),
        }
        for cat, data in all_category_stats.items()
    }

    aggregate_viz.plot_category_kl_breakdown(
        category_stats=category_summary,
        title="Mean KL Divergence by Token Category",
        save_path=str(output_dir / "category_kl_breakdown.png"),
    )

    # Entropy progression
    entropy_by_checkpoint = {}
    for ckpt_name, results in all_results.items():
        mean_entropies = [a["mean_entropy"] for a in results["entropy_analyses"]]
        entropy_by_checkpoint[ckpt_name] = np.mean(mean_entropies)

    aggregate_viz.plot_entropy_reduction(
        entropy_progression=entropy_by_checkpoint,
        title="Entropy Reduction Across RLVR Training",
        save_path=str(output_dir / "entropy_reduction.png"),
    )

    # Summary dashboard
    aggregate_viz.plot_summary_dashboard(
        checkpoint_means=checkpoint_means,
        spike_densities=density_by_checkpoint,
        category_stats=category_summary,
        title="RL-Probe Analysis Summary",
        save_path=str(output_dir / "summary_dashboard.png"),
    )

    # =========================================================================
    # 2. Sample Heatmaps and Line Plots
    # =========================================================================
    logger.info("Generating sample visualizations...")

    # Get a sample problem
    sample_problem_id = list(all_results[first_ckpt]["kl_analyses"][0].keys())[0] if isinstance(
        all_results[first_ckpt]["kl_analyses"][0], dict
    ) else all_results[first_ckpt]["kl_analyses"][0]["problem_id"]

    # Find corresponding data
    sample_kl_data = next(
        (a for a in all_results[first_ckpt]["kl_analyses"] if a["problem_id"] == sample_problem_id),
        all_results[first_ckpt]["kl_analyses"][0]
    )
    sample_spike_data = next(
        (a for a in all_results[first_ckpt]["spike_analyses"] if a["problem_id"] == sample_problem_id),
        all_results[first_ckpt]["spike_analyses"][0]
    )

    # Single sequence heatmap
    kl_values = np.array(sample_kl_data["kl_forward"])
    heatmap_plotter.plot_single_sequence(
        kl_values=kl_values,
        prompt_length=sample_kl_data["prompt_length"],
        spike_positions=sample_spike_data["spike_positions"],
        title=f"KL Divergence Heatmap (Problem: {sample_kl_data['problem_id'][:20]}...)",
        save_path=str(output_dir / "sample_heatmap.png"),
    )

    # Line plot with spikes
    line_plotter.plot_single_sequence(
        kl_values=kl_values,
        prompt_length=sample_kl_data["prompt_length"],
        spike_positions=sample_spike_data["spike_positions"],
        threshold=sample_spike_data["threshold"],
        title=f"KL Divergence with Spikes",
        save_path=str(output_dir / "sample_lineplot.png"),
    )

    # Checkpoint comparison for same problem
    if len(all_results) > 1:
        kl_by_checkpoint = {}
        for ckpt_name, results in all_results.items():
            matching = next(
                (a for a in results["kl_analyses"] if a["problem_id"] == sample_problem_id),
                None
            )
            if matching:
                kl_by_checkpoint[ckpt_name] = np.array(matching["kl_forward"])

        if len(kl_by_checkpoint) > 1:
            heatmap_plotter.plot_checkpoint_comparison(
                kl_by_checkpoint=kl_by_checkpoint,
                prompt_length=sample_kl_data["prompt_length"],
                title="KL Divergence Across Checkpoints",
                save_path=str(output_dir / "checkpoint_comparison_heatmap.png"),
            )

            line_plotter.plot_checkpoint_comparison(
                kl_by_checkpoint=kl_by_checkpoint,
                prompt_length=sample_kl_data["prompt_length"],
                title="KL Divergence Across Checkpoints",
                save_path=str(output_dir / "checkpoint_comparison_lineplot.png"),
            )

    # =========================================================================
    # 3. Case Studies
    # =========================================================================
    logger.info(f"Generating {args.num_case_studies} case studies...")

    # Select problems with most spikes for case studies
    spike_counts = [
        (a["problem_id"], a["num_spikes"], i)
        for i, a in enumerate(all_results[first_ckpt]["spike_analyses"])
    ]
    spike_counts.sort(key=lambda x: x[1], reverse=True)

    for case_idx in range(min(args.num_case_studies, len(spike_counts))):
        problem_id, num_spikes, data_idx = spike_counts[case_idx]

        kl_data = all_results[first_ckpt]["kl_analyses"][data_idx]
        spike_data = all_results[first_ckpt]["spike_analyses"][data_idx]

        # Reconstruct objects
        kl_result = KLResult(
            kl_forward=np.array(kl_data["kl_forward"]),
            token_strings=kl_data["token_strings"],
            prompt_length=kl_data["prompt_length"],
        )
        spike_analysis = reconstruct_spike_analysis(spike_data)

        # Get checkpoint comparison if available
        checkpoint_kl = None
        if len(all_results) > 1:
            checkpoint_kl = {}
            for ckpt_name, results in all_results.items():
                matching = next(
                    (a for a in results["kl_analyses"] if a["problem_id"] == problem_id),
                    None
                )
                if matching:
                    checkpoint_kl[ckpt_name] = np.array(matching["kl_forward"])

        case_study_viz.create_case_study(
            problem_text=f"Problem ID: {problem_id}",
            rollout_text="[Rollout text not stored in results]",
            kl_result=kl_result,
            spike_analysis=spike_analysis,
            checkpoint_kl=checkpoint_kl,
            title=f"Case Study {case_idx + 1}: {problem_id[:30]}...",
            save_path=str(output_dir / f"case_study_{case_idx + 1}.png"),
        )

    logger.info("\n" + "=" * 50)
    logger.info("Visualization Summary")
    logger.info("=" * 50)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Generated figures:")
    for fig_file in sorted(output_dir.glob("*.png")):
        logger.info(f"  - {fig_file.name}")


if __name__ == "__main__":
    main()
