#!/usr/bin/env python3
"""
Step 4: Visualize Results and Generate Analysis Report

Generate comprehensive visualizations and markdown analysis report for KL divergence analysis.
"""

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rcParams
from scipy import signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
rcParams["figure.figsize"] = (12, 6)
rcParams["font.size"] = 10


def extract_step_number(ckpt_name: str) -> int:
    """Extract step number from checkpoint name (e.g., 'rlvr_step_500' -> 500)."""
    try:
        return int(ckpt_name.replace("rlvr_step_", ""))
    except:
        return 0


def detect_turning_points(
    step_numbers: List[int],
    values: List[float],
    method: str = "derivative",
    min_change: Optional[float] = None,
) -> List[Tuple[int, int, str]]:
    """
    Detect turning points (inflection points) in a time series.
    
    Args:
        step_numbers: List of step numbers (x-axis)
        values: List of corresponding values (y-axis)
        method: Detection method ("derivative", "smooth_derivative", "peaks")
        min_change: Minimum change threshold for detecting turning points
    
    Returns:
        List of (step_number, value, type) tuples where type is "min", "max", or "inflection"
    """
    if len(values) < 3:
        return []
    
    step_numbers = np.array(step_numbers)
    values = np.array(values)
    turning_points = []
    
    if method == "derivative":
        # Compute first derivative (slope)
        dy = np.diff(values)
        dx = np.diff(step_numbers)
        slopes = dy / dx
        
        # Find where slope changes sign (turning points)
        sign_changes = np.where(np.diff(np.sign(slopes)) != 0)[0]
        
        for idx in sign_changes:
            if idx + 1 < len(values):
                step = step_numbers[idx + 1]
                value = values[idx + 1]
                # Determine if it's a local min or max
                if idx > 0 and idx + 1 < len(slopes):
                    if slopes[idx] < 0 and slopes[idx + 1] > 0:
                        turning_points.append((step, value, "min"))
                    elif slopes[idx] > 0 and slopes[idx + 1] < 0:
                        turning_points.append((step, value, "max"))
    
    elif method == "smooth_derivative":
        # Smooth the data first to reduce noise
        if len(values) >= 5:
            window_size = min(5, len(values) // 2)
            if window_size % 2 == 0:
                window_size += 1
            smoothed = signal.savgol_filter(values, window_size, 2)
        else:
            smoothed = values
        
        # Compute second derivative to find inflection points
        dy = np.diff(smoothed)
        dx = np.diff(step_numbers)
        first_deriv = dy / dx
        
        d2y = np.diff(first_deriv)
        d2x = np.diff(step_numbers[:-1])
        second_deriv = d2y / d2x
        
        # Find where second derivative changes sign (inflection points)
        sign_changes = np.where(np.diff(np.sign(second_deriv)) != 0)[0]
        
        for idx in sign_changes:
            if idx + 2 < len(values):
                step = step_numbers[idx + 2]
                value = smoothed[idx + 2]
                turning_points.append((step, value, "inflection"))
    
    elif method == "peaks":
        # Find local peaks and valleys
        peaks, _ = signal.find_peaks(values, distance=2)
        valleys, _ = signal.find_peaks(-values, distance=2)
        
        for idx in peaks:
            step = step_numbers[idx]
            value = values[idx]
            turning_points.append((step, value, "max"))
        
        for idx in valleys:
            step = step_numbers[idx]
            value = values[idx]
            turning_points.append((step, value, "min"))
    
    # Filter by minimum change if specified
    if min_change is not None and turning_points:
        filtered_points = []
        for step, value, point_type in turning_points:
            # Check if this turning point represents a significant change
            step_idx = np.where(step_numbers == step)[0][0]
            if step_idx > 0 and step_idx < len(values) - 1:
                prev_val = values[step_idx - 1]
                next_val = values[step_idx + 1] if step_idx + 1 < len(values) else values[step_idx]
                change = abs(value - prev_val) + abs(next_val - value)
                if change >= min_change:
                    filtered_points.append((step, value, point_type))
        turning_points = filtered_points
    
    return turning_points


def load_results(results_dir: Path):
    """Load all result files."""
    all_results = {}
    for result_file in results_dir.glob("*_results.json"):
        ckpt_name = result_file.stem.replace("_results", "")
        logger.info(f"Loading {ckpt_name}...")
        with open(result_file) as f:
            all_results[ckpt_name] = json.load(f)
    return all_results


def analyze_statistics(all_results: dict):
    """Compute comprehensive statistics."""
    stats = {}
    
    for ckpt_name, results in all_results.items():
        # KL statistics
        mean_kls = [a["mean_kl"] for a in results["spike_analyses"]]
        std_kls = [a["std_kl"] for a in results["spike_analyses"]]
        
        # Spike statistics
        num_spikes_list = [a["num_spikes"] for a in results["spike_analyses"]]
        spike_positions_all = []
        spike_values_all = []
        for a in results["spike_analyses"]:
            spike_positions_all.extend(a["spike_positions"])
            spike_values_all.extend(a["spike_values"])
        
        # Entropy statistics
        mean_entropies = [a["mean_entropy"] for a in results["entropy_analyses"]]
        
        # Category statistics
        category_stats = defaultdict(lambda: {"mean_kls": [], "counts": []})
        for cat_stat in results["category_stats"]:
            for cat, stats_data in cat_stat["stats"].items():
                if stats_data["count"] > 0:
                    category_stats[cat]["mean_kls"].append(stats_data["mean_kl"])
                    category_stats[cat]["counts"].append(stats_data["count"])
        
        stats[ckpt_name] = {
            "num_problems": len(results["kl_analyses"]),
            "mean_kl": np.mean(mean_kls),
            "std_kl": np.std(mean_kls),
            "median_kl": np.median(mean_kls),
            "max_kl": np.max(mean_kls),
            "min_kl": np.min(mean_kls),
            "mean_spikes_per_problem": np.mean(num_spikes_list),
            "total_spikes": sum(num_spikes_list),
            "mean_spike_value": np.mean(spike_values_all) if spike_values_all else 0,
            "mean_entropy": np.mean(mean_entropies),
            "category_stats": {
                cat: {
                    "mean_kl": np.mean(data["mean_kls"]) if data["mean_kls"] else 0,
                    "total_count": sum(data["counts"]),
                }
                for cat, data in category_stats.items()
            },
        }
    
    return stats


def plot_kl_distribution(stats: dict, output_dir: Path):
    """Plot KL divergence distribution with turning point detection."""
    # Sort checkpoints by step number
    sorted_ckpts = sorted(stats.items(), key=lambda x: extract_step_number(x[0]))
    ckpt_names = [c[0] for c in sorted_ckpts]
    step_numbers = [extract_step_number(c[0]) for c in sorted_ckpts]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Mean KL by checkpoint (line plot for progression)
    mean_kls = [stats[c]["mean_kl"] for c in ckpt_names]
    std_kls = [stats[c]["std_kl"] for c in ckpt_names]
    
    # Detect turning points
    turning_points = detect_turning_points(step_numbers, mean_kls, method="smooth_derivative")
    
    axes[0].plot(step_numbers, mean_kls, marker="o", linewidth=2, markersize=8, label="Mean KL", zorder=1)
    axes[0].fill_between(step_numbers, 
                         [m - s for m, s in zip(mean_kls, std_kls)],
                         [m + s for m, s in zip(mean_kls, std_kls)],
                         alpha=0.3, label="±1 Std", zorder=0)
    
    # Mark turning points
    if turning_points:
        for step, value, point_type in turning_points:
            color = "green" if point_type == "min" else "red" if point_type == "max" else "orange"
            marker = "v" if point_type == "min" else "^" if point_type == "max" else "s"
            axes[0].scatter(step, value, s=200, c=color, marker=marker, 
                           edgecolors="black", linewidths=2, zorder=3,
                           label=f"Turning Point ({point_type})" if turning_points.index((step, value, point_type)) == 0 else "")
            axes[0].annotate(f"Step {step}", xy=(step, value), 
                            xytext=(10, 10), textcoords="offset points",
                            fontsize=9, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"))
    
    axes[0].set_title("KL Divergence Progression Across Training\n(Turning Points Marked)", 
                     fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Mean KL Divergence")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    
    # Bar plot with error bars
    axes[1].bar(range(len(ckpt_names)), mean_kls, yerr=std_kls, capsize=5, alpha=0.7)
    axes[1].set_xticks(range(len(ckpt_names)))
    axes[1].set_xticklabels([f"Step {s}" for s in step_numbers], rotation=45, ha="right")
    axes[1].set_title("Mean KL Divergence by Checkpoint", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Mean KL Divergence")
    axes[1].grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_dir / "kl_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: kl_distribution.png")


def plot_spike_analysis(stats: dict, output_dir: Path):
    """Plot spike analysis."""
    # Sort checkpoints by step number
    sorted_ckpts = sorted(stats.items(), key=lambda x: extract_step_number(x[0]))
    ckpt_names = [c[0] for c in sorted_ckpts]
    step_numbers = [extract_step_number(c[0]) for c in sorted_ckpts]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Mean spikes per problem (line plot)
    mean_spikes = [stats[c]["mean_spikes_per_problem"] for c in ckpt_names]
    axes[0].plot(step_numbers, mean_spikes, marker="o", linewidth=2, markersize=8, color="coral")
    axes[0].set_title("Mean Spikes per Problem Across Training", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Mean Number of Spikes")
    axes[0].grid(True, alpha=0.3)
    
    # Total spikes (bar plot)
    total_spikes = [stats[c]["total_spikes"] for c in ckpt_names]
    axes[1].bar(range(len(ckpt_names)), total_spikes, alpha=0.7, color="steelblue")
    axes[1].set_xticks(range(len(ckpt_names)))
    axes[1].set_xticklabels([f"Step {s}" for s in step_numbers], rotation=45, ha="right")
    axes[1].set_title("Total Spikes Detected", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Total Spikes")
    axes[1].grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_dir / "spike_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: spike_analysis.png")


def plot_category_breakdown(stats: dict, output_dir: Path):
    """Plot token category breakdown."""
    # Use first checkpoint for category stats
    first_ckpt = list(stats.keys())[0]
    category_stats = stats[first_ckpt]["category_stats"]
    
    categories = list(category_stats.keys())
    mean_kls = [category_stats[cat]["mean_kl"] for cat in categories]
    counts = [category_stats[cat]["total_count"] for cat in categories]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mean KL by category
    axes[0].barh(categories, mean_kls, alpha=0.7)
    axes[0].set_title("Mean KL Divergence by Token Category", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Mean KL Divergence")
    axes[0].grid(True, alpha=0.3)
    
    # Token count by category
    axes[1].barh(categories, counts, alpha=0.7, color="green")
    axes[1].set_title("Token Count by Category", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Total Count")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "category_breakdown.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: category_breakdown.png")


def plot_sample_kl_sequence(all_results: dict, output_dir: Path, num_samples: int = 3):
    """Plot sample KL sequences with spikes."""
    first_ckpt = list(all_results.keys())[0]
    results = all_results[first_ckpt]
    
    # Select problems with most spikes
    spike_counts = [
        (i, a["num_spikes"], a["problem_id"])
        for i, a in enumerate(results["spike_analyses"])
    ]
    spike_counts.sort(key=lambda x: x[1], reverse=True)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for idx, (sample_idx, num_spikes, problem_id) in enumerate(spike_counts[:num_samples]):
        kl_data = results["kl_analyses"][sample_idx]
        spike_data = results["spike_analyses"][sample_idx]
        
        kl_values = np.array(kl_data["kl_forward"])
        prompt_length = kl_data["prompt_length"]
        
        # Plot KL values
        axes[idx].plot(kl_values, alpha=0.7, linewidth=1.5, label="KL Divergence")
        
        # Mark prompt region
        axes[idx].axvline(x=prompt_length, color="red", linestyle="--", alpha=0.5, label="Prompt End")
        
        # Mark spikes
        for spike_pos in spike_data["spike_positions"]:
            axes[idx].axvline(x=spike_pos, color="orange", linestyle=":", alpha=0.5)
            axes[idx].plot(spike_pos, kl_values[spike_pos], "ro", markersize=8)
        
        # Threshold line
        axes[idx].axhline(y=spike_data["threshold"], color="green", linestyle="--", alpha=0.5, label="Threshold")
        
        axes[idx].set_title(f"Problem: {problem_id[:50]}... (Spikes: {num_spikes})", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Token Position")
        axes[idx].set_ylabel("KL Divergence")
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "sample_kl_sequences.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: sample_kl_sequences.png")


def plot_training_progression(stats: dict, output_dir: Path):
    """Plot comprehensive training progression across all checkpoints with turning points."""
    # Sort checkpoints by step number
    sorted_ckpts = sorted(stats.items(), key=lambda x: extract_step_number(x[0]))
    ckpt_names = [c[0] for c in sorted_ckpts]
    step_numbers = [extract_step_number(c[0]) for c in sorted_ckpts]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. KL Divergence progression with turning points
    mean_kls = [stats[c]["mean_kl"] for c in ckpt_names]
    std_kls = [stats[c]["std_kl"] for c in ckpt_names]
    
    # Detect turning points
    turning_points = detect_turning_points(step_numbers, mean_kls, method="smooth_derivative")
    
    axes[0, 0].plot(step_numbers, mean_kls, marker="o", linewidth=2, markersize=8, label="Mean KL", zorder=1)
    axes[0, 0].fill_between(step_numbers, 
                            [m - s for m, s in zip(mean_kls, std_kls)],
                            [m + s for m, s in zip(mean_kls, std_kls)],
                            alpha=0.3, label="±1 Std", zorder=0)
    
    # Mark turning points
    if turning_points:
        for step, value, point_type in turning_points:
            color = "green" if point_type == "min" else "red" if point_type == "max" else "orange"
            marker = "v" if point_type == "min" else "^" if point_type == "max" else "s"
            axes[0, 0].scatter(step, value, s=200, c=color, marker=marker, 
                             edgecolors="black", linewidths=2, zorder=3)
    
    axes[0, 0].set_title("KL Divergence Progression (Turning Points)", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("Training Step")
    axes[0, 0].set_ylabel("Mean KL Divergence")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Spike progression
    mean_spikes = [stats[c]["mean_spikes_per_problem"] for c in ckpt_names]
    axes[0, 1].plot(step_numbers, mean_spikes, marker="s", linewidth=2, markersize=8, color="coral", label="Mean Spikes")
    axes[0, 1].set_title("Spike Pattern Progression", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlabel("Training Step")
    axes[0, 1].set_ylabel("Mean Spikes per Problem")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Entropy progression
    mean_entropies = [stats[c]["mean_entropy"] for c in ckpt_names]
    axes[1, 0].plot(step_numbers, mean_entropies, marker="^", linewidth=2, markersize=8, color="green", label="Mean Entropy")
    axes[1, 0].set_title("Entropy Progression", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("Training Step")
    axes[1, 0].set_ylabel("Mean Entropy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Combined metrics (normalized)
    # Normalize all metrics to [0, 1] for comparison
    def normalize(values):
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return [0.5] * len(values)
        return [(v - min_val) / (max_val - min_val) for v in values]
    
    norm_kl = normalize(mean_kls)
    norm_spikes = normalize(mean_spikes)
    norm_entropy = normalize(mean_entropies)
    
    axes[1, 1].plot(step_numbers, norm_kl, marker="o", linewidth=2, markersize=6, label="KL (normalized)")
    axes[1, 1].plot(step_numbers, norm_spikes, marker="s", linewidth=2, markersize=6, label="Spikes (normalized)")
    axes[1, 1].plot(step_numbers, norm_entropy, marker="^", linewidth=2, markersize=6, label="Entropy (normalized)")
    axes[1, 1].set_title("Normalized Metrics Comparison", fontsize=14, fontweight="bold")
    axes[1, 1].set_xlabel("Training Step")
    axes[1, 1].set_ylabel("Normalized Value")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_progression.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved: training_progression.png")


def generate_markdown_report(stats: dict, all_results: dict, output_dir: Path):
    """Generate comprehensive markdown analysis report."""
    report = []
    report.append("# RL-Probe: KL Divergence Analysis Report")
    report.append("")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("---")
    report.append("")
    
    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    first_ckpt = list(stats.keys())[0]
    first_stats = stats[first_ckpt]
    
    report.append(f"This report analyzes **{first_stats['num_problems']} problems** where DPO model failed but RLVR model succeeded.")
    report.append("")
    report.append("### Key Findings")
    report.append("")
    report.append(f"- **Mean KL Divergence:** {first_stats['mean_kl']:.4f} ± {first_stats['std_kl']:.4f}")
    report.append(f"- **Total Spikes Detected:** {first_stats['total_spikes']}")
    report.append(f"- **Mean Spikes per Problem:** {first_stats['mean_spikes_per_problem']:.2f}")
    report.append(f"- **Mean Entropy:** {first_stats['mean_entropy']:.4f}")
    report.append("")
    
    # Checkpoint Comparison (sorted by step)
    if len(stats) > 1:
        report.append("## Checkpoint Comparison")
        report.append("")
        report.append("| Checkpoint | Step | Mean KL | Std KL | Mean Spikes | Total Spikes | Mean Entropy |")
        report.append("|------------|------|---------|--------|-------------|--------------|-------------|")
        sorted_ckpts = sorted(stats.items(), key=lambda x: extract_step_number(x[0]))
        for ckpt_name, ckpt_stats in sorted_ckpts:
            step_num = extract_step_number(ckpt_name)
            report.append(
                f"| {ckpt_name} | {step_num} | {ckpt_stats['mean_kl']:.4f} | {ckpt_stats['std_kl']:.4f} | "
                f"{ckpt_stats['mean_spikes_per_problem']:.2f} | {ckpt_stats['total_spikes']} | "
                f"{ckpt_stats['mean_entropy']:.4f} |"
            )
        report.append("")
        
        # Training progression analysis
        report.append("### Training Progression Analysis")
        report.append("")
        if len(sorted_ckpts) >= 2:
            step_numbers = [extract_step_number(c[0]) for c in sorted_ckpts]
            mean_kls = [c[1]['mean_kl'] for c in sorted_ckpts]
            
            first_step = step_numbers[0]
            first_kl = mean_kls[0]
            last_step = step_numbers[-1]
            last_kl = mean_kls[-1]
            kl_change = last_kl - first_kl
            kl_change_pct = (kl_change / first_kl * 100) if first_kl > 0 else 0
            
            report.append(f"- **KL Divergence Change:** {first_kl:.4f} (Step {first_step}) → {last_kl:.4f} (Step {last_step})")
            report.append(f"  - Change: {kl_change:+.4f} ({kl_change_pct:+.1f}%)")
            
            first_spikes = sorted_ckpts[0][1]['mean_spikes_per_problem']
            last_spikes = sorted_ckpts[-1][1]['mean_spikes_per_problem']
            spikes_change = last_spikes - first_spikes
            
            report.append(f"- **Spike Pattern Change:** {first_spikes:.2f} (Step {first_step}) → {last_spikes:.2f} (Step {last_step})")
            report.append(f"  - Change: {spikes_change:+.2f} spikes per problem")
            
            # Detect and report turning points
            turning_points = detect_turning_points(step_numbers, mean_kls, method="smooth_derivative")
            if turning_points:
                report.append("")
                report.append("### 🔍 KL Turning Points Detected")
                report.append("")
                report.append("Turning points indicate where KL divergence trend changes significantly.")
                report.append("These may indicate checkpoints where the model's behavior shifts.")
                report.append("")
                report.append("| Step | KL Value | Type | Interpretation |")
                report.append("|------|----------|------|-----------------|")
                for step, value, point_type in turning_points:
                    if point_type == "min":
                        interpretation = "Local minimum - KL divergence decreases"
                    elif point_type == "max":
                        interpretation = "Local maximum - KL divergence increases"
                    else:
                        interpretation = "Inflection point - trend changes"
                    report.append(f"| {step} | {value:.4f} | {point_type} | {interpretation} |")
                report.append("")
                report.append("💡 **Key Insight**: Turning points may indicate checkpoints where the model")
                report.append("   starts to 'correct' DPO errors more effectively, suggesting improved reasoning capability.")
            report.append("")
    
    # Detailed Statistics (sorted by step)
    report.append("## Detailed Statistics")
    report.append("")
    sorted_ckpts = sorted(stats.items(), key=lambda x: extract_step_number(x[0]))
    for ckpt_name, ckpt_stats in sorted_ckpts:
        report.append(f"### {ckpt_name}")
        report.append("")
        report.append(f"- **Number of Problems:** {ckpt_stats['num_problems']}")
        report.append(f"- **KL Divergence:**")
        report.append(f"  - Mean: {ckpt_stats['mean_kl']:.4f}")
        report.append(f"  - Median: {ckpt_stats['median_kl']:.4f}")
        report.append(f"  - Std: {ckpt_stats['std_kl']:.4f}")
        report.append(f"  - Min: {ckpt_stats['min_kl']:.4f}")
        report.append(f"  - Max: {ckpt_stats['max_kl']:.4f}")
        report.append(f"- **Spike Analysis:**")
        report.append(f"  - Mean spikes per problem: {ckpt_stats['mean_spikes_per_problem']:.2f}")
        report.append(f"  - Total spikes: {ckpt_stats['total_spikes']}")
        report.append(f"  - Mean spike value: {ckpt_stats['mean_spike_value']:.4f}")
        report.append(f"- **Entropy:** {ckpt_stats['mean_entropy']:.4f}")
        report.append("")
    
    # Category Analysis
    report.append("## Token Category Analysis")
    report.append("")
    first_stats = stats[list(stats.keys())[0]]
    category_stats = first_stats["category_stats"]
    
    report.append("| Category | Mean KL | Total Count |")
    report.append("|----------|---------|-------------|")
    for cat, cat_stats in sorted(category_stats.items(), key=lambda x: x[1]["mean_kl"], reverse=True):
        report.append(f"| {cat} | {cat_stats['mean_kl']:.4f} | {cat_stats['total_count']} |")
    report.append("")
    
    # Spike Distribution Analysis
    report.append("## Spike Distribution Analysis")
    report.append("")
    first_ckpt = list(all_results.keys())[0]
    results = all_results[first_ckpt]
    
    spike_counts_list = [a["num_spikes"] for a in results["spike_analyses"]]
    zero_spikes = sum(1 for s in spike_counts_list if s == 0)
    low_spikes = sum(1 for s in spike_counts_list if 1 <= s <= 5)
    mid_spikes = sum(1 for s in spike_counts_list if 6 <= s <= 10)
    high_spikes = sum(1 for s in spike_counts_list if s > 10)
    
    report.append("| Spike Range | Count | Percentage |")
    report.append("|-------------|-------|------------|")
    report.append(f"| 0 spikes | {zero_spikes} | {zero_spikes/len(spike_counts_list)*100:.1f}% |")
    report.append(f"| 1-5 spikes | {low_spikes} | {low_spikes/len(spike_counts_list)*100:.1f}% |")
    report.append(f"| 6-10 spikes | {mid_spikes} | {mid_spikes/len(spike_counts_list)*100:.1f}% |")
    report.append(f"| >10 spikes | {high_spikes} | {high_spikes/len(spike_counts_list)*100:.1f}% |")
    report.append("")
    
    # Top Problems with Most Spikes
    report.append("## Top Problems with Most Spikes")
    report.append("")
    spike_counts = [
        (a["problem_id"], a["num_spikes"], a["mean_kl"], i)
        for i, a in enumerate(results["spike_analyses"])
    ]
    spike_counts.sort(key=lambda x: x[1], reverse=True)
    
    report.append("| Rank | Problem ID | Spikes | Mean KL |")
    report.append("|------|------------|--------|---------|")
    for rank, (problem_id, num_spikes, mean_kl, _) in enumerate(spike_counts[:10], 1):
        report.append(f"| {rank} | `{problem_id[:50]}...` | {num_spikes} | {mean_kl:.4f} |")
    report.append("")
    
    # Problems with Highest Mean KL
    report.append("## Problems with Highest Mean KL Divergence")
    report.append("")
    kl_rankings = [
        (a["problem_id"], a["mean_kl"], a["num_spikes"], i)
        for i, a in enumerate(results["spike_analyses"])
    ]
    kl_rankings.sort(key=lambda x: x[1], reverse=True)
    
    report.append("| Rank | Problem ID | Mean KL | Spikes |")
    report.append("|------|------------|---------|--------|")
    for rank, (problem_id, mean_kl, num_spikes, _) in enumerate(kl_rankings[:10], 1):
        report.append(f"| {rank} | `{problem_id[:50]}...` | {mean_kl:.4f} | {num_spikes} |")
    report.append("")
    
    # Visualizations
    report.append("## Visualizations")
    report.append("")
    report.append("The following visualizations have been generated:")
    report.append("")
    report.append("1. **KL Distribution** (`kl_distribution.png`)")
    report.append("   - Mean KL divergence by checkpoint")
    report.append("   - Distribution of KL values")
    report.append("")
    report.append("2. **Spike Analysis** (`spike_analysis.png`)")
    report.append("   - Mean spikes per problem")
    report.append("   - Total spikes detected")
    report.append("")
    report.append("3. **Category Breakdown** (`category_breakdown.png`)")
    report.append("   - Mean KL divergence by token category")
    report.append("   - Token count by category")
    report.append("")
    report.append("4. **Sample KL Sequences** (`sample_kl_sequences.png`)")
    report.append("   - Sample problems with spike annotations")
    report.append("")
    
    # Methodology
    report.append("## Methodology")
    report.append("")
    report.append("### Teacher Forcing Analysis")
    report.append("")
    report.append("This analysis uses **Teacher Forcing** to examine how RLVR models respond to DPO-generated errors:")
    report.append("")
    report.append("1. **Input:** DPO responses (wrong answers) from `filtered_problems.json`")
    report.append("2. **Process:** Feed `prompt + DPO_response` to both RLVR and DPO models")
    report.append("3. **Analysis:** Compute token-level KL divergence: D_KL(RLVR || DPO)")
    report.append("4. **Detection:** Identify KL spikes (significant deviations)")
    report.append("")
    
    # Interpretation
    report.append("## Interpretation")
    report.append("")
    report.append("### What High KL Values Mean")
    report.append("")
    report.append("- **High KL divergence** indicates that RLVR model's probability distribution differs significantly from DPO's at that token position")
    report.append("- **KL spikes** may indicate:")
    report.append("  - RLVR model attempting to 'correct' DPO's error")
    report.append("  - Critical decision points in reasoning")
    report.append("  - Areas where RLVR training has shifted the model's behavior")
    report.append("")
    report.append("### Key Insights")
    report.append("")
    first_stats = stats[list(stats.keys())[0]]
    
    report.append(f"1. **Overall KL Divergence:** {first_stats['mean_kl']:.4f}")
    report.append("   - This relatively low mean KL suggests that RLVR model, while different from DPO, maintains similar token-level distributions")
    report.append("   - The presence of spikes indicates localized differences rather than global distribution shifts")
    report.append("")
    
    report.append(f"2. **Spike Patterns:**")
    report.append(f"   - Mean of {first_stats['mean_spikes_per_problem']:.1f} spikes per problem indicates frequent localized corrections")
    report.append(f"   - {high_spikes} problems ({high_spikes/len(spike_counts_list)*100:.1f}%) have more than 10 spikes, suggesting complex error correction")
    report.append("")
    
    report.append("3. **Category Analysis:**")
    if category_stats:
        top_category = max(category_stats.items(), key=lambda x: x[1]["mean_kl"])
        bottom_category = min(category_stats.items(), key=lambda x: x[1]["mean_kl"])
        report.append(f"   - **Highest KL:** {top_category[0]} (Mean KL: {top_category[1]['mean_kl']:.4f})")
        report.append(f"     - RLVR model shows significant differences in handling {top_category[0]} tokens")
        report.append(f"   - **Lowest KL:** {bottom_category[0]} (Mean KL: {bottom_category[1]['mean_kl']:.4f})")
        report.append(f"     - RLVR model maintains similar behavior for {bottom_category[0]} tokens")
    report.append("")
    
    report.append("4. **Entropy Analysis:**")
    report.append(f"   - Mean entropy: {first_stats['mean_entropy']:.4f}")
    report.append("   - Lower entropy suggests RLVR model is more confident/consistent in its predictions")
    report.append("")
    
    report.append("### Implications for RLVR Training")
    report.append("")
    report.append("- **Localized Corrections:** The spike pattern suggests RLVR learns to make targeted corrections rather than wholesale distribution changes")
    report.append("- **Category-Specific Learning:** Different KL values across categories indicate RLVR has learned category-specific improvements")
    report.append("- **Error Correction Mechanism:** High spike values in specific problems suggest RLVR has developed mechanisms to identify and correct DPO's errors")
    report.append("")
    
    # Save report
    report_path = output_dir / "analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    logger.info(f"Saved analysis report: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Visualize KL analysis results and generate report")
    parser.add_argument("--results-dir", type=str, default="outputs/results", help="Results directory")
    parser.add_argument("--output-dir", type=str, default="outputs/figures", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of sample sequences to plot")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results_dir = Path(args.results_dir)
    logger.info(f"Loading results from {results_dir}...")
    all_results = load_results(results_dir)
    
    if not all_results:
        logger.error("No results found. Run 03_compute_kl.py first.")
        return
    
    logger.info(f"Loaded results for checkpoints: {list(all_results.keys())}")
    
    # Analyze statistics
    logger.info("Computing statistics...")
    stats = analyze_statistics(all_results)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    plot_kl_distribution(stats, output_dir)
    plot_spike_analysis(stats, output_dir)
    plot_category_breakdown(stats, output_dir)
    plot_sample_kl_sequence(all_results, output_dir, args.num_samples)
    
    # Training progression (if multiple checkpoints)
    if len(stats) > 1:
        plot_training_progression(stats, output_dir)
    
    # Generate markdown report
    logger.info("Generating markdown report...")
    report_path = generate_markdown_report(stats, all_results, output_dir)
    
    logger.info("\n" + "=" * 50)
    logger.info("Visualization Complete!")
    logger.info("=" * 50)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Analysis report: {report_path}")
    logger.info("\nGenerated files:")
    for fig_file in sorted(output_dir.glob("*.png")):
        logger.info(f"  - {fig_file.name}")
    logger.info(f"  - analysis_report.md")


if __name__ == "__main__":
    main()
