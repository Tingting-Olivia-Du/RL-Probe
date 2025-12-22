#!/usr/bin/env python3
"""
Baseline Comparison Experiment

Compare DPO vs SFT to isolate RLVR's unique contribution.
This addresses the suggestion to add baseline comparisons.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.analysis.kl_divergence import KLDivergenceAnalyzer
from src.data.preprocessor import DataPreprocessor
from src.models.model_loader import ModelLoader
from src.visualization.aggregate import AggregateVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Baseline comparison experiment")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--rollouts", type=str, default="rollouts/dpo_errors/all_rollouts.json")
    parser.add_argument("--output-dir", type=str, default="outputs/baseline_comparison")
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load rollouts
    with open(args.rollouts) as f:
        all_rollouts = json.load(f)

    # Flatten and limit samples
    samples = []
    for problem_id, rollouts in all_rollouts.items():
        for r in rollouts[:1]:  # Take first rollout per problem
            samples.append({
                "problem_id": problem_id,
                "problem_text": r["problem_text"],
                "rollout": r["rollout"],
            })
            if len(samples) >= args.max_samples:
                break
        if len(samples) >= args.max_samples:
            break

    logger.info(f"Analyzing {len(samples)} samples")

    # Load models
    loader = ModelLoader(
        device=config["hardware"]["device"],
        dtype=getattr(torch, config["hardware"]["dtype"]),
    )
    loader.register_from_config(config)

    tokenizer = loader.load_tokenizer()
    preprocessor = DataPreprocessor(tokenizer)

    # Load all models
    sft_model = loader.load_model("sft")
    dpo_model = loader.load_model("dpo")
    rlvr_model = loader.load_model(loader.get_rlvr_checkpoints()[-1])

    kl_analyzer = KLDivergenceAnalyzer()

    # Compute comparisons
    comparisons = {
        "dpo_vs_sft": [],      # How much did DPO change from SFT?
        "rlvr_vs_dpo": [],     # How much did RLVR change from DPO?
        "rlvr_vs_sft": [],     # Total change from SFT to RLVR
    }

    for sample in tqdm(samples, desc="Computing baselines"):
        prompt = preprocessor.format_prompt(sample["problem_text"])

        # DPO vs SFT
        kl_dpo_sft = kl_analyzer.analyze_rollout(
            model_p=dpo_model,
            model_q=sft_model,
            tokenizer=tokenizer,
            prompt=prompt,
            rollout=sample["rollout"],
        )
        comparisons["dpo_vs_sft"].append(
            float(kl_dpo_sft.kl_forward[kl_dpo_sft.prompt_length:].mean())
        )

        # RLVR vs DPO
        kl_rlvr_dpo = kl_analyzer.analyze_rollout(
            model_p=rlvr_model,
            model_q=dpo_model,
            tokenizer=tokenizer,
            prompt=prompt,
            rollout=sample["rollout"],
        )
        comparisons["rlvr_vs_dpo"].append(
            float(kl_rlvr_dpo.kl_forward[kl_rlvr_dpo.prompt_length:].mean())
        )

        # RLVR vs SFT
        kl_rlvr_sft = kl_analyzer.analyze_rollout(
            model_p=rlvr_model,
            model_q=sft_model,
            tokenizer=tokenizer,
            prompt=prompt,
            rollout=sample["rollout"],
        )
        comparisons["rlvr_vs_sft"].append(
            float(kl_rlvr_sft.kl_forward[kl_rlvr_sft.prompt_length:].mean())
        )

    # Save results
    results = {
        comparison: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
        }
        for comparison, values in comparisons.items()
    }

    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Visualize
    viz = AggregateVisualizer()
    viz.plot_mean_kl_progression(
        checkpoint_stats={
            "DPO vs SFT": {"mean": results["dpo_vs_sft"]["mean"], "std": results["dpo_vs_sft"]["std"]},
            "RLVR vs DPO": {"mean": results["rlvr_vs_dpo"]["mean"], "std": results["rlvr_vs_dpo"]["std"]},
            "RLVR vs SFT": {"mean": results["rlvr_vs_sft"]["mean"], "std": results["rlvr_vs_sft"]["std"]},
        },
        title="Baseline Comparison: Distribution Shifts",
        save_path=str(output_dir / "baseline_comparison.png"),
    )

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("Baseline Comparison Results")
    logger.info("=" * 50)
    logger.info(f"DPO vs SFT:  {results['dpo_vs_sft']['mean']:.4f} +/- {results['dpo_vs_sft']['std']:.4f}")
    logger.info(f"RLVR vs DPO: {results['rlvr_vs_dpo']['mean']:.4f} +/- {results['rlvr_vs_dpo']['std']:.4f}")
    logger.info(f"RLVR vs SFT: {results['rlvr_vs_sft']['mean']:.4f} +/- {results['rlvr_vs_sft']['std']:.4f}")
    logger.info("")
    logger.info("Interpretation:")
    logger.info("  - 'DPO vs SFT' shows the change during preference optimization")
    logger.info("  - 'RLVR vs DPO' shows the unique contribution of RL training")
    logger.info("  - 'RLVR vs SFT' shows the total distribution shift")

    loader.unload_all()


if __name__ == "__main__":
    main()
