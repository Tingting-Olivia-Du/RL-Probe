#!/usr/bin/env python3
"""
Step 3: Compute KL Divergence

Compute token-level KL divergence between RLVR checkpoints and DPO baseline.
Uses teacher forcing to analyze how RLVR models respond to DPO-generated errors.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.analysis.kl_divergence import KLDivergenceAnalyzer
from src.analysis.spike_detector import SpikeDetector
from src.analysis.entropy import EntropyAnalyzer
from src.analysis.token_classifier import TokenClassifier
from src.data.preprocessor import DataPreprocessor
from src.models.model_loader import ModelLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Compute KL divergence")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--rollouts", type=str, default="rollouts/dpo_errors/all_rollouts.json", help="Rollouts path")
    parser.add_argument("--output-dir", type=str, default="outputs/results", help="Output directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint to analyze")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load rollouts
    logger.info(f"Loading rollouts from {args.rollouts}...")
    with open(args.rollouts) as f:
        all_rollouts = json.load(f)

    logger.info(f"Loaded rollouts for {len(all_rollouts)} problems")

    # Load models
    logger.info("Loading models...")
    loader = ModelLoader(
        device=config["hardware"]["device"],
        dtype=getattr(torch, config["hardware"]["dtype"]),
    )
    loader.register_from_config(config)

    tokenizer = loader.load_tokenizer()
    preprocessor = DataPreprocessor(tokenizer)

    # Load DPO as reference
    dpo_model = loader.load_model("dpo")

    # Determine which checkpoints to analyze
    if args.checkpoint:
        checkpoints_to_analyze = [args.checkpoint]
    else:
        checkpoints_to_analyze = loader.get_rlvr_checkpoints()

    # Initialize analyzers
    kl_analyzer = KLDivergenceAnalyzer(
        epsilon=config["analysis"]["kl_divergence"]["epsilon"],
        compute_forward=config["analysis"]["kl_divergence"]["compute_forward"],
        compute_reverse=config["analysis"]["kl_divergence"]["compute_reverse"],
        compute_js=config["analysis"]["kl_divergence"]["compute_js"],
    )

    spike_detector = SpikeDetector(
        method=config["analysis"]["spike_detection"]["method"],
        threshold=config["analysis"]["spike_detection"]["threshold"],
        min_spike_distance=config["analysis"]["spike_detection"]["min_spike_distance"],
    )

    entropy_analyzer = EntropyAnalyzer(normalize=config["analysis"]["entropy"]["normalize"])
    token_classifier = TokenClassifier()

    # Results storage
    all_results = {}

    for ckpt_name in checkpoints_to_analyze:
        logger.info(f"\nAnalyzing checkpoint: {ckpt_name}")
        ckpt_model = loader.load_model(ckpt_name)

        ckpt_results = {
            "kl_analyses": [],
            "spike_analyses": [],
            "entropy_analyses": [],
            "category_stats": [],
        }

        for problem_id, rollouts in tqdm(all_rollouts.items(), desc=f"Processing {ckpt_name}"):
            for rollout_data in rollouts:
                rollout_text = rollout_data["rollout"]
                problem_text = rollout_data["problem_text"]

                prompt = preprocessor.format_prompt(problem_text)

                # Compute KL divergence
                kl_result = kl_analyzer.analyze_rollout(
                    model_p=ckpt_model,
                    model_q=dpo_model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    rollout=rollout_text,
                )

                # Detect spikes
                spike_analysis = spike_detector.detect_spikes(
                    kl_values=kl_result.kl_forward,
                    token_ids=kl_result.token_ids,
                    token_strings=kl_result.token_strings,
                    prompt_length=kl_result.prompt_length,
                )

                # Analyze entropy
                entropy_result = entropy_analyzer.analyze_sequence(
                    model=ckpt_model,
                    input_ids=kl_result.token_ids.unsqueeze(0),
                    prompt_length=kl_result.prompt_length,
                )

                # Classify tokens
                classified_tokens = token_classifier.classify_sequence(
                    token_strings=kl_result.token_strings,
                    token_ids=kl_result.token_ids.tolist(),
                    kl_values=kl_result.kl_forward.cpu().numpy(),
                )
                category_stats = token_classifier.get_category_statistics(classified_tokens)

                # Store results (convert tensors to lists for serialization)
                ckpt_results["kl_analyses"].append({
                    "problem_id": problem_id,
                    "rollout_idx": rollout_data["sample_idx"],
                    "kl_forward": kl_result.kl_forward.cpu().numpy().tolist(),
                    "kl_reverse": kl_result.kl_reverse.cpu().numpy().tolist() if kl_result.kl_reverse is not None else None,
                    "js_divergence": kl_result.js_divergence.cpu().numpy().tolist() if kl_result.js_divergence is not None else None,
                    "prompt_length": kl_result.prompt_length,
                    "token_strings": kl_result.token_strings,
                })

                ckpt_results["spike_analyses"].append({
                    "problem_id": problem_id,
                    "rollout_idx": rollout_data["sample_idx"],
                    "num_spikes": len(spike_analysis.spikes),
                    "spike_positions": [s.position for s in spike_analysis.spikes],
                    "spike_values": [s.value for s in spike_analysis.spikes],
                    "spike_tokens": [s.token_string for s in spike_analysis.spikes],
                    "spike_regions": [s.region for s in spike_analysis.spikes],
                    "mean_kl": spike_analysis.mean_kl,
                    "std_kl": spike_analysis.std_kl,
                    "threshold": spike_analysis.threshold,
                    "spike_density": spike_analysis.spike_density,
                })

                ckpt_results["entropy_analyses"].append({
                    "problem_id": problem_id,
                    "rollout_idx": rollout_data["sample_idx"],
                    "mean_entropy": entropy_result.mean_entropy,
                    "std_entropy": entropy_result.std_entropy,
                    "mean_normalized": entropy_result.mean_normalized,
                    "region_entropy": entropy_result.region_entropy,
                })

                ckpt_results["category_stats"].append({
                    "problem_id": problem_id,
                    "rollout_idx": rollout_data["sample_idx"],
                    "stats": category_stats,
                })

        all_results[ckpt_name] = ckpt_results

        # Save checkpoint results
        ckpt_output = output_dir / f"{ckpt_name}_results.json"
        with open(ckpt_output, "w") as f:
            json.dump(ckpt_results, f, indent=2)
        logger.info(f"Saved {ckpt_name} results to {ckpt_output}")

        # Unload to save memory
        loader.unload_model(ckpt_name)

    # Save combined results
    combined_output = output_dir / "all_results.pkl"
    with open(combined_output, "wb") as f:
        pickle.dump(all_results, f)
    logger.info(f"Saved combined results to {combined_output}")

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("KL Analysis Summary")
    logger.info("=" * 50)

    for ckpt_name, results in all_results.items():
        mean_kls = [a["mean_kl"] for a in results["spike_analyses"]]
        total_spikes = sum(a["num_spikes"] for a in results["spike_analyses"])

        logger.info(f"\n{ckpt_name}:")
        logger.info(f"  Analyses: {len(results['kl_analyses'])}")
        logger.info(f"  Mean KL: {np.mean(mean_kls):.4f} +/- {np.std(mean_kls):.4f}")
        logger.info(f"  Total spikes: {total_spikes}")

    loader.unload_all()


if __name__ == "__main__":
    main()
