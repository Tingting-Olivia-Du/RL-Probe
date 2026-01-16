#!/usr/bin/env python3
"""
Step 1: Prepare Data

Load MATH dataset and filter for problems where:
- DPO model produces incorrect answers
- RLVR model produces correct answers

This script evaluates models on the dataset and creates the filtered subset.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from src.data.dataset import MATHDataset, FilteredMATHDataset
from src.data.filter import ProblemFilter
from src.data.preprocessor import DataPreprocessor
from src.models.model_loader import ModelLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    tokenizer,
    problems,
    preprocessor,
    filter_obj,
    max_new_tokens=2048,
    cache_file=None,
    save_interval=10,
):
    """Evaluate a model on problems and return correctness dict.

    Args:
        cache_file: Path to cache file for auto-saving results
        save_interval: Save results every N problems (default: 10)
    """
    results = {}

    # Load existing results if cache file exists
    if cache_file and Path(cache_file).exists():
        logger.info(f"Loading existing results from {cache_file}")
        with open(cache_file) as f:
            results = json.load(f)
        logger.info(f"Loaded {len(results)} existing results")

    for idx, problem in enumerate(tqdm(problems, desc="Evaluating"), 1):
        problem_id = problem.get("unique_id", str(hash(problem["problem"])))

        # Skip if already evaluated
        if problem_id in results:
            continue

        prompt = preprocessor.format_prompt(problem["problem"])

        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,  # Greedy for evaluation
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Check correctness
        ground_truth = problem.get("solution", problem.get("answer", ""))
        is_correct = filter_obj.check_correctness(response, filter_obj.extract_answer(ground_truth) or ground_truth)

        results[problem_id] = is_correct

        # Auto-save every save_interval problems
        if cache_file and idx % save_interval == 0:
            with open(cache_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Auto-saved results after {idx} problems")

    # Final save
    if cache_file:
        with open(cache_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Final save: {len(results)} results saved to {cache_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Prepare filtered MATH dataset")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--output-dir", type=str, default="data/filtered", help="Output directory")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation, use cached results")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info("Loading MATH dataset...")
    dataset = MATHDataset(
        levels=config["dataset"]["levels"],
        split=config["dataset"]["split"],
    )
    dataset.load()
    problems = dataset.get_problems()
    logger.info(f"Loaded {len(problems)} problems")

    # Initialize filter
    filter_obj = ProblemFilter()

    # Check for cached results
    dpo_cache = output_dir / "dpo_results.json"
    rlvr_cache = output_dir / "rlvr_results.json"

    if args.skip_eval and dpo_cache.exists() and rlvr_cache.exists():
        logger.info("Loading cached evaluation results...")
        with open(dpo_cache) as f:
            dpo_results = json.load(f)
        with open(rlvr_cache) as f:
            rlvr_results = json.load(f)
    else:
        # Load models
        logger.info("Loading models for evaluation...")
        loader = ModelLoader(
            device=config["hardware"]["device"],
            dtype=getattr(torch, config["hardware"]["dtype"]),
        )
        loader.register_from_config(config)

        tokenizer = loader.load_tokenizer()
        preprocessor = DataPreprocessor(tokenizer)

        # Evaluate DPO model
        logger.info("Evaluating DPO model...")
        dpo_model = loader.load_model("dpo")
        dpo_results = evaluate_model(
            dpo_model, tokenizer, problems, preprocessor, filter_obj,
            cache_file=dpo_cache
        )
        loader.unload_model("dpo")

        # Evaluate final RLVR model
        logger.info("Evaluating RLVR model...")
        rlvr_checkpoints = loader.get_rlvr_checkpoints()
        final_rlvr = rlvr_checkpoints[-1]  # Get final checkpoint
        rlvr_model = loader.load_model(final_rlvr)
        rlvr_results = evaluate_model(
            rlvr_model, tokenizer, problems, preprocessor, filter_obj,
            cache_file=rlvr_cache
        )
        loader.unload_model(final_rlvr)

    # Filter problems
    logger.info("Filtering problems (DPO-wrong, RLVR-right)...")
    filtered_dataset = FilteredMATHDataset(dataset)
    filtered_problems = filtered_dataset.filter_by_model_performance(dpo_results, rlvr_results)

    # Save filtered problems
    filtered_path = output_dir / "filtered_problems.json"
    filtered_dataset.save_to_cache(str(filtered_path))

    # Print summary
    dpo_correct = sum(dpo_results.values())
    rlvr_correct = sum(rlvr_results.values())

    logger.info("=" * 50)
    logger.info("Data Preparation Summary")
    logger.info("=" * 50)
    logger.info(f"Total problems: {len(problems)}")
    logger.info(f"DPO correct: {dpo_correct} ({dpo_correct/len(problems)*100:.1f}%)")
    logger.info(f"RLVR correct: {rlvr_correct} ({rlvr_correct/len(problems)*100:.1f}%)")
    logger.info(f"Filtered (DPO-wrong, RLVR-right): {len(filtered_problems)}")
    logger.info(f"Saved to: {filtered_path}")


if __name__ == "__main__":
    main()
