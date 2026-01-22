#!/usr/bin/env python3
"""
Step 1: Prepare Data

Load MATH dataset and filter for problems where:
- DPO model produces incorrect answers
- RLVR model produces correct answers

This script evaluates models on the dataset and creates the filtered subset.
# 使用 GPU 0 运行 DPO 模型，评测 50 道题
python scripts/01_prepare_data.py --model dpo --gpu 6 --max-questions 500

# 使用 GPU 1 和 2 运行两个模型
python scripts/01_prepare_data.py --model both --gpu 1,2 --max-questions 50

# 使用 GPU 4 运行 RLVR 模型
python scripts/01_prepare_data.py --model rlvr --gpu 2 --max-questions 500

"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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


def _load_results_cache(cache_file):
    if not cache_file or not Path(cache_file).exists():
        return {}
    with open(cache_file) as f:
        raw_results = json.load(f)
    results = {}
    for problem_id, value in raw_results.items():
        if isinstance(value, dict) and "correct" in value:
            results[problem_id] = bool(value["correct"])
        else:
            results[problem_id] = bool(value)
    return results


def _load_response_cache(response_cache_file):
    if not response_cache_file or not Path(response_cache_file).exists():
        return {}
    with open(response_cache_file) as f:
        return json.load(f)


def evaluate_model(
    model,
    tokenizer,
    problems,
    preprocessor,
    filter_obj,
    max_new_tokens=2048,
    cache_file=None,
    response_cache_file=None,
    save_interval=10,
    max_questions=None,
):
    """Evaluate a model on problems and return correctness dict.

    Args:
        cache_file: Path to cache file for auto-saving results
        save_interval: Save results every N problems (default: 10)
    """
    results = _load_results_cache(cache_file)
    responses = _load_response_cache(response_cache_file)
    if cache_file and results:
        logger.info(f"Loaded {len(results)} existing results from {cache_file}")
    if response_cache_file and responses:
        logger.info(f"Loaded {len(responses)} existing responses from {response_cache_file}")

    evaluated_new = 0

    for idx, problem in enumerate(tqdm(problems, desc="Evaluating"), 1):
        # 优先使用 unique_id，如果没有则使用稳定的 hash
        # 使用 abs() 确保 hash 值为正数，避免负号导致的字符串不一致
        problem_id = problem.get("unique_id") or f"hash_{abs(hash(problem['problem']))}"

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
                temperature=0.7,  # Not Greedy for generation Diversity
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Check correctness
        ground_truth = problem.get("solution", problem.get("answer", ""))
        extracted_truth = filter_obj.extract_answer(ground_truth) or ground_truth
        is_correct = filter_obj.check_correctness(response, extracted_truth)

        results[problem_id] = is_correct
        if response_cache_file:
            responses[problem_id] = {
                "response": response,
                "correct": is_correct,
                "ground_truth": extracted_truth,
            }
        evaluated_new += 1

        # Auto-save every save_interval new problems evaluated
        if cache_file and evaluated_new % save_interval == 0:
            with open(cache_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Auto-saved results after {evaluated_new} new problems evaluated (total: {len(results)})")
        if response_cache_file and evaluated_new % save_interval == 0:
            with open(response_cache_file, "w") as f:
                json.dump(responses, f, indent=2)
            logger.info(f"Auto-saved responses after {evaluated_new} new problems evaluated (total: {len(responses)})")

        if max_questions is not None and evaluated_new >= max_questions:
            logger.info(f"Reached max_questions={max_questions}, stopping early")
            break

    # Final save
    if cache_file:
        with open(cache_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Final save: {len(results)} results saved to {cache_file}")
    if response_cache_file:
        with open(response_cache_file, "w") as f:
            json.dump(responses, f, indent=2)
        logger.info(
            f"Final save: {len(responses)} responses saved to {response_cache_file}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Prepare filtered MATH dataset")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--output-dir", type=str, default="data/filtered", help="Output directory")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation, use cached results")
    parser.add_argument("--skip-dpo", action="store_true", help="Skip DPO evaluation, use cached results")
    parser.add_argument("--skip-rlvr", action="store_true", help="Skip RLVR evaluation, use cached results")
    parser.add_argument(
        "--model",
        choices=["dpo", "rlvr", "both"],
        default="both",
        help="Which model(s) to evaluate",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=50,
        help="Max number of new problems to evaluate per run (<=0 means all)",
    )
    parser.add_argument(
        "--rlvr-checkpoint",
        type=str,
        default="final",
        help="RLVR checkpoint name (e.g., rlvr_step_1000) or 'final'",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=None,
        help="Optional prompt template override",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Specify GPU(s) to use. Examples: '0' for cuda:0, '0,1,2' for multiple GPUs, 'cuda:1' for specific device",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override GPU settings from command line if provided
    if args.gpu:
        # Try parsing as direct device string first (e.g., "cuda:1", "cpu")
        if args.gpu.startswith("cuda:") or args.gpu == "cpu":
            config["hardware"]["device"] = args.gpu
            config["hardware"]["multi_gpu"] = {"enabled": False}
            logger.info(f"Using device: {args.gpu}")
        else:
            # Try parsing as comma-separated GPU IDs
            try:
                gpu_ids = [int(x.strip()) for x in args.gpu.split(",") if x.strip().isdigit()]
                if len(gpu_ids) == 1:
                    # Single GPU: use cuda:X format
                    config["hardware"]["device"] = f"cuda:{gpu_ids[0]}"
                    config["hardware"]["multi_gpu"] = {"enabled": False}
                    logger.info(f"Using single GPU: cuda:{gpu_ids[0]}")
                elif len(gpu_ids) > 1:
                    # Multiple GPUs: use auto/balanced mode
                    config["hardware"]["device"] = "auto"
                    if "multi_gpu" not in config["hardware"]:
                        config["hardware"]["multi_gpu"] = {}
                    config["hardware"]["multi_gpu"]["enabled"] = True
                    config["hardware"]["multi_gpu"]["gpu_ids"] = gpu_ids
                    logger.info(f"Using multiple GPUs: {gpu_ids}")
                else:
                    logger.warning(f"Could not parse GPU specification '{args.gpu}', using config default")
            except ValueError:
                logger.warning(f"Invalid GPU specification '{args.gpu}', using config default")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info("Loading MATH dataset...")
    dataset = MATHDataset(
        subjects=config["dataset"].get("subjects", ["all"]),
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
    dpo_responses_cache = output_dir / "dpo_responses.json"
    rlvr_responses_cache = output_dir / "rlvr_responses.json"

    dpo_results = None
    rlvr_results = None

    if args.skip_eval and dpo_cache.exists() and rlvr_cache.exists():
        logger.info("Loading cached evaluation results...")
        dpo_results = _load_results_cache(dpo_cache)
        rlvr_results = _load_results_cache(rlvr_cache)
    else:
        max_questions = None if args.max_questions <= 0 else args.max_questions

        # Load models
        logger.info("Loading models for evaluation...")
        loader = ModelLoader(
            device=config["hardware"]["device"],
            dtype=getattr(torch, config["hardware"]["dtype"]),
            multi_gpu_config=config["hardware"].get("multi_gpu"),
        )
        loader.register_from_config(config)

        tokenizer = loader.load_tokenizer()
        preprocessor = DataPreprocessor(tokenizer, prompt_template=args.prompt_template)

        # Evaluate DPO model
        if args.model in ("dpo", "both"):
            if args.skip_dpo and dpo_cache.exists():
                logger.info("Skipping DPO evaluation, loading cached results...")
                dpo_results = _load_results_cache(dpo_cache)
            else:
                logger.info("Evaluating DPO model...")
                dpo_model = loader.load_model("dpo")
                dpo_results = evaluate_model(
                    dpo_model,
                    tokenizer,
                    problems,
                    preprocessor,
                    filter_obj,
                    cache_file=dpo_cache,
                    response_cache_file=dpo_responses_cache,
                    max_questions=max_questions,
                )
                loader.unload_model("dpo")

        # Evaluate RLVR model
        if args.model in ("rlvr", "both"):
            if args.skip_rlvr and rlvr_cache.exists():
                logger.info("Skipping RLVR evaluation, loading cached results...")
                rlvr_results = _load_results_cache(rlvr_cache)
            else:
                logger.info("Evaluating RLVR model...")
                rlvr_checkpoints = loader.get_rlvr_checkpoints()
                if not rlvr_checkpoints:
                    raise ValueError("No RLVR checkpoints registered in config")
                if args.rlvr_checkpoint == "final":
                    selected_rlvr = rlvr_checkpoints[-1]
                else:
                    if args.rlvr_checkpoint not in rlvr_checkpoints:
                        raise ValueError(
                            f"RLVR checkpoint '{args.rlvr_checkpoint}' not found. "
                            f"Available: {rlvr_checkpoints}"
                        )
                    selected_rlvr = args.rlvr_checkpoint
                rlvr_model = loader.load_model(selected_rlvr)
                rlvr_results = evaluate_model(
                    rlvr_model,
                    tokenizer,
                    problems,
                    preprocessor,
                    filter_obj,
                    cache_file=rlvr_cache,
                    response_cache_file=rlvr_responses_cache,
                    max_questions=max_questions,
                )
                loader.unload_model(selected_rlvr)

    # Load any missing results from cache if available
    if dpo_results is None and dpo_cache.exists():
        dpo_results = _load_results_cache(dpo_cache)
    if rlvr_results is None and rlvr_cache.exists():
        rlvr_results = _load_results_cache(rlvr_cache)

    if dpo_results and rlvr_results:
        # Filter problems
        logger.info("Filtering problems (DPO-wrong, RLVR-right)...")
        filtered_dataset = FilteredMATHDataset(dataset)
        filtered_problems = filtered_dataset.filter_by_model_performance(dpo_results, rlvr_results)

        # Save filtered problems
        filtered_path = output_dir / "filtered_problems.json"
        filtered_dataset.save_to_cache(str(filtered_path))
    else:
        filtered_problems = []
        filtered_path = output_dir / "filtered_problems.json"
        logger.info(
            "Skipping filtering: both DPO and RLVR results are required. "
            f"DPO results: {bool(dpo_results)}, RLVR results: {bool(rlvr_results)}"
        )

    # Print summary
    dpo_correct = sum(dpo_results.values()) if dpo_results else 0
    rlvr_correct = sum(rlvr_results.values()) if rlvr_results else 0
    dpo_total = len(dpo_results) if dpo_results else 0
    rlvr_total = len(rlvr_results) if rlvr_results else 0

    logger.info("=" * 50)
    logger.info("Data Preparation Summary")
    logger.info("=" * 50)
    logger.info(f"Total problems: {len(problems)}")
    if dpo_results:
        logger.info(f"DPO correct: {dpo_correct} ({dpo_correct/dpo_total*100:.1f}%)")
    else:
        logger.info("DPO correct: n/a (no DPO results)")
    if rlvr_results:
        logger.info(f"RLVR correct: {rlvr_correct} ({rlvr_correct/rlvr_total*100:.1f}%)")
    else:
        logger.info("RLVR correct: n/a (no RLVR results)")
    if dpo_results and rlvr_results:
        logger.info(f"Filtered (DPO-wrong, RLVR-right): {len(filtered_problems)}")
        logger.info(f"Saved to: {filtered_path}")


if __name__ == "__main__":
    main()
