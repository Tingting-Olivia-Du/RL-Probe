#!/usr/bin/env python3
"""
Step 6: Evaluate Checkpoints 100-1000 on Filtered Problems

Evaluate RLVR checkpoints (steps 100-1000) on problems where DPO failed but RLVR succeeded.
This script generates responses for each checkpoint and tracks when problems start being solved correctly.

Usage:
    # Evaluate 10 checkpoints (100-1000) on GPU 5
    python scripts/06_evaluate_checkpoints_100_1000.py --gpu 5
    
    # Or specify custom checkpoints
    python scripts/06_evaluate_checkpoints_100_1000.py --gpu 5 --checkpoints 100,200,300,400,500,600,700,800,900,1000
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import yaml
from tqdm import tqdm

from src.data.dataset import MATHDataset
from src.data.filter import ProblemFilter
from src.data.preprocessor import DataPreprocessor
from src.models.model_loader import ModelLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_filtered_problems(filtered_problems_path: Path):
    """Load problems where DPO failed but RLVR succeeded."""
    with open(filtered_problems_path) as f:
        problems = json.load(f)
    
    # Filter for problems where DPO failed but RLVR succeeded
    filtered = [
        p for p in problems 
        if p.get("dpo_correct") == False and p.get("rlvr_correct") == True
    ]
    
    logger.info(f"Loaded {len(filtered)} problems where DPO failed but RLVR succeeded")
    return filtered


def evaluate_checkpoint(
    model,
    tokenizer,
    problems,
    preprocessor,
    filter_obj,
    checkpoint_name: str,
    max_new_tokens=2048,
    cache_file=None,
    response_cache_file=None,
    save_interval=10,
):
    """Evaluate a checkpoint on problems and return results."""
    # Load existing cache
    results = {}
    responses = {}
    if cache_file and cache_file.exists():
        with open(cache_file) as f:
            results = json.load(f)
        logger.info(f"Loaded {len(results)} existing results from {cache_file}")
    if response_cache_file and response_cache_file.exists():
        with open(response_cache_file) as f:
            responses = json.load(f)
        logger.info(f"Loaded {len(responses)} existing responses from {response_cache_file}")

    evaluated_new = 0

    for idx, problem in enumerate(tqdm(problems, desc=f"Evaluating {checkpoint_name}"), 1):
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
                temperature=0.7,
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
                "problem": problem["problem"],
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

    # Final save
    if cache_file:
        with open(cache_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Final save: {len(results)} results saved to {cache_file}")
    if response_cache_file:
        with open(response_cache_file, "w") as f:
            json.dump(responses, f, indent=2)
        logger.info(f"Final save: {len(responses)} responses saved to {response_cache_file}")

    return results, responses


def analyze_first_correct_checkpoint(all_results: dict, problem_ids: list):
    """Analyze which checkpoint first solved each problem correctly."""
    first_correct = {}
    
    # Sort checkpoints by step number
    sorted_checkpoints = sorted(
        all_results.keys(),
        key=lambda x: int(x.replace("rlvr_step_", ""))
    )
    
    for problem_id in problem_ids:
        for ckpt_name in sorted_checkpoints:
            if problem_id in all_results[ckpt_name] and all_results[ckpt_name][problem_id]:
                first_correct[problem_id] = ckpt_name
                break
    
    return first_correct


def generate_summary_report(all_results: dict, all_responses: dict, first_correct: dict, output_dir: Path):
    """Generate summary report of evaluation results."""
    report = []
    report.append("# Checkpoint Evaluation Report (Steps 100-1000)")
    report.append("")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("---")
    report.append("")
    
    # Summary statistics
    report.append("## Summary Statistics")
    report.append("")
    
    sorted_checkpoints = sorted(
        all_results.keys(),
        key=lambda x: int(x.replace("rlvr_step_", ""))
    )
    
    report.append("| Checkpoint | Step | Correct | Total | Accuracy |")
    report.append("|------------|------|---------|-------|----------|")
    
    total_problems = len(list(all_results.values())[0]) if all_results else 0
    
    for ckpt_name in sorted_checkpoints:
        results = all_results[ckpt_name]
        correct_count = sum(1 for v in results.values() if v)
        accuracy = (correct_count / total_problems * 100) if total_problems > 0 else 0
        step = ckpt_name.replace("rlvr_step_", "")
        report.append(f"| {ckpt_name} | {step} | {correct_count} | {total_problems} | {accuracy:.2f}% |")
    
    report.append("")
    
    # First correct checkpoint analysis
    report.append("## First Correct Checkpoint Analysis")
    report.append("")
    report.append("This section shows which checkpoint first solved each problem correctly.")
    report.append("")
    
    # Count problems solved at each checkpoint
    first_correct_counts = defaultdict(int)
    for ckpt_name in first_correct.values():
        first_correct_counts[ckpt_name] += 1
    
    report.append("| Checkpoint | Step | Problems First Solved | Percentage |")
    report.append("|------------|------|----------------------|------------|")
    total_solved = len(first_correct)
    for ckpt_name in sorted_checkpoints:
        step = ckpt_name.replace("rlvr_step_", "")
        count = first_correct_counts.get(ckpt_name, 0)
        percentage = (count / total_solved * 100) if total_solved > 0 else 0
        report.append(f"| {ckpt_name} | {step} | {count} | {percentage:.1f}% |")
    
    report.append("")
    
    # Cumulative accuracy progression
    report.append("## Cumulative Accuracy Progression")
    report.append("")
    report.append("Shows how accuracy improves as training progresses.")
    report.append("")
    report.append("| Checkpoint | Step | Cumulative Correct | Cumulative Accuracy |")
    report.append("|------------|------|-------------------|---------------------|")
    
    cumulative_correct = set()
    for ckpt_name in sorted_checkpoints:
        results = all_results[ckpt_name]
        correct_problems = {pid for pid, correct in results.items() if correct}
        cumulative_correct.update(correct_problems)
        cumulative_accuracy = (len(cumulative_correct) / total_problems * 100) if total_problems > 0 else 0
        step = ckpt_name.replace("rlvr_step_", "")
        report.append(f"| {ckpt_name} | {step} | {len(cumulative_correct)} | {cumulative_accuracy:.2f}% |")
    
    report.append("")
    
    # Problems solved at each checkpoint
    report.append("## Problems Solved at Each Checkpoint")
    report.append("")
    for ckpt_name in sorted_checkpoints:
        step = ckpt_name.replace("rlvr_step_", "")
        results = all_results[ckpt_name]
        correct_problems = [pid for pid, correct in results.items() if correct]
        
        report.append(f"### {ckpt_name} (Step {step})")
        report.append(f"**Correct:** {len(correct_problems)}/{len(results)} ({len(correct_problems)/len(results)*100:.2f}%)")
        report.append("")
        report.append("| Problem ID | Correct |")
        report.append("|------------|--------|")
        for pid in correct_problems[:20]:  # Show first 20
            report.append(f"| `{pid}` | ✓ |")
        if len(correct_problems) > 20:
            report.append(f"| ... | ({len(correct_problems) - 20} more) |")
        report.append("")
    
    # Problems never solved
    report.append("## Problems Never Solved")
    report.append("")
    all_correct_problems = set()
    for results in all_results.values():
        all_correct_problems.update(pid for pid, correct in results.items() if correct)
    
    all_problem_ids = set()
    if all_results:
        all_problem_ids = set(list(all_results.values())[0].keys())
    
    never_solved = all_problem_ids - all_correct_problems
    if never_solved:
        report.append(f"**Total:** {len(never_solved)} problems were never solved correctly by any checkpoint")
        report.append("")
        report.append("| Problem ID |")
        report.append("|------------|")
        for pid in list(never_solved)[:20]:  # Show first 20
            report.append(f"| `{pid}` |")
        if len(never_solved) > 20:
            report.append(f"| ... | ({len(never_solved) - 20} more) |")
    else:
        report.append("**All problems were solved correctly by at least one checkpoint!**")
    
    report.append("")
    
    # Save report
    report_path = output_dir / "checkpoint_evaluation_report_100_1000.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    logger.info(f"Saved evaluation report: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate RLVR checkpoints 100-1000 on filtered problems")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="100,200,300,400,500,600,700,800,900,1000",
        help="Comma-separated list of checkpoint steps to evaluate (default: 100-1000, 10 checkpoints)"
    )
    parser.add_argument(
        "--filtered-problems",
        type=str,
        default="data/filtered/filtered_problems.json",
        help="Path to filtered problems file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/checkpoint_evaluations_100_1000",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save results every N problems (default: 10)"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Specify GPU to use. Examples: '0' for cuda:0, '5' for cuda:5"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip checkpoints that already have complete results"
    )
    args = parser.parse_args()

    # Parse checkpoint steps
    checkpoint_steps = [int(s.strip()) for s in args.checkpoints.split(",") if s.strip().isdigit()]
    checkpoint_steps.sort()
    logger.info(f"Will evaluate {len(checkpoint_steps)} checkpoints: {checkpoint_steps}")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override GPU settings from command line if provided
    if args.gpu:
        if args.gpu.startswith("cuda:") or args.gpu == "cpu":
            config["hardware"]["device"] = args.gpu
            config["hardware"]["multi_gpu"] = {"enabled": False}
            logger.info(f"Using device: {args.gpu}")
        else:
            try:
                gpu_id = int(args.gpu.strip())
                config["hardware"]["device"] = f"cuda:{gpu_id}"
                config["hardware"]["multi_gpu"] = {"enabled": False}
                logger.info(f"Using GPU: cuda:{gpu_id}")
            except ValueError:
                logger.warning(f"Could not parse GPU specification '{args.gpu}', using config default")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load filtered problems
    filtered_problems_path = Path(args.filtered_problems)
    if not filtered_problems_path.exists():
        raise FileNotFoundError(f"Filtered problems file not found: {filtered_problems_path}")
    
    problems = load_filtered_problems(filtered_problems_path)
    logger.info(f"Loaded {len(problems)} problems to evaluate")

    # Initialize components
    preprocessor = DataPreprocessor(config.get("prompt_template", None))
    filter_obj = ProblemFilter()

    # Initialize model loader
    model_loader = ModelLoader(
        device=config["hardware"]["device"],
        dtype=getattr(torch, config["hardware"].get("dtype", "bfloat16")),
        cache_dir=config.get("cache_dir"),
        multi_gpu_config=config["hardware"].get("multi_gpu"),
    )
    model_loader.register_from_config(config)

    # Load tokenizer (use first checkpoint's tokenizer)
    tokenizer = model_loader.load_tokenizer()

    # Evaluate each checkpoint
    all_results = {}
    all_responses = {}

    for step in checkpoint_steps:
        ckpt_name = f"rlvr_step_{step}"
        
        # Check if checkpoint exists in config
        if ckpt_name not in model_loader.checkpoints:
            logger.warning(f"Checkpoint {ckpt_name} not found in config, skipping...")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {ckpt_name} ({checkpoint_steps.index(step) + 1}/{len(checkpoint_steps)})")
        logger.info(f"{'='*60}")

        # Check if results already exist
        cache_file = output_dir / f"{ckpt_name}_results.json"
        response_cache_file = output_dir / f"{ckpt_name}_responses.json"
        
        if args.skip_existing and cache_file.exists():
            with open(cache_file) as f:
                existing_results = json.load(f)
            if len(existing_results) >= len(problems):
                logger.info(f"Skipping {ckpt_name} - already has complete results ({len(existing_results)}/{len(problems)})")
                all_results[ckpt_name] = existing_results
                if response_cache_file.exists():
                    with open(response_cache_file) as f:
                        all_responses[ckpt_name] = json.load(f)
                continue

        # Load model
        try:
            model = model_loader.load_model(ckpt_name)
            logger.info(f"Loaded model: {ckpt_name}")
        except Exception as e:
            logger.error(f"Failed to load model {ckpt_name}: {e}")
            continue

        # Evaluate
        try:
            results, responses = evaluate_checkpoint(
                model=model,
                tokenizer=tokenizer,
                problems=problems,
                preprocessor=preprocessor,
                filter_obj=filter_obj,
                checkpoint_name=ckpt_name,
                max_new_tokens=args.max_new_tokens,
                cache_file=cache_file,
                response_cache_file=response_cache_file,
                save_interval=args.save_interval,
            )
            
            all_results[ckpt_name] = results
            all_responses[ckpt_name] = responses
            
            # Calculate accuracy
            correct_count = sum(1 for v in results.values() if v)
            accuracy = (correct_count / len(results) * 100) if results else 0
            logger.info(f"{ckpt_name} Accuracy: {correct_count}/{len(results)} ({accuracy:.2f}%)")
            
        except Exception as e:
            logger.error(f"Error evaluating {ckpt_name}: {e}", exc_info=True)
        finally:
            # Unload model to free GPU memory
            model_loader.unload_model(ckpt_name)
            torch.cuda.empty_cache()

    # Analyze results
    if all_results:
        problem_ids = [p.get("unique_id") or f"hash_{abs(hash(p['problem']))}" for p in problems]
        
        first_correct = analyze_first_correct_checkpoint(all_results, problem_ids)
        
        # Generate summary report
        report_path = generate_summary_report(all_results, all_responses, first_correct, output_dir)
        
        # Save combined results
        combined_results_file = output_dir / "all_checkpoint_results.json"
        with open(combined_results_file, "w") as f:
            json.dump({
                "all_results": all_results,
                "first_correct": first_correct,
                "checkpoints_evaluated": sorted(all_results.keys()),
                "total_problems": len(problems),
            }, f, indent=2)
        logger.info(f"Saved combined results: {combined_results_file}")
        
        logger.info("\n" + "="*60)
        logger.info("Evaluation Complete!")
        logger.info("="*60)
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Report: {report_path}")
        
        # Print summary
        logger.info("\nSummary:")
        sorted_ckpts = sorted(all_results.keys(), key=lambda x: int(x.replace("rlvr_step_", "")))
        for ckpt_name in sorted_ckpts:
            results = all_results[ckpt_name]
            correct_count = sum(1 for v in results.values() if v)
            accuracy = (correct_count / len(results) * 100) if results else 0
            logger.info(f"  {ckpt_name}: {correct_count}/{len(results)} ({accuracy:.2f}%)")


if __name__ == "__main__":
    main()
