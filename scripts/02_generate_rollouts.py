#!/usr/bin/env python3
"""
Step 2: Generate Rollouts

Generate diverse reasoning trajectories (rollouts) from the DPO model
for the filtered problems.

Multiple rollouts per problem enable analysis of:
- Consistency of spike positions across different error paths
- Diversity of reasoning approaches
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from src.data.preprocessor import DataPreprocessor
from src.models.model_loader import ModelLoader
from src.models.rollout_generator import RolloutGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate DPO rollouts")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--problems", type=str, default="data/filtered/filtered_problems.json", help="Filtered problems path")
    parser.add_argument("--output-dir", type=str, default="rollouts/dpo_errors", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=None, help="Override samples per problem")
    parser.add_argument("--max-problems", type=int, default=None, help="Limit number of problems")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override config if specified
    num_samples = args.num_samples or config["rollout"]["num_samples_per_problem"]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load filtered problems
    logger.info(f"Loading problems from {args.problems}...")
    with open(args.problems) as f:
        problems = json.load(f)

    if args.max_problems:
        problems = problems[:args.max_problems]

    logger.info(f"Generating rollouts for {len(problems)} problems, {num_samples} samples each")

    # Load model
    logger.info("Loading DPO model...")
    loader = ModelLoader(
        device=config["hardware"]["device"],
        dtype=getattr(torch, config["hardware"]["dtype"]),
    )
    loader.register_from_config(config)

    tokenizer = loader.load_tokenizer()
    dpo_model = loader.load_model("dpo")

    # Initialize generator
    generator = RolloutGenerator(
        model=dpo_model,
        tokenizer=tokenizer,
        device=config["hardware"]["device"],
    )

    # Prepare prompt template
    preprocessor = DataPreprocessor(tokenizer)
    prompt_template = preprocessor.prompt_template

    # Generate rollouts
    all_rollouts = {}

    for problem in tqdm(problems, desc="Generating rollouts"):
        problem_id = problem.get("unique_id", str(hash(problem["problem"])))
        prompt = prompt_template.format(problem=problem["problem"])

        rollouts = generator.generate_diverse_rollouts(
            prompt=prompt,
            num_samples=num_samples,
            max_new_tokens=config["rollout"]["generation"]["max_new_tokens"],
        )

        # Add problem info
        for rollout in rollouts:
            rollout["problem_id"] = problem_id
            rollout["problem_text"] = problem["problem"]
            rollout["ground_truth"] = problem.get("solution", "")

        all_rollouts[problem_id] = rollouts

        # Save incrementally
        if len(all_rollouts) % 10 == 0:
            with open(output_dir / "rollouts_checkpoint.json", "w") as f:
                json.dump(all_rollouts, f, indent=2)

    # Save final results
    output_path = output_dir / "all_rollouts.json"
    with open(output_path, "w") as f:
        json.dump(all_rollouts, f, indent=2)

    logger.info("=" * 50)
    logger.info("Rollout Generation Summary")
    logger.info("=" * 50)
    logger.info(f"Problems processed: {len(problems)}")
    logger.info(f"Total rollouts: {sum(len(r) for r in all_rollouts.values())}")
    logger.info(f"Samples per problem: {num_samples}")
    logger.info(f"Saved to: {output_path}")

    # Cleanup
    loader.unload_all()


if __name__ == "__main__":
    main()
