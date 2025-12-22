"""
Rollout Generation

Generate reasoning trajectories (rollouts) from models for analysis.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class RolloutGenerator:
    """
    Generates reasoning rollouts from language models.

    Produces multiple diverse rollouts per problem for comprehensive analysis.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
    ):
        """
        Initialize rollout generator.

        Args:
            model: The language model to generate from
            tokenizer: Tokenizer for the model
            device: Device for generation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_rollout(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate rollout(s) for a given prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            num_return_sequences: Number of rollouts to generate

        Returns:
            List of generated rollout strings
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode outputs, removing the prompt
        prompt_length = inputs["input_ids"].shape[1]
        rollouts = []

        for output in outputs:
            generated_tokens = output[prompt_length:]
            rollout = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )
            rollouts.append(rollout)

        return rollouts

    def generate_diverse_rollouts(
        self,
        prompt: str,
        num_samples: int = 5,
        temperatures: Optional[List[float]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate diverse rollouts using different temperatures.

        Args:
            prompt: Input prompt
            num_samples: Number of rollouts to generate
            temperatures: List of temperatures to use
            **kwargs: Additional generation parameters

        Returns:
            List of rollout dictionaries with metadata
        """
        if temperatures is None:
            # Use range of temperatures for diversity
            temperatures = [0.3, 0.5, 0.7, 0.9, 1.0][:num_samples]

        rollouts = []
        for i, temp in enumerate(temperatures[:num_samples]):
            rollout = self.generate_rollout(
                prompt=prompt,
                temperature=temp,
                do_sample=True,
                num_return_sequences=1,
                **kwargs,
            )[0]

            rollouts.append({
                "rollout": rollout,
                "temperature": temp,
                "sample_idx": i,
            })

        return rollouts

    def generate_for_problems(
        self,
        problems: List[Dict[str, Any]],
        prompt_template: str,
        num_samples_per_problem: int = 5,
        save_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate rollouts for a list of problems.

        Args:
            problems: List of problem dictionaries
            prompt_template: Template for formatting prompts
            num_samples_per_problem: Rollouts per problem
            save_path: Optional path to save results
            **kwargs: Additional generation parameters

        Returns:
            Dictionary mapping problem IDs to rollout lists
        """
        all_rollouts = {}

        for problem in tqdm(problems, desc="Generating rollouts"):
            problem_id = problem.get("unique_id", str(hash(problem["problem"])))
            prompt = prompt_template.format(problem=problem["problem"])

            rollouts = self.generate_diverse_rollouts(
                prompt=prompt,
                num_samples=num_samples_per_problem,
                **kwargs,
            )

            # Add problem metadata to each rollout
            for rollout in rollouts:
                rollout["problem_id"] = problem_id
                rollout["problem_text"] = problem["problem"]

            all_rollouts[problem_id] = rollouts

        # Save if path provided
        if save_path:
            self._save_rollouts(all_rollouts, save_path)

        return all_rollouts

    def _save_rollouts(
        self,
        rollouts: Dict[str, List[Dict[str, Any]]],
        save_path: str,
    ) -> None:
        """Save rollouts to JSON file."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(rollouts, f, indent=2)
        logger.info(f"Saved rollouts to {save_path}")

    @staticmethod
    def load_rollouts(load_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Load rollouts from JSON file."""
        with open(load_path, "r") as f:
            return json.load(f)
