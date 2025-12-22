"""
Data Preprocessing Utilities

Handles tokenization, prompt formatting, and data preparation
for KL-divergence analysis.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocessor for preparing data for KL-divergence analysis.

    Handles prompt formatting, tokenization, and batch preparation.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 4096,
        prompt_template: Optional[str] = None,
    ):
        """
        Initialize preprocessor.

        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            prompt_template: Custom prompt template (uses default if None)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Default Tulu-style prompt template
        self.prompt_template = prompt_template or (
            "<|user|>\n{problem}\n<|assistant|>\n"
        )

    def format_prompt(self, problem: str) -> str:
        """
        Format a problem into a prompt.

        Args:
            problem: The math problem text

        Returns:
            Formatted prompt string
        """
        return self.prompt_template.format(problem=problem)

    def tokenize(
        self,
        text: str,
        return_tensors: str = "pt",
        add_special_tokens: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize text input.

        Args:
            text: Input text
            return_tensors: Tensor format ("pt" for PyTorch)
            add_special_tokens: Whether to add special tokens

        Returns:
            Dictionary with input_ids and attention_mask
        """
        return self.tokenizer(
            text,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )

    def prepare_teacher_forcing_input(
        self,
        prompt: str,
        rollout: str,
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Prepare input for teacher forcing (feeding pre-generated rollout).

        Args:
            prompt: The formatted prompt
            rollout: The pre-generated response to force

        Returns:
            Tuple of (tokenized input, prompt_length in tokens)
        """
        # Tokenize prompt to find its length
        prompt_tokens = self.tokenize(prompt, add_special_tokens=True)
        prompt_length = prompt_tokens["input_ids"].shape[1]

        # Tokenize full sequence (prompt + rollout)
        full_text = prompt + rollout
        full_tokens = self.tokenize(full_text, add_special_tokens=True)

        return full_tokens, prompt_length

    def prepare_batch(
        self,
        problems: List[Dict[str, Any]],
        rollouts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Prepare a batch of problems for analysis.

        Args:
            problems: List of problem dictionaries
            rollouts: Optional list of pre-generated rollouts

        Returns:
            Batch dictionary ready for model input
        """
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "prompt_lengths": [],
            "problem_ids": [],
        }

        for i, problem in enumerate(problems):
            prompt = self.format_prompt(problem["problem"])

            if rollouts and i < len(rollouts):
                tokens, prompt_len = self.prepare_teacher_forcing_input(
                    prompt, rollouts[i]
                )
            else:
                tokens = self.tokenize(prompt)
                prompt_len = tokens["input_ids"].shape[1]

            batch["input_ids"].append(tokens["input_ids"].squeeze(0))
            batch["attention_mask"].append(tokens["attention_mask"].squeeze(0))
            batch["prompt_lengths"].append(prompt_len)
            batch["problem_ids"].append(
                problem.get("unique_id", str(hash(problem["problem"])))
            )

        return batch

    def decode_tokens(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_token_strings(
        self,
        token_ids: Union[List[int], torch.Tensor],
    ) -> List[str]:
        """
        Get individual token strings for each token ID.

        Useful for visualization and analysis of specific tokens.

        Args:
            token_ids: Token IDs

        Returns:
            List of token strings
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        return [self.tokenizer.decode([tid]) for tid in token_ids]
