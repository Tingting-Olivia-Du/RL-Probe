"""
KL-Divergence Computation

Core module for computing token-level KL divergence between model checkpoints.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class KLResult:
    """Container for KL divergence results."""

    # Token-level KL values
    kl_forward: torch.Tensor  # D_KL(P || Q) - RLVR || DPO
    kl_reverse: Optional[torch.Tensor] = None  # D_KL(Q || P) - DPO || RLVR
    js_divergence: Optional[torch.Tensor] = None  # Jensen-Shannon

    # Metadata
    token_ids: Optional[torch.Tensor] = None
    token_strings: Optional[List[str]] = None
    prompt_length: int = 0

    # Logits for further analysis
    logits_p: Optional[torch.Tensor] = None
    logits_q: Optional[torch.Tensor] = None


class KLDivergenceAnalyzer:
    """
    Computes token-level KL divergence between model distributions.

    Supports forward KL, reverse KL, and Jensen-Shannon divergence
    for comprehensive analysis of distribution shifts.
    """

    def __init__(
        self,
        epsilon: float = 1e-10,
        compute_forward: bool = True,
        compute_reverse: bool = True,
        compute_js: bool = True,
    ):
        """
        Initialize KL divergence analyzer.

        Args:
            epsilon: Small value for numerical stability
            compute_forward: Whether to compute D_KL(P || Q)
            compute_reverse: Whether to compute D_KL(Q || P)
            compute_js: Whether to compute Jensen-Shannon divergence
        """
        self.epsilon = epsilon
        self.compute_forward = compute_forward
        self.compute_reverse = compute_reverse
        self.compute_js = compute_js

    def get_logits(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract logits from a model for given input.

        Args:
            model: The language model
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask

        Returns:
            Logits tensor [batch, seq_len, vocab_size]
        """
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        return outputs.logits

    def compute_kl_divergence(
        self,
        logits_p: torch.Tensor,
        logits_q: torch.Tensor,
        direction: str = "forward",
    ) -> torch.Tensor:
        """
        Compute KL divergence between two logit distributions.

        Args:
            logits_p: Logits from model P [batch, seq_len, vocab_size]
            logits_q: Logits from model Q [batch, seq_len, vocab_size]
            direction: "forward" for D_KL(P||Q), "reverse" for D_KL(Q||P)

        Returns:
            Token-level KL divergence [batch, seq_len]
        """
        # Convert logits to log probabilities
        log_probs_p = F.log_softmax(logits_p, dim=-1)
        log_probs_q = F.log_softmax(logits_q, dim=-1)

        # Convert to probabilities for KL computation
        probs_p = torch.exp(log_probs_p)
        probs_q = torch.exp(log_probs_q)

        # Add epsilon for numerical stability
        probs_p = probs_p + self.epsilon
        probs_q = probs_q + self.epsilon

        # Renormalize
        probs_p = probs_p / probs_p.sum(dim=-1, keepdim=True)
        probs_q = probs_q / probs_q.sum(dim=-1, keepdim=True)

        if direction == "forward":
            # D_KL(P || Q) = sum_x P(x) * log(P(x) / Q(x))
            kl = (probs_p * (torch.log(probs_p) - torch.log(probs_q))).sum(dim=-1)
        else:
            # D_KL(Q || P) = sum_x Q(x) * log(Q(x) / P(x))
            kl = (probs_q * (torch.log(probs_q) - torch.log(probs_p))).sum(dim=-1)

        # Ensure KL values are float32 (not bfloat16) for numpy compatibility
        if kl.dtype == torch.bfloat16:
            kl = kl.float()
        
        return kl

    def compute_js_divergence(
        self,
        logits_p: torch.Tensor,
        logits_q: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Jensen-Shannon divergence (symmetric).

        JS(P||Q) = 0.5 * D_KL(P||M) + 0.5 * D_KL(Q||M)
        where M = 0.5 * (P + Q)

        Args:
            logits_p: Logits from model P
            logits_q: Logits from model Q

        Returns:
            Token-level JS divergence [batch, seq_len]
        """
        probs_p = F.softmax(logits_p, dim=-1) + self.epsilon
        probs_q = F.softmax(logits_q, dim=-1) + self.epsilon

        # Renormalize
        probs_p = probs_p / probs_p.sum(dim=-1, keepdim=True)
        probs_q = probs_q / probs_q.sum(dim=-1, keepdim=True)

        # Compute mixture M
        m = 0.5 * (probs_p + probs_q)

        # Compute KL divergences
        kl_pm = (probs_p * (torch.log(probs_p) - torch.log(m))).sum(dim=-1)
        kl_qm = (probs_q * (torch.log(probs_q) - torch.log(m))).sum(dim=-1)

        # JS divergence
        js = 0.5 * (kl_pm + kl_qm)

        # Ensure JS values are float32 (not bfloat16) for numpy compatibility
        if js.dtype == torch.bfloat16:
            js = js.float()
        
        return js

    def analyze_sequence(
        self,
        model_p: PreTrainedModel,
        model_q: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_length: int = 0,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> KLResult:
        """
        Analyze KL divergence for a full sequence.

        Args:
            model_p: First model (typically RLVR checkpoint)
            model_q: Second model (typically DPO baseline)
            input_ids: Input token IDs
            attention_mask: Attention mask
            prompt_length: Length of prompt (for masking)
            tokenizer: Optional tokenizer for token strings

        Returns:
            KLResult with all computed metrics
        """
        # Get logits from both models
        logits_p = self.get_logits(model_p, input_ids, attention_mask)
        logits_q = self.get_logits(model_q, input_ids, attention_mask)

        # Compute forward KL: D_KL(RLVR || DPO)
        kl_forward = None
        if self.compute_forward:
            kl_forward = self.compute_kl_divergence(logits_p, logits_q, "forward")

        # Compute reverse KL: D_KL(DPO || RLVR)
        kl_reverse = None
        if self.compute_reverse:
            kl_reverse = self.compute_kl_divergence(logits_p, logits_q, "reverse")

        # Compute JS divergence
        js_div = None
        if self.compute_js:
            js_div = self.compute_js_divergence(logits_p, logits_q)

        # Get token strings if tokenizer provided
        token_strings = None
        if tokenizer is not None:
            token_strings = [
                tokenizer.decode([tid]) for tid in input_ids.squeeze().tolist()
            ]

        # Ensure all tensors are float32 before returning (for numpy compatibility)
        if kl_forward is not None and kl_forward.dtype == torch.bfloat16:
            kl_forward = kl_forward.float()
        if kl_reverse is not None and kl_reverse.dtype == torch.bfloat16:
            kl_reverse = kl_reverse.float()
        if js_div is not None and js_div.dtype == torch.bfloat16:
            js_div = js_div.float()
        
        return KLResult(
            kl_forward=kl_forward.squeeze() if kl_forward is not None else None,
            kl_reverse=kl_reverse.squeeze() if kl_reverse is not None else None,
            js_divergence=js_div.squeeze() if js_div is not None else None,
            token_ids=input_ids.squeeze(),
            token_strings=token_strings,
            prompt_length=prompt_length,
            logits_p=logits_p.squeeze(),
            logits_q=logits_q.squeeze(),
        )

    def analyze_rollout(
        self,
        model_p: PreTrainedModel,
        model_q: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompt: str,
        rollout: str,
    ) -> KLResult:
        """
        Analyze KL divergence for a prompt + rollout sequence.

        This is the main method for teacher forcing analysis.

        Args:
            model_p: RLVR checkpoint
            model_q: DPO model
            tokenizer: Tokenizer
            prompt: The input prompt
            rollout: The generated rollout to analyze

        Returns:
            KLResult with analysis
        """
        # Tokenize prompt to get its length
        prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        prompt_length = prompt_tokens["input_ids"].shape[1]

        # Tokenize full sequence
        full_text = prompt + rollout
        full_tokens = tokenizer(full_text, return_tensors="pt", add_special_tokens=True)

        # Move to same device as model
        device = next(model_p.parameters()).device
        input_ids = full_tokens["input_ids"].to(device)
        attention_mask = full_tokens["attention_mask"].to(device)

        return self.analyze_sequence(
            model_p=model_p,
            model_q=model_q,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_length=prompt_length,
            tokenizer=tokenizer,
        )

    def batch_analyze(
        self,
        model_p: PreTrainedModel,
        model_q: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompts: List[str],
        rollouts: List[str],
        batch_size: int = 4,
    ) -> List[KLResult]:
        """
        Analyze KL divergence for multiple rollouts.

        Args:
            model_p: RLVR checkpoint
            model_q: DPO model
            tokenizer: Tokenizer
            prompts: List of prompts
            rollouts: List of rollouts
            batch_size: Batch size for processing

        Returns:
            List of KLResult objects
        """
        results = []

        for i in tqdm(range(0, len(prompts), batch_size), desc="Analyzing"):
            batch_prompts = prompts[i : i + batch_size]
            batch_rollouts = rollouts[i : i + batch_size]

            for prompt, rollout in zip(batch_prompts, batch_rollouts):
                result = self.analyze_rollout(
                    model_p=model_p,
                    model_q=model_q,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    rollout=rollout,
                )
                results.append(result)

        return results
