"""
Model Loading Utilities

Handles loading and managing multiple model checkpoints for comparison.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)


@dataclass
class ModelCheckpoint:
    """Container for a model checkpoint."""

    name: str
    path: str
    step: Optional[int] = None
    stage: Optional[str] = None  # "sft", "dpo", "early", "mid", "final"
    model: Optional[PreTrainedModel] = None
    loaded: bool = False


class ModelLoader:
    """
    Manages loading and caching of multiple model checkpoints.

    Efficiently loads models for comparison across training stages.
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize model loader.

        Args:
            device: Device to load models on
            dtype: Data type for model weights
            cache_dir: Directory for model cache
        """
        self.device = device
        self.dtype = dtype
        self.cache_dir = cache_dir
        self.checkpoints: Dict[str, ModelCheckpoint] = {}
        self.tokenizer = None

    def register_checkpoint(
        self,
        name: str,
        path: str,
        step: Optional[int] = None,
        stage: Optional[str] = None,
    ) -> None:
        """
        Register a checkpoint for later loading.

        Args:
            name: Unique identifier for this checkpoint
            path: HuggingFace model path or local path
            step: Training step number
            stage: Training stage identifier
        """
        self.checkpoints[name] = ModelCheckpoint(
            name=name,
            path=path,
            step=step,
            stage=stage,
        )
        logger.info(f"Registered checkpoint: {name} ({path})")

    def register_from_config(self, config: Dict[str, Any]) -> None:
        """
        Register all checkpoints from configuration.

        Args:
            config: Configuration dictionary with model specifications
        """
        models_config = config.get("models", {})

        # Register base DPO model
        if "base_model" in models_config:
            self.register_checkpoint(
                name="dpo",
                path=models_config["base_model"],
                stage="dpo",
            )

        # Register SFT model if specified
        if "sft_model" in models_config:
            self.register_checkpoint(
                name="sft",
                path=models_config["sft_model"],
                stage="sft",
            )

        # Register RLVR checkpoints
        for ckpt in models_config.get("rlvr_checkpoints", []):
            name = f"rlvr_step_{ckpt['step']}"
            self.register_checkpoint(
                name=name,
                path=ckpt["path"],
                step=ckpt["step"],
                stage=ckpt["stage"],
            )

    def load_tokenizer(self, model_path: Optional[str] = None) -> AutoTokenizer:
        """
        Load tokenizer (shared across all checkpoints).

        Args:
            model_path: Path to load tokenizer from (uses first registered if None)

        Returns:
            Loaded tokenizer
        """
        if self.tokenizer is not None:
            return self.tokenizer

        if model_path is None:
            # Use first registered checkpoint
            if not self.checkpoints:
                raise ValueError("No checkpoints registered")
            model_path = list(self.checkpoints.values())[0].path

        logger.info(f"Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, name: str) -> PreTrainedModel:
        """
        Load a specific model checkpoint.

        Args:
            name: Name of registered checkpoint

        Returns:
            Loaded model
        """
        if name not in self.checkpoints:
            raise ValueError(f"Checkpoint '{name}' not registered")

        ckpt = self.checkpoints[name]

        if ckpt.loaded and ckpt.model is not None:
            logger.debug(f"Returning cached model: {name}")
            return ckpt.model

        logger.info(f"Loading model: {name} from {ckpt.path}")

        model = AutoModelForCausalLM.from_pretrained(
            ckpt.path,
            torch_dtype=self.dtype,
            device_map=self.device,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        model.eval()

        ckpt.model = model
        ckpt.loaded = True

        return model

    def load_all(self) -> Dict[str, PreTrainedModel]:
        """
        Load all registered checkpoints.

        Returns:
            Dictionary mapping names to loaded models
        """
        models = {}
        for name in self.checkpoints:
            models[name] = self.load_model(name)
        return models

    def unload_model(self, name: str) -> None:
        """
        Unload a model to free memory.

        Args:
            name: Name of checkpoint to unload
        """
        if name in self.checkpoints:
            ckpt = self.checkpoints[name]
            if ckpt.model is not None:
                del ckpt.model
                ckpt.model = None
                ckpt.loaded = False
                torch.cuda.empty_cache()
                logger.info(f"Unloaded model: {name}")

    def unload_all(self) -> None:
        """Unload all models to free memory."""
        for name in self.checkpoints:
            self.unload_model(name)

    def get_checkpoint_info(self, name: str) -> ModelCheckpoint:
        """Get information about a checkpoint."""
        return self.checkpoints.get(name)

    def list_checkpoints(self) -> List[str]:
        """List all registered checkpoint names."""
        return list(self.checkpoints.keys())

    def get_rlvr_checkpoints(self) -> List[str]:
        """Get names of RLVR checkpoints in order of training step."""
        rlvr_ckpts = [
            (name, ckpt)
            for name, ckpt in self.checkpoints.items()
            if ckpt.step is not None
        ]
        rlvr_ckpts.sort(key=lambda x: x[1].step)
        return [name for name, _ in rlvr_ckpts]
