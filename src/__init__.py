"""
RL-Probe: Token-Level KL-Divergence Analysis for RL Reasoning Evolution

This package provides tools for analyzing how RLVR training reshapes
LLM reasoning by computing token-level KL divergence across checkpoints.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from src.analysis.kl_divergence import KLDivergenceAnalyzer
from src.analysis.spike_detector import SpikeDetector
from src.data.dataset import MATHDataset
from src.models.model_loader import ModelLoader

__all__ = [
    "KLDivergenceAnalyzer",
    "SpikeDetector",
    "MATHDataset",
    "ModelLoader",
]
