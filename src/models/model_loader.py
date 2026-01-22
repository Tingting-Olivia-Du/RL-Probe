"""
Model Loading Utilities

Handles loading and managing multiple model checkpoints for comparison.
Supports multi-GPU configuration for large models.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)

def get_device_map_config(
    device: str,
    multi_gpu_config: Optional[Dict[str, Any]] = None,
) -> Union[str, Dict[str, Any]]:
    """
    根据配置生成设备映射策略或显存限制字典。
    """
    if multi_gpu_config is None or not multi_gpu_config.get("enabled", False):
        return device

    gpu_ids = multi_gpu_config.get("gpu_ids")
    max_memory_per_gpu = multi_gpu_config.get("max_memory_per_gpu")

    # 如果指定了 GPU ID 列表，设置环境变量
    if gpu_ids is not None and len(gpu_ids) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        logger.info(f"已设置 CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    if device in ("auto", "balanced", "sequential"):
        if max_memory_per_gpu:
            # 获取当前进程可见的 GPU 数量
            # 如果 CUDA_VISIBLE_DEVICES=4，这里返回 1
            num_gpus = torch.cuda.device_count()
            
            # 修复方案：GPU ID 使用 int，其他使用 str
            # 这样 accelerate 就能识别出 0 是一个 GPU 索引
            max_memory = {}
            for i in range(num_gpus):
                max_memory[i] = max_memory_per_gpu  # 键是 int 类型
            
            max_memory["cpu"] = multi_gpu_config.get("max_memory_cpu", "64GiB")
            
            logger.info(f"生成 max_memory 限制: {max_memory}")
            return max_memory
        return device

    return device

@dataclass
class ModelCheckpoint:
    """模型检查点容器。"""
    name: str
    path: str
    step: Optional[int] = None
    stage: Optional[str] = None  # "sft", "dpo", "early", "mid", "final"
    model: Optional[PreTrainedModel] = None
    loaded: bool = False


class ModelLoader:
    """
    管理多个模型检查点的加载和缓存。
    支持多 GPU 显存分配策略。
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
        multi_gpu_config: Optional[Dict[str, Any]] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.cache_dir = cache_dir
        self.multi_gpu_config = multi_gpu_config
        self.checkpoints: Dict[str, ModelCheckpoint] = {}
        self.tokenizer = None

        # 预先计算设备配置
        self.device_config = get_device_map_config(device, multi_gpu_config)

    def register_checkpoint(
        self,
        name: str,
        path: str,
        step: Optional[int] = None,
        stage: Optional[str] = None,
    ) -> None:
        self.checkpoints[name] = ModelCheckpoint(
            name=name, path=path, step=step, stage=stage
        )
        logger.info(f"已注册检查点: {name} ({path})")

    def register_from_config(self, config: Dict[str, Any]) -> None:
        models_config = config.get("models", {})
        if "base_model" in models_config:
            self.register_checkpoint("dpo", models_config["base_model"], stage="dpo")
        if "sft_model" in models_config:
            self.register_checkpoint("sft", models_config["sft_model"], stage="sft")
        for ckpt in models_config.get("rlvr_checkpoints", []):
            name = f"rlvr_step_{ckpt['step']}"
            self.register_checkpoint(name, ckpt["path"], step=ckpt["step"], stage=ckpt["stage"])

    def load_tokenizer(self, model_path: Optional[str] = None) -> AutoTokenizer:
        if self.tokenizer is not None:
            return self.tokenizer
        if model_path is None:
            if not self.checkpoints:
                raise ValueError("尚未注册任何检查点")
            model_path = list(self.checkpoints.values())[0].path

        logger.info(f"正在加载分词器: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, cache_dir=self.cache_dir, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, name: str) -> PreTrainedModel:
        """加载指定的模型检查点。"""
        if name not in self.checkpoints:
            raise ValueError(f"检查点 '{name}' 未注册")

        ckpt = self.checkpoints[name]
        if ckpt.loaded and ckpt.model is not None:
            logger.debug(f"返回已缓存的模型: {name}")
            return ckpt.model

        logger.info(f"正在加载模型: {name} 来自 {ckpt.path}")

        # 核心修复 2：正确分发 device_map 和 max_memory 参数
        if isinstance(self.device_config, dict):
            # 如果配置是字典，说明是显存限制，device_map 必须设为策略（如 "auto"）
            current_device_map = "auto"
            current_max_memory = self.device_config
        else:
            # 如果是字符串（如 "cuda:0"），直接作为 device_map
            current_device_map = self.device_config
            current_max_memory = None

        # 核心修复 3：low_cpu_mem_usage=True 解决 embed_tokens 分配失败问题
        model = AutoModelForCausalLM.from_pretrained(
            ckpt.path,
            torch_dtype=self.dtype,
            device_map=current_device_map,
            max_memory=current_max_memory,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True, 
        )
        model.eval()

        ckpt.model = model
        ckpt.loaded = True
        return model

    def unload_model(self, name: str) -> None:
        if name in self.checkpoints:
            ckpt = self.checkpoints[name]
            if ckpt.model is not None:
                del ckpt.model
                ckpt.model = None
                ckpt.loaded = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"已卸载模型: {name}")

    def unload_all(self) -> None:
        for name in list(self.checkpoints.keys()):
            self.unload_model(name)

    def get_rlvr_checkpoints(self) -> List[str]:
        rlvr_ckpts = [(n, c) for n, c in self.checkpoints.items() if c.step is not None]
        rlvr_ckpts.sort(key=lambda x: x[1].step)
        return [name for name, _ in rlvr_ckpts]