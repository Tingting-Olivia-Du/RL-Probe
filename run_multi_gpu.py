#!/usr/bin/env python3
"""
Multi-GPU Run Script

快速指定多GPU并运行RL-Probe流程的便捷脚本。

用法示例:
    # 使用GPU 0,1,2
    python run_multi_gpu.py --gpus 0,4,5

    # 使用GPU 0,1,2,3，每个GPU限制30GB显存
    python run_multi_gpu.py --gpus 4,5 --max-memory 40GB

    # 使用所有可用GPU，自动均衡模式
    python run_multi_gpu.py --gpus all --device auto

    # 只运行特定步骤
    python run_multi_gpu.py --gpus 0,1 --steps prepare,rollout

    # 干运行模式（只显示配置不实际运行）
    python run_multi_gpu.py --gpus 0,1,2 --dry-run
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
import yaml

def setup_logging(log_dir: str = None):
    """设置日志记录到文件和控制台"""
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"run_multi_gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    else:
        log_file = None

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"日志保存至: {log_file}")

    return logger, log_file

logger, log_file = setup_logging()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-GPU launcher for RL-Probe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用GPU 0,1,2
  %(prog)s --gpus 0,1,2

  # 使用GPU 0,1,2,3，每个GPU限制30GB显存
  %(prog)s --gpus 0,1,2,3 --max-memory 30GiB

  # 使用所有可用GPU
  %(prog)s --gpus all

  # 只运行数据准备和rollout生成
  %(prog)s --gpus 0,1 --steps prepare,rollout

  # 查看配置但不实际运行
  %(prog)s --gpus 0,1,2 --dry-run
        """,
    )

    # GPU配置
    parser.add_argument(
        "--gpus",
        type=str,
        required=True,
        help='GPU IDs (逗号分隔，如 "0,1,2" 或 "all" 使用所有GPU)',
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "balanced", "sequential", "cuda"],
        help="设备分配模式 (默认: auto)",
    )
    parser.add_argument(
        "--max-memory",
        type=str,
        default=None,
        help='每个GPU最大显存 (如 "30GiB", 默认: 不限制)',
    )

    # 配置文件
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径 (默认: configs/config.yaml)",
    )
    parser.add_argument(
        "--output-config",
        type=str,
        default=None,
        help="生成的配置文件保存路径 (默认: configs/config_multi_gpu.yaml)",
    )

    # 运行步骤
    parser.add_argument(
        "--steps",
        type=str,
        default="all",
        help='运行的步骤 (逗号分隔: prepare,rollout,kl,visualize 或 "all")',
    )

    # 其他选项
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干运行模式：只显示配置不实际运行",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细输出",
    )

    return parser.parse_args()


def parse_gpu_ids(gpu_str: str) -> Optional[List[int]]:
    """解析GPU ID字符串"""
    if gpu_str.lower() == "all":
        return None  # None表示使用所有可用GPU

    try:
        gpu_ids = [int(x.strip()) for x in gpu_str.split(",")]
        return gpu_ids
    except ValueError:
        raise ValueError(f"Invalid GPU IDs: {gpu_str}. Use comma-separated integers or 'all'")


def update_config(
    config_path: str,
    gpu_ids: Optional[List[int]],
    device: str,
    max_memory: Optional[str],
    output_path: Optional[str] = None,
) -> str:
    """更新配置文件以启用多GPU"""

    # 读取原始配置
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 更新硬件配置
    if "hardware" not in config:
        config["hardware"] = {}

    config["hardware"]["device"] = device

    # 配置多GPU
    if "multi_gpu" not in config["hardware"]:
        config["hardware"]["multi_gpu"] = {}

    config["hardware"]["multi_gpu"]["enabled"] = True
    config["hardware"]["multi_gpu"]["gpu_ids"] = gpu_ids

    if max_memory:
        config["hardware"]["multi_gpu"]["max_memory_per_gpu"] = max_memory
    else:
        config["hardware"]["multi_gpu"]["max_memory_per_gpu"] = None

    # 保存更新的配置
    if output_path is None:
        output_path = "configs/config_multi_gpu.yaml"

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return str(output_file)


def print_gpu_info(gpu_ids: Optional[List[int]]):
    """打印GPU信息"""
    logger.info("=" * 60)
    logger.info("GPU 配置信息")
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.error("❌ CUDA 不可用！")
        return False

    total_gpus = torch.cuda.device_count()
    logger.info(f"系统总GPU数: {total_gpus}")

    if gpu_ids is None:
        logger.info(f"将使用所有 {total_gpus} 个GPU")
        gpu_ids_to_show = list(range(total_gpus))
    else:
        logger.info(f"将使用 {len(gpu_ids)} 个GPU: {gpu_ids}")
        gpu_ids_to_show = gpu_ids

        # 检查GPU ID是否有效
        for gpu_id in gpu_ids:
            if gpu_id >= total_gpus:
                logger.error(f"❌ GPU {gpu_id} 不存在（系统只有 {total_gpus} 个GPU）")
                return False

    # 显示每个GPU的信息
    logger.info("\nGPU 详情:")
    for gpu_id in gpu_ids_to_show:
        props = torch.cuda.get_device_properties(gpu_id)
        memory_gb = props.total_memory / (1024**3)
        logger.info(f"  GPU {gpu_id}: {props.name}")
        logger.info(f"    总显存: {memory_gb:.1f} GB")

        # 显示当前显存使用情况
        if torch.cuda.is_initialized():
            allocated = torch.cuda.memory_allocated(gpu_id) / (1024**3)
            cached = torch.cuda.memory_reserved(gpu_id) / (1024**3)
            logger.info(f"    已分配: {allocated:.2f} GB")
            logger.info(f"    已缓存: {cached:.2f} GB")

    logger.info("=" * 60)
    return True


def run_step(script_path: str, config_path: str, step_name: str, verbose: bool = False):
    """运行单个步骤"""
    logger.info(f"\n{'='*60}")
    logger.info(f"执行步骤: {step_name}")
    logger.info(f"{'='*60}")

    cmd = [sys.executable, script_path, "--config", config_path]

    # 设置环境变量，确保子进程能找到 src 模块
    env = os.environ.copy()
    project_root = str(Path(__file__).parent.absolute())
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{project_root}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = project_root

    try:
        if verbose:
            result = subprocess.run(cmd, check=True, env=env, cwd=project_root)
        else:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=project_root,
            )
            # 只显示最后几行输出
            if result.stdout:
                lines = result.stdout.strip().split("\n")
                for line in lines[-10:]:
                    logger.info(line)

        logger.info(f"✅ {step_name} 完成")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {step_name} 失败")
        if not verbose and e.stderr:
            logger.error(f"错误信息:\n{e.stderr}")
        return False


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 解析GPU IDs
    try:
        gpu_ids = parse_gpu_ids(args.gpus)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # 显示GPU信息
    if not print_gpu_info(gpu_ids):
        sys.exit(1)

    # 更新配置文件
    logger.info("\n更新配置文件...")
    logger.info(f"  原始配置: {args.config}")

    try:
        new_config_path = update_config(
            config_path=args.config,
            gpu_ids=gpu_ids,
            device=args.device,
            max_memory=args.max_memory,
            output_path=args.output_config,
        )
        logger.info(f"  新配置保存至: {new_config_path}")
    except Exception as e:
        logger.error(f"❌ 配置更新失败: {e}")
        sys.exit(1)

    # 显示配置摘要
    logger.info("\n配置摘要:")
    logger.info(f"  设备模式: {args.device}")
    logger.info(f"  GPU IDs: {gpu_ids if gpu_ids else '所有可用GPU'}")
    logger.info(f"  最大显存: {args.max_memory if args.max_memory else '不限制'}")

    # 解析运行步骤
    if args.steps.lower() == "all":
        steps = ["prepare", "rollout", "kl", "visualize"]
    else:
        steps = [s.strip() for s in args.steps.split(",")]

    logger.info(f"  运行步骤: {', '.join(steps)}")

    if args.dry_run:
        logger.info("\n🔍 干运行模式 - 不执行实际命令")
        logger.info(f"配置文件已生成: {new_config_path}")
        logger.info("\n要实际运行，请移除 --dry-run 参数")
        return

    # 运行各个步骤
    step_scripts = {
        "prepare": "scripts/01_prepare_data.py",
        "rollout": "scripts/02_generate_rollouts.py",
        "kl": "scripts/03_compute_kl.py",
        "visualize": "scripts/04_visualize.py",
    }

    logger.info("\n" + "="*60)
    logger.info("开始执行流程")
    logger.info("="*60)

    for step in steps:
        if step not in step_scripts:
            logger.warning(f"⚠️  未知步骤: {step}，跳过")
            continue

        script_path = step_scripts[step]
        if not Path(script_path).exists():
            logger.warning(f"⚠️  脚本不存在: {script_path}，跳过")
            continue

        success = run_step(script_path, new_config_path, step, args.verbose)
        if not success:
            logger.error(f"\n❌ 步骤 '{step}' 失败，终止执行")
            sys.exit(1)

    logger.info("\n" + "="*60)
    logger.info("✅ 所有步骤完成！")
    logger.info("="*60)


if __name__ == "__main__":
    main()
