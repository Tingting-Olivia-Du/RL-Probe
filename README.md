# RL-Probe: Understanding RLVR Reasoning via Token-Level KL-Divergence

This repository investigates the reasoning behavior of Large Language Models (LLMs) during **Reinforcement Learning (RL)** training. By analyzing the **Tulu 3.1-8B** model suite, we explore how **RLVR (Reinforcement Learning with Verifiable Rewards)** reshapes logit distributions to correct logical errors inherited from the **DPO (Direct Preference Optimization)** stage.

---

## English Version

### Project Overview

The core research question: **At which exact tokens does a model's reasoning shift from incorrect to correct during RL training?**

We calculate **Token-level KL-Divergence** *across* different training checkpoints (Steps 400, 1200, 2440) to visualize the "evolution" of a model's internal confidence and logical pathing.

### Key Features

- **Multi-directional KL Analysis**: Forward KL, Reverse KL, and Jensen-Shannon divergence for comprehensive distribution comparison
- **Baseline Comparisons**: DPO vs SFT analysis to isolate RLVR's unique contribution
- **Spike Detection**: Statistical detection of critical reasoning tokens with configurable thresholds
- **Token Classification**: Categorize spikes by type (math symbols, numbers, logical connectors, etc.)
- **Entropy Analysis**: Track uncertainty reduction across training
- **Comprehensive Visualization**: Heatmaps, line plots, case studies, and aggregate dashboards

### Experimental Design

#### Models
| Model | Role | Description |
|-------|------|-------------|
| `allenai/Llama-3.1-Tulu-3-8B-SFT` | Baseline | Pre-DPO supervised fine-tuned model |
| `allenai/Llama-3.1-Tulu-3-8B-DPO` | Reference | RL training starting point |
| `Tulu 3.1-8B RLVR Step 400` | Early | Early RL checkpoint |
| `Tulu 3.1-8B RLVR Step 1200` | Mid | Mid-training checkpoint |
| `Tulu 3.1-8B RLVR Step 2440` | Final | Converged RLVR model |

#### Dataset
**MATH (Levels 3 & 4)**: Mid-difficulty problems filtered for cases where DPO fails but RLVR succeeds.

#### Methodology

1. **Rollout Generation**: Generate multiple diverse reasoning trajectories from DPO model (5 samples per problem with varying temperatures)

2. **Teacher Forcing Analysis**: Feed DPO rollouts into RLVR checkpoints to compute:
   - Forward KL: $D_{KL}(P_{RLVR} \parallel P_{DPO})$
   - Reverse KL: $D_{KL}(P_{DPO} \parallel P_{RLVR})$
   - Jensen-Shannon: Symmetric divergence measure

3. **Spike Detection**: Identify critical tokens using z-score thresholding (default: 2σ)

4. **Token Classification**: Categorize spike tokens into semantic categories

5. **Aggregate Analysis**: Compute statistics across problems and checkpoints

### Research Questions

1. **Spike Detection**: Can we identify critical tokens where RLVR begins to "reject" DPO-generated errors?

2. **Entropy Collapse**: How does RLVR reduce uncertainty in mathematical reasoning steps?

3. **Training Dynamics**: How do logical corrections stabilize over training steps?

4. **Token Categories**: Which types of tokens (math symbols, logical connectors, etc.) show the most significant distribution shifts?

5. **Baseline Isolation**: What is the unique contribution of RLVR compared to DPO?

---

## 中文版

### 项目简介

本项目探究大语言模型在**强化学习 (RL)** 阶段的推理行为演变。核心问题：**模型在哪些具体的 Token 位置从错误推理转向正确推理？**

通过计算不同训练阶段 Checkpoints（Step 400, 1200, 2440）**之间**的 **Token 级 KL 散度**，精准定位模型逻辑从"错误"转向"正确"的决定性瞬间。

### 核心特性

- **多方向 KL 分析**：前向 KL、反向 KL、Jensen-Shannon 散度
- **基线对比**：DPO vs SFT 分析，隔离 RLVR 的独特贡献
- **突变点检测**：可配置阈值的统计检测方法
- **Token 分类**：按类型（数学符号、数字、逻辑连接词等）分类突变点
- **熵值分析**：追踪训练过程中的不确定性降低
- **全面可视化**：热力图、折线图、案例研究、汇总仪表板

### 实验设计

#### 模型配置
| 模型 | 角色 | 说明 |
|------|------|------|
| `allenai/Llama-3.1-Tulu-3-8B-SFT` | 基线 | DPO 前的监督微调模型 |
| `allenai/Llama-3.1-Tulu-3-8B-DPO` | 参考 | RL 训练起点 |
| `Tulu 3.1-8B RLVR Step 400` | 早期 | 早期 RL 检查点 |
| `Tulu 3.1-8B RLVR Step 1200` | 中期 | 中期训练检查点 |
| `Tulu 3.1-8B RLVR Step 2440` | 最终 | 收敛的 RLVR 模型 |

#### 数据集
**MATH (Level 3 & 4)**：筛选 DPO 做错而 RLVR 做对的中等难度题目。

#### 方法论

1. **轨迹生成**：从 DPO 模型生成多样化的推理轨迹（每题 5 个样本，使用不同温度）

2. **教师强制分析**：将 DPO 轨迹喂入 RLVR 检查点，计算：
   - 前向 KL：$D_{KL}(P_{RLVR} \parallel P_{DPO})$
   - 反向 KL：$D_{KL}(P_{DPO} \parallel P_{RLVR})$
   - Jensen-Shannon：对称散度度量

3. **突变点检测**：使用 z-score 阈值检测关键 Token（默认：2σ）

4. **Token 分类**：将突变点 Token 分类为语义类别

5. **汇总分析**：跨问题和检查点计算统计数据

### 核心观察指标

1. **突变点检测**：识别 RLVR 模型开始"否定" DPO 错误路径的关键 Token

2. **熵值演变**：观察 RLVR 如何在数学推理中降低不确定性

3. **训练动态**：追踪推理逻辑在训练过程中逐渐固化的过程

4. **Token 类别**：哪类 Token（数学符号、逻辑连接词等）显示最显著的分布变化？

5. **基线隔离**：相比 DPO，RLVR 的独特贡献是什么？

---

## Project Structure

```
RL-Probe/
├── configs/
│   └── config.yaml           # Main configuration file
├── data/
│   ├── raw/                   # Raw MATH dataset
│   ├── processed/             # Processed data
│   └── filtered/              # Filtered problems (DPO-wrong, RLVR-right)
├── rollouts/
│   ├── dpo_errors/            # DPO model rollouts
│   └── samples/               # Sample rollouts for testing
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py         # MATH dataset loader
│   │   ├── filter.py          # Problem filtering utilities
│   │   └── preprocessor.py    # Data preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_loader.py    # Multi-checkpoint model loader
│   │   └── rollout_generator.py # Rollout generation
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── kl_divergence.py   # KL divergence computation
│   │   ├── spike_detector.py  # Spike detection algorithms
│   │   ├── entropy.py         # Entropy analysis
│   │   └── token_classifier.py # Token categorization
│   └── visualization/
│       ├── __init__.py
│       ├── heatmap.py         # Heatmap visualizations
│       ├── lineplot.py        # Line plot visualizations
│       ├── case_study.py      # Case study figures
│       └── aggregate.py       # Aggregate statistics plots
├── scripts/
│   ├── 01_prepare_data.py     # Step 1: Data preparation
│   ├── 02_generate_rollouts.py # Step 2: Rollout generation
│   ├── 03_compute_kl.py       # Step 3: KL computation
│   ├── 04_visualize.py        # Step 4: Visualization
│   └── run_full_pipeline.sh   # Run all steps
├── experiments/
│   └── baseline_comparison.py # DPO vs SFT baseline experiment
├── outputs/
│   ├── figures/               # Generated visualizations
│   ├── logs/                  # Execution logs
│   └── results/               # Analysis results
├── checkpoints/               # Local model checkpoint links
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Project configuration
└── README.md                  # This file
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Tingting-Olivia-Du/RL-Probe.git
cd RL-Probe

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Run the Full Pipeline

```bash
# Run all steps
bash scripts/run_full_pipeline.sh

# Or run steps individually:
python scripts/01_prepare_data.py
python scripts/02_generate_rollouts.py
python scripts/03_compute_kl.py
python scripts/04_visualize.py
```

### Run Baseline Comparison

```bash
python experiments/baseline_comparison.py
```

---

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
# Model configuration
models:
  base_model: "allenai/Llama-3.1-Tulu-3-8B-DPO"
  sft_model: "allenai/Llama-3.1-Tulu-3-8B-SFT"
  rlvr_checkpoints:
    - step: 400
      path: "..."

# Analysis parameters
analysis:
  spike_detection:
    method: "zscore"      # zscore, percentile, adaptive
    threshold: 2.0        # z-score threshold
    min_spike_distance: 3

# Hardware settings
hardware:
  device: "cuda"
  dtype: "bfloat16"
  batch_size: 4
```

---

## Expected Outputs

### Visualizations

1. **`mean_kl_progression.png`**: Mean KL divergence across checkpoints
2. **`spike_density_comparison.png`**: Spike density by region and checkpoint
3. **`category_kl_breakdown.png`**: KL divergence by token category
4. **`entropy_reduction.png`**: Entropy reduction across training
5. **`summary_dashboard.png`**: Comprehensive analysis summary
6. **`case_study_*.png`**: Detailed case studies for selected problems

### Quantitative Metrics

- Mean KL divergence per checkpoint
- Spike counts and density by region (early/mid/late)
- Token category statistics
- Entropy progression
- Statistical significance tests

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{rl_probe_2026,
  title={RL-Probe: Token-Level KL-Divergence Analysis for RLVR Reasoning},
  author={Tingting Du},
  year={2026},
  url={https://github.com/Tingting-Olivia-Du/RL-Probe}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Allen Institute for AI](https://allenai.org/) for the Tulu model suite
- [HuggingFace](https://huggingface.co/) for the Transformers library
- [Lighteval](https://github.com/huggingface/lighteval) for the MATH dataset
