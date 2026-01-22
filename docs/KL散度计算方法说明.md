# KL 散度计算方法说明

## 方法概述

你的方法使用 **Teacher Forcing** 技术来分析模型在面对错误推理时的反应：

```
1. DPO 模型生成错误的 rollout（推理轨迹）
   ↓
2. 将这个错误的 rollout 作为输入，输入到不同的 RLVR checkpoints
   ↓
3. 使用 Teacher Forcing：强制模型按照 DPO 的错误路径进行推理
   ↓
4. 获取每个 checkpoint 在每个 token 位置的 logits（概率分布）
   ↓
5. 计算 KL 散度：D_KL(RLVR_checkpoint || DPO) 或 D_KL(DPO || RLVR_checkpoint)
```

### ✅ **使用 DPO Responses 作为 Rollouts**

**重要发现**：`dpo_responses.json` 中的 `response` 可以直接作为 rollouts 使用！

**优势**：
- ✅ **无需重新生成**：直接使用 DPO 实际生成的错误推理
- ✅ **真实错误**：这些是 DPO 模型在实际评测中产生的错误
- ✅ **已保存完整**：包含完整的推理过程，不仅仅是最终答案
- ✅ **效率更高**：避免重复生成，节省计算资源

**处理逻辑**：
1. **针对 `filtered_problems.json`**：只处理 DPO 错误但 RLVR 正确的问题
2. **匹配 DPO responses**：从 `dpo_responses.json` 中找到对应问题的 responses
3. **Teacher Forcing 流程**：
   - 对于每个 filtered problem，使用其对应的 DPO response 作为 rollout
   - 构建输入：`prompt + DPO_response`
   - 使用 Teacher Forcing：强制 RLVR checkpoint 按照 DPO 的错误路径进行推理
   - 获取每个 token 位置的 logits（概率分布）
   - 计算 KL 散度：`D_KL(RLVR_checkpoint || DPO)`

**使用方法**：
```bash
# 分析单个 checkpoint
python scripts/03_compute_kl.py --use-dpo-responses --gpu 0 --checkpoint rlvr_step_500

# 分析所有 checkpoints（配置文件中的所有 checkpoints）
python scripts/03_compute_kl.py --use-dpo-responses --gpu 0

# 配置文件中包含的 checkpoints:
# - rlvr_step_100, rlvr_step_200, rlvr_step_300 (early stage)
# - rlvr_step_400, rlvr_step_500, rlvr_step_600, rlvr_step_700 (mid stage)
# - rlvr_step_800, rlvr_step_900 (late stage)
# - rlvr_step_1000 (final)
```

## 具体实现

### 1. Rollout 生成阶段

```python
# 从 RLVR checkpoints 生成 rollouts（但实际应该从 DPO 生成错误的）
rollouts = generator.generate_diverse_rollouts(
    prompt=prompt,
    num_samples=5,  # 每个问题生成 5 个不同的 rollouts
)
```

**注意**：根据你的逻辑，rollouts 应该从 **DPO 模型**生成，而不是从 RLVR checkpoints。这样可以确保：
- Rollouts 是 DPO 产生的错误推理
- 这些错误是 RLVR 模型需要"纠正"或"反应"的

### 2. Teacher Forcing 阶段

```python
# 将 DPO 生成的错误 rollout 作为输入
full_text = prompt + rollout  # prompt + DPO的错误推理

# 使用 Teacher Forcing：强制模型按照这个序列进行推理
input_ids = tokenizer(full_text, return_tensors="pt")
logits = model(input_ids)  # 获取每个位置的 logits
```

**Teacher Forcing** 的含义：
- 不是让模型自由生成
- 而是强制模型按照给定的序列（DPO 的错误路径）进行前向传播
- 在每个 token 位置，模型会输出一个概率分布（logits）

### 3. KL 散度计算

```python
# 对于每个 token 位置 t：
logits_p = RLVR_checkpoint(input_ids)  # RLVR 模型在该位置的分布
logits_q = DPO_model(input_ids)        # DPO 模型在该位置的分布

# 转换为概率分布
probs_p = softmax(logits_p)
probs_q = softmax(logits_q)

# 计算 KL 散度
KL[t] = sum(probs_p[t] * log(probs_p[t] / probs_q[t]))
```

## 方法合理性分析

### ✅ **优点和合理性**

1. **错误传播分析**
   - 可以观察 RLVR 模型在面对 DPO 错误时的反应
   - 识别哪些 token 位置 RLVR 与 DPO 差异最大（KL spike）
   - 理解 RLVR 如何"纠正"或"偏离"DPO 的错误路径

2. **训练动态追踪**
   - 通过比较不同 checkpoints 的 KL 值，可以看到：
     - 早期 checkpoint：可能更接近 DPO 的分布
     - 后期 checkpoint：可能更偏离 DPO 的错误路径
   - 这反映了 RLVR 训练过程中对错误推理的"纠正"能力

3. **Token-level 细粒度分析**
   - 可以识别关键的错误点（spike detection）
   - 分析哪些类型的 token（数学符号、数字、逻辑词）差异最大

4. **Teacher Forcing 的合理性**
   - 这是标准的序列模型分析方法
   - 类似于语言模型的 perplexity 计算
   - 可以公平地比较不同模型在相同输入下的分布差异

### ⚠️ **潜在问题和注意事项**

1. **Rollout 来源问题**
   - **当前实现**：从 RLVR checkpoints 生成 rollouts
   - **应该改为**：从 DPO 模型生成**错误的** rollouts
   - **原因**：这样才能分析 RLVR 如何反应 DPO 的错误

2. **KL 散度的解释**
   - **高 KL 值**可能意味着：
     - RLVR 在该位置"纠正"了 DPO 的错误（好的）
     - 或者 RLVR 在该位置产生了新的错误（坏的）
   - **需要结合正确性判断**：如果 RLVR 最终答案正确，高 KL 可能是好的

3. **Teacher Forcing vs 自由生成**
   - Teacher Forcing：强制模型按照给定路径推理
   - 自由生成：模型自己选择路径
   - **差异**：Teacher Forcing 下的 KL 反映的是"如果按照这个路径，模型会怎么想"
   - **可能不反映**：模型实际生成时的行为

4. **因果性 vs 相关性**
   - KL 散度显示的是**相关性**（分布差异）
   - 不一定能证明**因果性**（RLVR 因为看到错误而纠正）
   - 需要结合其他分析（如注意力机制、梯度分析）

## 改进建议

### 1. ✅ **使用 DPO Responses（推荐）**

**最佳方案**：直接使用 `dpo_responses.json` 中的 responses 作为 rollouts

```bash
# 使用 DPO responses 作为 rollouts（推荐）
python scripts/03_compute_kl.py --use-dpo-responses --gpu 0 --checkpoint rlvr_step_500
```

**为什么这样更好**：
- ✅ 使用 DPO 实际生成的错误推理（真实错误）
- ✅ 无需重新生成，节省时间和计算资源
- ✅ 这些 responses 已经保存在 `dpo_responses.json` 中
- ✅ 符合你的逻辑：使用 DPO 的错误作为起点

### 2. 生成多样化的 Rollouts（可选）

如果需要多个 rollouts 进行分析，可以从 DPO 生成：

```python
# 从 DPO 模型生成多个 rollouts
dpo_model = loader.load_model("dpo")
generator = RolloutGenerator(dpo_model, tokenizer)

# 为每个问题生成多个错误的 rollouts
for problem in filtered_problems:  # DPO错误但RLVR正确的问题
    rollouts = generator.generate_diverse_rollouts(prompt, num_samples=5)
```

### 2. 添加对比分析

除了计算 KL(RLVR || DPO)，还可以：

```python
# 1. 对比不同 checkpoints 之间的 KL
KL(RLVR_step_1000 || RLVR_step_100)  # 看训练过程中的变化

# 2. 对比正确 vs 错误的 rollouts
KL(RLVR_on_correct_rollout || RLVR_on_wrong_rollout)

# 3. 对比不同位置的 KL
KL_at_math_symbols vs KL_at_numbers vs KL_at_logical_words
```

### 3. 结合正确性分析

```python
# 对于每个 rollout，检查：
# 1. DPO 的答案是否正确（应该错误）
# 2. RLVR 在该 rollout 下的答案是否正确
# 3. KL spike 的位置是否与错误位置相关
```

## 方法的应用场景

这个方法特别适合分析：

1. **错误纠正机制**
   - RLVR 如何识别和纠正 DPO 的错误
   - 哪些 token 位置是关键的错误点

2. **训练动态**
   - 不同训练阶段对错误的敏感度
   - RLVR 如何逐步学会纠正错误

3. **模型对齐**
   - RLVR 与 DPO 的分布差异
   - 哪些方面 RLVR 偏离了 DPO

## 总结

### 方法合理性：✅ **合理且有价值**

这个方法在以下方面是合理的：
- ✅ Teacher Forcing 是标准的分析方法
- ✅ KL 散度可以量化分布差异
- ✅ Token-level 分析提供了细粒度洞察
- ✅ 可以追踪训练动态

### 需要注意的问题：

1. ⚠️ **Rollout 应该从 DPO 生成**（当前实现可能不对）
2. ⚠️ **KL 值的解释需要结合正确性**
3. ⚠️ **Teacher Forcing 下的 KL 可能不完全反映自由生成的行为**

### 建议的改进：

1. 确保 rollouts 从 DPO 模型生成（错误的推理）
2. 添加正确性分析，区分"好的 KL"和"坏的 KL"
3. 对比不同 checkpoints 之间的 KL，追踪训练动态
4. 结合注意力机制分析，理解模型如何"关注"错误位置
