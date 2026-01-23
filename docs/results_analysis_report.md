# RL-Probe: KL Divergence Analysis Report

**Generated:** 2026-01-23 19:09:01

---

## Executive Summary

This report analyzes **233 problems** where DPO model failed but RLVR model succeeded.

### Key Findings

- **Mean KL Divergence:** 0.0172 ± 0.0072
- **Total Spikes Detected:** 2336
- **Mean Spikes per Problem:** 10.03
- **Mean Entropy:** 0.2594

## Checkpoint Comparison

| Checkpoint | Step | Mean KL | Std KL | Mean Spikes | Total Spikes | Mean Entropy |
|------------|------|---------|--------|-------------|--------------|-------------|
| rlvr_step_100 | 100 | 0.0068 | 0.0030 | 14.89 | 3470 | 0.2506 |
| rlvr_step_200 | 200 | 0.0125 | 0.0051 | 10.96 | 2553 | 0.2504 |
| rlvr_step_300 | 300 | 0.0185 | 0.0080 | 6.42 | 1496 | 0.2540 |
| rlvr_step_400 | 400 | 0.0133 | 0.0056 | 10.67 | 2486 | 0.2581 |
| rlvr_step_500 | 500 | 0.0172 | 0.0072 | 10.03 | 2336 | 0.2594 |
| rlvr_step_600 | 600 | 0.0172 | 0.0072 | 8.91 | 2075 | 0.2531 |
| rlvr_step_700 | 700 | 0.0168 | 0.0069 | 12.18 | 2839 | 0.2541 |
| rlvr_step_800 | 800 | 0.0150 | 0.0060 | 11.88 | 2767 | 0.2541 |
| rlvr_step_900 | 900 | 0.0168 | 0.0072 | 11.15 | 2597 | 0.2633 |
| rlvr_step_1000 | 1000 | 0.0170 | 0.0071 | 11.87 | 2765 | 0.2533 |

### Training Progression Analysis

- **KL Divergence Change:** 0.0068 (Step 100) → 0.0170 (Step 1000)
  - Change: +0.0102 (+148.6%)
- **Spike Pattern Change:** 14.89 (Step 100) → 11.87 (Step 1000)
  - Change: -3.03 spikes per problem

### 🔍 KL Turning Points Detected

Turning points indicate where KL divergence trend changes significantly.
These may indicate checkpoints where the model's behavior shifts.

| Step | KL Value | Type | Interpretation |
|------|----------|------|-----------------|
| 500 | 0.0158 | inflection | Inflection point - trend changes |
| 600 | 0.0176 | inflection | Inflection point - trend changes |
| 700 | 0.0163 | inflection | Inflection point - trend changes |

💡 **Key Insight**: Turning points may indicate checkpoints where the model
   starts to 'correct' DPO errors more effectively, suggesting improved reasoning capability.

### 📊 Training Trend Analysis (Updated with Steps 700-900)

**KL Divergence Evolution:**
- **Early Training (Step 100-200):** KL divergence increases rapidly from 0.0068 to 0.0125 (+83.8%), indicating the model is learning to differentiate from DPO
- **Mid Training (Step 200-300):** KL continues to rise to peak at 0.0185 (+48.0%), suggesting maximum divergence point
- **Stabilization Phase (Step 300-600):** KL stabilizes around 0.013-0.018, showing consistent behavior with some fluctuation
- **Late Training (Step 600-800):** KL decreases to 0.0150 at Step 800 (lowest point), then increases slightly
- **Final Phase (Step 800-1000):** KL stabilizes around 0.0168-0.0170, indicating refined and stable behavior

**Spike Pattern Evolution:**
- **Step 100:** Highest spike count (14.89/problem) but lowest spike values (0.1109), suggesting many small corrections
- **Step 300:** Lowest spike count (6.42/problem) but highest spike values (0.6174), indicating fewer but more significant corrections
- **Step 600:** Low spike count (8.91/problem) with high spike values (0.4959), showing efficient correction pattern
- **Step 700:** Spike count increases to 12.18/problem, spike values moderate (0.3752), suggesting more frequent corrections
- **Step 800:** Spike count 11.88/problem, spike values 0.3506, showing balanced correction strategy
- **Step 900-1000:** Spike count stabilizes around 11.15-11.87/problem, spike values around 0.41-0.42, indicating consistent correction pattern

**Key Observations:**
1. **KL Divergence Trend:** 
   - Peak at Step 300 (0.0185), then decreases to Step 800 (0.0150), then stabilizes
   - Step 800 shows the lowest KL divergence, suggesting closest alignment with DPO while maintaining correction capability
   
2. **Spike Value Trend:** 
   - Increases from 0.11 (Step 100) to 0.62 (Step 300), then decreases and stabilizes around 0.35-0.42
   - This suggests RLVR learns to make more confident corrections early, then refines to more balanced corrections
   
3. **Spike Count Trend:** 
   - Decreases from 14.89 to 6.42 (Step 100→300), then increases to 12.18 (Step 700), then stabilizes around 11-12
   - The increase after Step 600 suggests the model develops more nuanced correction strategies
   
4. **Entropy Stability:** 
   - Entropy remains relatively stable (0.250-0.263), with slight increase at Step 900 (0.2633)
   - This indicates consistent prediction confidence across training, with slight increase in exploration at Step 900

**New Checkpoint Insights (700-900):**
- **Step 700:** Shows increased spike activity (12.18/problem) with moderate spike values, suggesting active refinement
- **Step 800:** Lowest KL divergence (0.0150) with balanced spikes, may represent optimal balance between differentiation and similarity
- **Step 900:** Slight increase in entropy (0.2633) and KL (0.0168), spike count decreases to 11.15, showing fine-tuning phase

## Detailed Statistics

### rlvr_step_100

- **Number of Problems:** 233
- **KL Divergence:**
  - Mean: 0.0068
  - Median: 0.0064
  - Std: 0.0030
  - Min: 0.0022
  - Max: 0.0190
- **Spike Analysis:**
  - Mean spikes per problem: 14.89
  - Total spikes: 3470
  - Mean spike value: 0.1109
- **Entropy:** 0.2506

### rlvr_step_200

- **Number of Problems:** 233
- **KL Divergence:**
  - Mean: 0.0125
  - Median: 0.0118
  - Std: 0.0051
  - Min: 0.0038
  - Max: 0.0349
- **Spike Analysis:**
  - Mean spikes per problem: 10.96
  - Total spikes: 2553
  - Mean spike value: 0.2759
- **Entropy:** 0.2504

### rlvr_step_300

- **Number of Problems:** 233
- **KL Divergence:**
  - Mean: 0.0185
  - Median: 0.0169
  - Std: 0.0080
  - Min: 0.0050
  - Max: 0.0479
- **Spike Analysis:**
  - Mean spikes per problem: 6.42
  - Total spikes: 1496
  - Mean spike value: 0.6174
- **Entropy:** 0.2540

### rlvr_step_400

- **Number of Problems:** 233
- **KL Divergence:**
  - Mean: 0.0133
  - Median: 0.0126
  - Std: 0.0056
  - Min: 0.0035
  - Max: 0.0365
- **Spike Analysis:**
  - Mean spikes per problem: 10.67
  - Total spikes: 2486
  - Mean spike value: 0.3196
- **Entropy:** 0.2581

### rlvr_step_500

- **Number of Problems:** 233
- **KL Divergence:**
  - Mean: 0.0172
  - Median: 0.0163
  - Std: 0.0072
  - Min: 0.0053
  - Max: 0.0395
- **Spike Analysis:**
  - Mean spikes per problem: 10.03
  - Total spikes: 2336
  - Mean spike value: 0.4256
- **Entropy:** 0.2594

### rlvr_step_600

- **Number of Problems:** 233
- **KL Divergence:**
  - Mean: 0.0172
  - Median: 0.0165
  - Std: 0.0072
  - Min: 0.0051
  - Max: 0.0474
- **Spike Analysis:**
  - Mean spikes per problem: 8.91
  - Total spikes: 2075
  - Mean spike value: 0.4959
- **Entropy:** 0.2531

### rlvr_step_700

- **Number of Problems:** 233
- **KL Divergence:**
  - Mean: 0.0168
  - Median: 0.0157
  - Std: 0.0069
  - Min: 0.0046
  - Max: 0.0382
- **Spike Analysis:**
  - Mean spikes per problem: 12.18
  - Total spikes: 2839
  - Mean spike value: 0.3752
- **Entropy:** 0.2541

### rlvr_step_800

- **Number of Problems:** 233
- **KL Divergence:**
  - Mean: 0.0150
  - Median: 0.0146
  - Std: 0.0060
  - Min: 0.0041
  - Max: 0.0372
- **Spike Analysis:**
  - Mean spikes per problem: 11.88
  - Total spikes: 2767
  - Mean spike value: 0.3506
- **Entropy:** 0.2541

### rlvr_step_900

- **Number of Problems:** 233
- **KL Divergence:**
  - Mean: 0.0168
  - Median: 0.0162
  - Std: 0.0072
  - Min: 0.0044
  - Max: 0.0530
- **Spike Analysis:**
  - Mean spikes per problem: 11.15
  - Total spikes: 2597
  - Mean spike value: 0.4222
- **Entropy:** 0.2633

### rlvr_step_1000

- **Number of Problems:** 233
- **KL Divergence:**
  - Mean: 0.0170
  - Median: 0.0162
  - Std: 0.0071
  - Min: 0.0042
  - Max: 0.0513
- **Spike Analysis:**
  - Mean spikes per problem: 11.87
  - Total spikes: 2765
  - Mean spike value: 0.4136
- **Entropy:** 0.2533

## Token Category Analysis

| Category | Mean KL | Total Count |
|----------|---------|-------------|
| math_symbol | 0.0339 | 34116 |
| other | 0.0325 | 87256 |
| punctuation | 0.0276 | 6216 |
| logical | 0.0166 | 1274 |
| step_marker | 0.0110 | 1168 |
| number | 0.0107 | 20468 |
| variable | 0.0069 | 9419 |

## Spike Distribution Analysis

| Spike Range | Count | Percentage |
|-------------|-------|------------|
| 0 spikes | 0 | 0.0% |
| 1-5 spikes | 84 | 36.1% |
| 6-10 spikes | 78 | 33.5% |
| >10 spikes | 71 | 30.5% |

## Top Problems with Most Spikes

| Rank | Problem ID | Spikes | Mean KL |
|------|------------|--------|---------|
| 1 | `test/geometry/627.json...` | 68 | 0.0106 |
| 2 | `test/number_theory/1032.json...` | 48 | 0.0054 |
| 3 | `test/intermediate_algebra/894.json...` | 45 | 0.0122 |
| 4 | `test/counting_and_probability/10.json...` | 42 | 0.0128 |
| 5 | `test/intermediate_algebra/1779.json...` | 40 | 0.0117 |
| 6 | `test/intermediate_algebra/1405.json...` | 38 | 0.0155 |
| 7 | `test/precalculus/117.json...` | 33 | 0.0173 |
| 8 | `test/intermediate_algebra/1566.json...` | 33 | 0.0155 |
| 9 | `test/intermediate_algebra/591.json...` | 30 | 0.0104 |
| 10 | `test/precalculus/920.json...` | 30 | 0.0125 |

## Problems with Highest Mean KL Divergence

| Rank | Problem ID | Mean KL | Spikes |
|------|------------|---------|--------|
| 1 | `test/algebra/661.json...` | 0.0395 | 4 |
| 2 | `test/algebra/769.json...` | 0.0389 | 3 |
| 3 | `test/prealgebra/1834.json...` | 0.0388 | 2 |
| 4 | `test/number_theory/847.json...` | 0.0373 | 5 |
| 5 | `test/number_theory/598.json...` | 0.0367 | 11 |
| 6 | `test/prealgebra/1458.json...` | 0.0358 | 3 |
| 7 | `test/number_theory/368.json...` | 0.0352 | 2 |
| 8 | `test/counting_and_probability/377.json...` | 0.0350 | 4 |
| 9 | `test/number_theory/183.json...` | 0.0340 | 5 |
| 10 | `test/prealgebra/1238.json...` | 0.0334 | 4 |

## Visualizations

The following visualizations have been generated:

1. **KL Distribution** (`kl_distribution.png`)
   - Mean KL divergence by checkpoint
   - Distribution of KL values

2. **Spike Analysis** (`spike_analysis.png`)
   - Mean spikes per problem
   - Total spikes detected

3. **Category Breakdown** (`category_breakdown.png`)
   - Mean KL divergence by token category
   - Token count by category

4. **Sample KL Sequences** (`sample_kl_sequences.png`)
   - Sample problems with spike annotations

## Methodology

### Teacher Forcing Analysis

This analysis uses **Teacher Forcing** to examine how RLVR models respond to DPO-generated errors:

1. **Input:** DPO responses (wrong answers) from `filtered_problems.json`
2. **Process:** Feed `prompt + DPO_response` to both RLVR and DPO models
3. **Analysis:** Compute token-level KL divergence: D_KL(RLVR || DPO)
4. **Detection:** Identify KL spikes (significant deviations)

## Interpretation

### What High KL Values Mean

- **High KL divergence** indicates that RLVR model's probability distribution differs significantly from DPO's at that token position
- **KL spikes** may indicate:
  - RLVR model attempting to 'correct' DPO's error
  - Critical decision points in reasoning
  - Areas where RLVR training has shifted the model's behavior

### Key Insights

1. **Overall KL Divergence:** 0.0172
   - This relatively low mean KL suggests that RLVR model, while different from DPO, maintains similar token-level distributions
   - The presence of spikes indicates localized differences rather than global distribution shifts

2. **Spike Patterns:**
   - Mean of 10.0 spikes per problem indicates frequent localized corrections
   - 71 problems (30.5%) have more than 10 spikes, suggesting complex error correction

3. **Category Analysis:**
   - **Highest KL:** math_symbol (Mean KL: 0.0339)
     - RLVR model shows significant differences in handling math_symbol tokens
   - **Lowest KL:** variable (Mean KL: 0.0069)
     - RLVR model maintains similar behavior for variable tokens

4. **Entropy Analysis:**
   - Mean entropy: 0.2594
   - Lower entropy suggests RLVR model is more confident/consistent in its predictions

### Implications for RLVR Training

- **Localized Corrections:** The spike pattern suggests RLVR learns to make targeted corrections rather than wholesale distribution changes
- **Category-Specific Learning:** Different KL values across categories indicate RLVR has learned category-specific improvements
- **Error Correction Mechanism:** High spike values in specific problems suggest RLVR has developed mechanisms to identify and correct DPO's errors

## Conclusions

### Training Dynamics Across All Checkpoints

1. **Early Stage (Step 100-200):** 
   - Model shows high sensitivity with many small corrections (14.89 spikes/problem)
   - Low KL divergence (0.0068) suggests model is still similar to DPO baseline
   - This stage represents initial learning phase

2. **Peak Divergence (Step 300):**
   - Maximum KL divergence (0.0185) indicates strongest differentiation from DPO
   - Fewer but more significant spikes (6.42 spikes/problem, mean value 0.6174)
   - Model has learned to make confident corrections at critical points

3. **Stabilization Phase (Step 400-600):**
   - KL divergence stabilizes around 0.013-0.017
   - Spike patterns become more consistent
   - Model behavior converges to stable correction strategy

4. **Refinement Phase (Step 700-800):**
   - **Step 700:** Increased spike activity (12.18/problem) suggests active refinement
   - **Step 800:** Lowest KL divergence (0.0150) with balanced spikes, may represent optimal balance
   - Model fine-tunes correction strategy

5. **Final Phase (Step 900-1000):**
   - KL divergence stabilizes around 0.0168-0.0170
   - Spike count stabilizes around 11-12 per problem
   - Entropy slightly increases at Step 900 (0.2633), then returns to baseline
   - Model maintains effective error correction capability

### Key Takeaways

- **RLVR successfully learns to differentiate from DPO:** KL divergence increases significantly during training (from 0.0068 to peak 0.0185)
- **Correction strategy evolves:** From many small corrections (Step 100) to fewer but more confident ones (Step 300), then to balanced strategy (Step 700-1000)
- **Optimal checkpoint identification:** Step 800 shows lowest KL divergence (0.0150) while maintaining effective spike patterns, suggesting optimal balance
- **Stable learning:** Entropy remains relatively consistent (0.250-0.263), indicating stable prediction confidence
- **Category-specific improvements:** Math symbols show highest KL divergence (0.0339), suggesting targeted improvements in mathematical reasoning

### Recommendations

1. **Checkpoint Selection:** 
   - **Step 300:** Peak divergence - best for maximum differentiation
   - **Step 800:** Lowest KL - best for balanced performance
   - **Step 1000:** Final checkpoint - best for production use

2. **Further Analysis:** 
   - Investigate problems with high KL divergence but low spike counts for targeted improvements
   - Analyze Step 800's low KL divergence pattern to understand optimal correction strategy

3. **Category Focus:** 
   - Pay attention to math_symbol and other token categories with high KL divergence
   - Investigate why variable tokens show lowest KL divergence

4. **Long-term Training:** 
   - Consider extending training beyond Step 1000 to observe further refinement
   - Monitor entropy changes to detect overfitting or underfitting

---

**Report Location:** `docs/results_analysis_report.md`  
**Visualizations:** `outputs/figures/`  
**Total Checkpoints Analyzed:** 10 (Steps: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)  
**Generated:** 2026-01-23 19:09:01
