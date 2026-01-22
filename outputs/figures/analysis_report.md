# RL-Probe: KL Divergence Analysis Report

**Generated:** 2026-01-22 06:46:52

---

## Executive Summary

This report analyzes **230 problems** where DPO model failed but RLVR model succeeded.

### Key Findings

- **Mean KL Divergence:** 0.0172 ± 0.0072
- **Total Spikes Detected:** 2313
- **Mean Spikes per Problem:** 10.06
- **Mean Entropy:** 0.2598

## Detailed Statistics

### rlvr_step_500

- **Number of Problems:** 230
- **KL Divergence:**
  - Mean: 0.0172
  - Median: 0.0164
  - Std: 0.0072
  - Min: 0.0053
  - Max: 0.0395
- **Spike Analysis:**
  - Mean spikes per problem: 10.06
  - Total spikes: 2313
  - Mean spike value: 0.4255
- **Entropy:** 0.2598

## Token Category Analysis

| Category | Mean KL | Total Count |
|----------|---------|-------------|
| math_symbol | 0.0338 | 33728 |
| other | 0.0326 | 86214 |
| punctuation | 0.0276 | 6146 |
| logical | 0.0168 | 1265 |
| step_marker | 0.0110 | 1158 |
| number | 0.0108 | 20259 |
| variable | 0.0069 | 9302 |

## Spike Distribution Analysis

| Spike Range | Count | Percentage |
|-------------|-------|------------|
| 0 spikes | 0 | 0.0% |
| 1-5 spikes | 82 | 35.7% |
| 6-10 spikes | 78 | 33.9% |
| >10 spikes | 70 | 30.4% |

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
   - Mean of 10.1 spikes per problem indicates frequent localized corrections
   - 70 problems (30.4%) have more than 10 spikes, suggesting complex error correction

3. **Category Analysis:**
   - **Highest KL:** math_symbol (Mean KL: 0.0338)
     - RLVR model shows significant differences in handling math_symbol tokens
   - **Lowest KL:** variable (Mean KL: 0.0069)
     - RLVR model maintains similar behavior for variable tokens

4. **Entropy Analysis:**
   - Mean entropy: 0.2598
   - Lower entropy suggests RLVR model is more confident/consistent in its predictions

### Implications for RLVR Training

- **Localized Corrections:** The spike pattern suggests RLVR learns to make targeted corrections rather than wholesale distribution changes
- **Category-Specific Learning:** Different KL values across categories indicate RLVR has learned category-specific improvements
- **Error Correction Mechanism:** High spike values in specific problems suggest RLVR has developed mechanisms to identify and correct DPO's errors
