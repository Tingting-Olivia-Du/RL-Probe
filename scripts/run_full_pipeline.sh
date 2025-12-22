#!/bin/bash
# RL-Probe: Full Analysis Pipeline
# Run all steps sequentially

set -e  # Exit on error

echo "=========================================="
echo "RL-Probe: Token-Level KL Divergence Analysis"
echo "=========================================="
echo ""

# Configuration
CONFIG="configs/config.yaml"
DATA_DIR="data/filtered"
ROLLOUT_DIR="rollouts/dpo_errors"
RESULTS_DIR="outputs/results"
FIGURES_DIR="outputs/figures"

# Step 1: Prepare Data
echo "[Step 1/4] Preparing data..."
echo "-------------------------------------------"
python scripts/01_prepare_data.py \
    --config "$CONFIG" \
    --output-dir "$DATA_DIR"
echo ""

# Step 2: Generate Rollouts
echo "[Step 2/4] Generating DPO rollouts..."
echo "-------------------------------------------"
python scripts/02_generate_rollouts.py \
    --config "$CONFIG" \
    --problems "$DATA_DIR/filtered_problems.json" \
    --output-dir "$ROLLOUT_DIR"
echo ""

# Step 3: Compute KL Divergence
echo "[Step 3/4] Computing KL divergence..."
echo "-------------------------------------------"
python scripts/03_compute_kl.py \
    --config "$CONFIG" \
    --rollouts "$ROLLOUT_DIR/all_rollouts.json" \
    --output-dir "$RESULTS_DIR"
echo ""

# Step 4: Visualize Results
echo "[Step 4/4] Generating visualizations..."
echo "-------------------------------------------"
python scripts/04_visualize.py \
    --config "$CONFIG" \
    --results-dir "$RESULTS_DIR" \
    --output-dir "$FIGURES_DIR" \
    --num-case-studies 3
echo ""

echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Data:    $DATA_DIR"
echo "  - Rollouts: $ROLLOUT_DIR"
echo "  - Results: $RESULTS_DIR"
echo "  - Figures: $FIGURES_DIR"
