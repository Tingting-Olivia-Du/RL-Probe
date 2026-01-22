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
RESULTS_DIR="outputs/results"
FIGURES_DIR="outputs/figures"

# Step 1: Prepare Data
echo "[Step 1/4] Preparing data..."
echo "-------------------------------------------"
python scripts/01_prepare_data.py \
    --config "$CONFIG" \
    --output-dir "$DATA_DIR"
echo ""

# Step 2: Compute KL Divergence (using DPO responses as rollouts)
echo "[Step 2/3] Computing KL divergence..."
echo "-------------------------------------------"
python scripts/03_compute_kl.py \
    --config "$CONFIG" \
    --use-dpo-responses \
    --dpo-responses "$DATA_DIR/dpo_responses.json" \
    --output-dir "$RESULTS_DIR"
echo ""

# Step 3: Visualize Results
echo "[Step 3/3] Generating visualizations..."
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
echo "  - Results: $RESULTS_DIR"
echo "  - Figures: $FIGURES_DIR"
