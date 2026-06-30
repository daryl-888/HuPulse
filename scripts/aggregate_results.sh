#!/bin/bash
# aggregate_results.sh
# Run on Carya after jobs complete to collect and compare results.
#
# Usage: bash scripts/aggregate_results.sh [output_dir]
#   output_dir defaults to ./results_summary/
#
# This script:
#   1. Finds all _per_fold_best.csv files in the PulseDB_multi_full directory
#   2. Computes mean ± std across 10 folds for each experiment
#   3. Checks AAMI/ISO 81060-2 pass
#   4. Writes a comparison table

set -e

OUT_DIR="${1:-./results_summary}"
mkdir -p "$OUT_DIR"

BASE=/project/rhu/PulseBP/Pulse/PulseDB_multi_full

echo "==========================================================="
echo "  MS4+ Results Aggregator"
echo "  Base: $BASE"
echo "  Out:  $OUT_DIR"
echo "  Date: $(date)"
echo "==========================================================="

# Find all per_fold_best.csv files
FOLD_FILES=$(find "$BASE" -maxdepth 2 -name '*_per_fold_best.csv' 2>/dev/null || true)

if [ -z "$FOLD_FILES" ]; then
    echo "WARNING: No _per_fold_best.csv files found. Jobs may not have finished."
    echo "Searching in: $BASE"
fi

SUMMARY="$OUT_DIR/comparison_table.txt"
echo "Writing comparison to $SUMMARY"

{
    echo "==========================================================="
    echo " MS4+ Experiment Results — $(date)"
    echo "==========================================================="
    echo ""
    echo "Baseline from paper:"
    echo "  MS4 (S4 + late concat): SBP MAE 7.10 / DBP MAE 4.64"
    echo "  MResNet50:              SBP MAE 5.56 / DBP MAE 3.73"
    echo ""
    echo "==========================================================="
    echo " PER-FOLD BREAKDOWN"
    echo "==========================================================="
} >> "$SUMMARY"

for f in $FOLD_FILES; do
    DIR=$(dirname "$f")
    NAME=$(basename "$DIR")
    BP=$(basename "$f" | cut -d'_' -f1)

    echo "" >> "$SUMMARY"
    echo "--- $NAME ($BP) ---" >> "$SUMMARY"

    # Check if file has data (more than 1 line)
    LINES=$(wc -l < "$f")
    if [ "$LINES" -le 1 ]; then
        echo "  (no data yet)" >> "$SUMMARY"
        continue
    fi

    # Print the CSV content
    cat "$f" >> "$SUMMARY"
done

# Compute aggregate metrics
{
    echo ""
    echo "==========================================================="
    echo " AGGREGATE COMPARISON"
    echo "==========================================================="
    echo ""
    printf "%-30s %-10s %-12s %-12s %-12s %-12s %-8s\n" \
           "Experiment" "BP" "cb_MAE" "cf_MAE" "cb_ME" "cf_ME" "AAMI"
    printf "%-30s %-10s %-12s %-12s %-12s %-12s %-8s\n" \
           "---------" "--" "-----" "-----" "-----" "-----" "----"
} >> "$SUMMARY"

for f in $FOLD_FILES; do
    DIR=$(dirname "$f")
    NAME=$(basename "$DIR")
    BP=$(basename "$f" | cut -d'_' -f1)

    LINES=$(wc -l < "$f")
    if [ "$LINES" -le 1 ]; then
        continue
    fi

    # Extract metrics from CSV (skip header)
    # cols: fold,best_cb_MAE,best_cb_epoch,best_cb_MEAN_ERR,best_cb_STD_ERR,best_cb_R2,best_cf_MAE,best_cf_epoch,best_cf_MEAN_ERR,best_cf_STD_ERR,best_cf_R2
    tail -n +2 "$f" | awk -F',' '
    {
        cb_mae_sum += $2; cf_mae_sum += $7;
        cb_me_sum  += $4; cf_me_sum  += $9;
        cb_se_sum  += $5; cf_se_sum  += $10;
        cb_r2_sum  += $6; cf_r2_sum  += $11;
        n++
    }
    END {
        if (n > 0) {
            printf "%-30s %-10s %-12.4f %-12.4f %-12.4f %-12.4f %-8s\n",
                   "'"$NAME"'",
                   "'"$BP"'",
                   cb_mae_sum / n,
                   cf_mae_sum / n,
                   cb_me_sum / n,
                   cf_me_sum / n,
                   (cb_me_sum / n >= -5 && cb_me_sum / n <= 5 && cb_se_sum / n <= 8) ? "PASS" : "FAIL"
        }
    }'
done >> "$SUMMARY"

{
    echo ""
    echo "==========================================================="
    echo " KEY QUESTIONS"
    echo "==========================================================="
    echo ""
    echo "1. Does better fusion (FiLM/Gate) fix MS4 poor performance?"
    echo "   Compare MS4film, MS4gate, MS4fg vs paper MS4 (7.10/4.64)"
    echo ""
    echo "2. Do architectural improvements (conv stem + AttnPool) help?"
    echo "   Compare MS4film vs MS4fb (film_baseline)"
    echo ""
    echo "3. Is bidirectionality critical?"
    echo "   Compare MS4film (bidir) vs MS4film_uni"
    echo ""
    echo "4. Which fusion works best — FiLM, Gate, or FiLM+Gate?"
    echo "   Compare MS4film vs MS4gate vs MS4fg"
    echo ""
    echo "5. Does good S4 match or beat MResNet50?"
    echo "   Compare best MS4 variant vs MResNet50"
    echo ""
    echo "6. Can transformers with deep FiLM beat all S4 variants?"
    echo "   Compare ViT1D_FiLM vs best MS4 variant"
    echo ""
    echo "==========================================================="
} >> "$SUMMARY"

echo "Done. Results: $SUMMARY"
cat "$SUMMARY"
