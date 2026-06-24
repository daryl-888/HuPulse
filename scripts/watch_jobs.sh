#!/bin/bash
# watch_jobs.sh — poll a set of SLURM jobs and report when all finish.
#
# Usage (run from the directory where the .out files are, e.g. ~):
#   nohup bash ~/jobs/watch_jobs.sh 7451337 7451338 7451339 7451340 \
#         7451341 7451342 7451343 7451344 > ~/watch_jobs.log 2>&1 &
#
# It checks squeue every 5 min. When none of the given job IDs are still
# in the queue, it scans the matching *_<jobid>.out files, extracts each
# final summary block, writes a combined report, and emails it.

USER_EMAIL="darylalfaro6817@gmail.com"
JOB_IDS=("$@")
POLL_SECS=300
OUT_DIR="${OUT_DIR:-$HOME}"          # where the .out files live
REPORT="$HOME/ms4_results_summary.txt"

if [ ${#JOB_IDS[@]} -eq 0 ]; then
    echo "Usage: bash watch_jobs.sh <jobid> [<jobid> ...]"
    exit 1
fi

echo "[watch] tracking jobs: ${JOB_IDS[*]}"
echo "[watch] polling every ${POLL_SECS}s; report -> $REPORT"

still_running() {
    # returns 0 if ANY tracked job is still in the queue
    local q
    q=$(squeue -h -u "$USER" -o "%i" 2>/dev/null)
    for j in "${JOB_IDS[@]}"; do
        if grep -qx "$j" <<< "$q"; then return 0; fi
    done
    return 1
}

while still_running; do
    n=$(squeue -h -u "$USER" -o "%i" | grep -cE "$(IFS='|'; echo "${JOB_IDS[*]}")" || true)
    echo "[watch] $(date '+%F %T')  $n tracked job(s) still in queue..."
    sleep "$POLL_SECS"
done

echo "[watch] all tracked jobs left the queue at $(date '+%F %T'). Building report."

{
    echo "=============================================================="
    echo " MS4 + MResNet50 run summary"
    echo " Generated: $(date '+%F %T')"
    echo " Jobs: ${JOB_IDS[*]}"
    echo "=============================================================="
    echo

    for j in "${JOB_IDS[@]}"; do
        f=$(ls -t "$OUT_DIR"/*_"$j".out 2>/dev/null | head -1)
        echo "########## job $j  ($f) ##########"
        if [ -z "$f" ]; then
            echo "  (no .out file found)"
            echo
            continue
        fi
        # Did it crash?
        if grep -qiE "Traceback|Error|CUDA out of memory|ModuleNotFound" "$f"; then
            echo "  !! possible failure — last error context:"
            grep -iE "Traceback|Error|CUDA out of memory|ModuleNotFound" "$f" | tail -5
            echo
        fi
        # Pull the final summary block (the ==== ... ==== table at the end)
        awk '/^={60,}/{c++} c>=1{print}' "$f" | tail -25
        echo
    done
} > "$REPORT"

cat "$REPORT"

# Try to email; fall back silently if no MTA configured
if command -v mail >/dev/null 2>&1; then
    mail -s "Carya MS4/MResNet50 results ($(date '+%F %T'))" "$USER_EMAIL" < "$REPORT" \
        && echo "[watch] emailed report to $USER_EMAIL" \
        || echo "[watch] mail command failed; report saved at $REPORT"
else
    echo "[watch] no 'mail' command; report saved at $REPORT"
fi

echo "[watch] done."
