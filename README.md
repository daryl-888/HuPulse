# HuPulse — MS4Plus: Improved S4-Based Cuffless Blood Pressure Estimation

This repo contains **MS4Plus**, an improved version of the MS4 model from:

> "Benchmarking and Enhancing PPG-Based Cuffless Blood Pressure Estimation Methods"  
> Mathew, Shen, Hu, Rahimi, Zouridakis — University of Houston + Houston Methodist Hospital

## Baseline Problem

The paper's MS4 (S4 backbone + late-concat fusion) achieves MAE 7.10/4.64 mmHg (SBP/DBP) — the
worst performer. Adding demographics made it *worse* because the 512-dim S4 output drowns out the
3-dim demographic vector via simple concatenation.

## MS4Plus Improvements

| Change | Why |
|--------|-----|
| **FiLM conditioning** | Demographics predict per-channel scale+shift of S4 features instead of being concatenated. Fixes the dominance problem. |
| **Bidirectional S4D** | Forward + backward passes; captures both causal and anti-causal PPG patterns. |
| **S4D diagonal** | Simpler than full S4, no CUDA extension needed, trains more stably. |
| **Huber loss (δ=5 mmHg)** | More robust to BP outliers than MSE. |
| **AdamW + cosine LR** | Better regularisation than Adam with flat LR. |
| **PPG augmentation** | Amplitude scaling, noise, baseline wander. |
| **Gradient clipping** | Stabilises S4 training. |

## Quick Start on Carya

```bash
# 1. Clone and set up code directory
ssh dpalfaro@carya.rcdc.uh.edu
cd /project/rhu/dpalfaro/code
git clone <repo_url> HuPulse
cd HuPulse

# 2. Check data layout (run once)
sbatch jobs/explore_data.sbatch
tail -f /project/rhu/dpalfaro/results/explore_<jobid>.out

# 3. Update data_dir in configs if needed, then train both models
sbatch jobs/train_ms4_improved_SBP.sbatch
sbatch jobs/train_ms4_improved_DBP.sbatch

# 4. Monitor
squeue -u dpalfaro
tail -f /project/rhu/dpalfaro/results/ms4plus_SBP_<jobid>.out

# 5. Evaluate
/home/yshen28/miniconda3/envs/ppgdata/bin/python evaluate.py \
    --ckpt /project/rhu/dpalfaro/results/ms4plus_SBP/best_ms4plus_SBP.pt \
    --data_dir /project/rhu/PulseBP/PulseDB_multi_full
```

## Important Notes

- **Separate models for SBP and DBP** — each predicts one BP component (matching the paper's
  one-task-per-model design).
- The `ppgdata` conda environment from `ppgdata_environment.yml` (shared by Yidan Shen) has
  all required dependencies. Use `/home/yshen28/miniconda3/envs/ppgdata/bin/python` or install
  your own copy via `sbatch jobs/setup_env.sbatch`.
- Run `scripts/explore_data.py` first if the data_dir keys/layout differ from what's expected —
  the loader will print which keys it found and raise a clear error if fields are missing.
- **Never touch `/project/rhu/aakash/`**.

## File Structure

```
models/
  s4_layer.py        S4D diagonal state space layer (kernel + block)
  ms4_improved.py    MS4Plus: bidirectional S4D + FiLM fusion
data/
  pulsedb.py         Flexible PulseDB loader (mat73 / scipy / h5py)
utils/
  metrics.py         MAE, std, AAMI/ISO 81060-2 check
scripts/
  explore_data.py    Print data directory layout and array shapes
configs/
  ms4_improved_SBP.yaml
  ms4_improved_DBP.yaml
jobs/
  explore_data.sbatch
  setup_env.sbatch
  train_ms4_improved_SBP.sbatch
  train_ms4_improved_DBP.sbatch
train.py             Training entry point
evaluate.py          Evaluation on cal_based / cal_free splits
```
