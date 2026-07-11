# COLM 2026 — "What AstroPT knows about galaxies"

Scripts for reproducing all results and figures in the paper
*"What AstroPT knows about galaxies, and what that can teach us about LLMs"* (COLM 2026).

All probe results, bootstrap CIs, and causal tracing outputs are at:
https://huggingface.co/datasets/HCVYM5w6Gn/colm-results

## Setup
```bash
pip install polars huggingface_hub matplotlib scipy scikit-learn torch einops datasets
export HF_TOKEN="<your_hf_token>"
export COLM_RESULTS_REPO="HCVYM5w6Gn/colm-results"
```

## Pipeline (run in order)

### 1. k-fold probing (paper Figs 2, A2, A3)
```bash
python kfold_reprobe.py --configs ar_aim_001M ar_aim_021M ar_aim_100M
```

### 2. Residual probe + direction cosines (paper Fig 3, Appendix C, D)
```bash
python reprobe_residual.py --configs ar_aim_001M ar_aim_021M ar_aim_100M
```

### 3. Causal tracing (paper Appendix D, Fig 9)
```bash
python causal_tracing.py --config ar_aim_021M --n_pairs 500
```

### 4. Bootstrap CIs (error bars in Figs 3, 8)
```bash
python bootstrap_ci.py --mode final --n_boot 300 --configs ar_aim_021M
python bootstrap_ci.py --mode patch --n_boot 1000
```

## Reproduce figures
```bash
python plot_grokking.py       --out_dir figures/   # Figs 2, A2, A3
python plot_geometry.py       --out_dir figures/   # Fig 3, Appendix C
python plot_residual_scaling.py --out_dir figures/ # Fig 8 left
python plot_with_ci.py        --out_dir figures/   # Figs 8, 3 with CIs
python plot_causal.py         --out_dir figures/   # Fig 9
```

## Experimental details
- 12 configs: {ar, mae} x {aim, affine} x {1M, 21M, 100M}
- 64 log-spaced checkpoints per config (steps 0-26212, one epoch)
- 50,000 test galaxies from Smith42/galaxies v2.0 test split
- 5-fold cross-validated ridge regression (lambda=100)
- Mass-luminosity fit on 2,000 training-split galaxies (no label leakage)
- Bootstrap CIs: 300 resamples retraining probes on bootstrapped training galaxies
