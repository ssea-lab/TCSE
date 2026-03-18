# TCSE: Temporal Causal Sequential Embedding

TCSE is a lightweight research prototype that couples **temporal segmentation**, **causal debiasing**, and **text-aware enrichment** to improve sequential recommendation. The public release focuses on the key components needed to reproduce our ablation studies without exposing internal tooling.

## Features

- **Two-channel user/item encoders** that disentangle interest-driven interactions from popularity-driven exposure bias.
- **Time-cohort weighting** that re-scales pairwise scores with a configurable schedule (`linear` or `exp`) based on discretized time buckets.
- **Optional text fusion** that injects frozen language-model embeddings through small projection heads.
- **Simple pairwise sampler** that preserves chronological order, supports time-based splits, and produces training tuples with both positive and negative time periods.

## Repository Layout

```
upload_github/
├── README.md
├── requirements.txt
├── configs/
│   └── tcse_example.yaml
├── scripts/
│   └── train_tcse.py
└── tcse/
    ├── __init__.py
    ├── data.py
    ├── metrics.py
    ├── model.py
    └── trainer.py
```

You can extend this tree with additional configs or notebooks as needed before pushing to GitHub.

## Quick Start

1. **Install dependencies**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Prepare data**
   Place your dataset under a directory that contains `train_record.csv`, `val_record.csv`, `test_record.csv`, and (optionally) `item_text_embeddings.npy`. Each CSV should have `uid,iid,ts` columns with zero-based IDs.
3. **Edit config**
   Copy `configs/tcse_example.yaml` and update the paths (`data_root`, `item_text_emb_path`, `output_dir`) plus hyper-parameters such as `int_weight`, `pop_weight`, `pop_margin`, and early-stopping controls (`monitor_metric`, `patience`).
4. **Train & evaluate**
   ```bash
   python scripts/train_tcse.py --config configs/tcse_example.yaml
   ```
   Training logs, checkpoints, and evaluation summaries are stored under the configured `output_dir`.

## Citation

If you use TCSE in academic work, please cite the accompanying paper once it is available. For now, referencing this repository is sufficient.
