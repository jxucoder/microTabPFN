# microTabPFN

A minimal ("micro") code of [TabPFN](https://github.com/PriorLabs/TabPFN) in **~210 lines** (~100 lines of core code ignoreing the comments).

Inspired by [micrograd](https://github.com/karpathy/micrograd) and [nanoTabPFN](https://github.com/automl/nanoTabPFN).

## What is TabPFN?

TabPFN uses **in-context learning** for tabular modeling:
- Pre-trained on synthetic "prior" data (SCM + BNN prior)
- At inference: pass training data as context, predict in one forward pass
- No gradient updates at test time

## Results on Iris

```
Method           ROC AUC      Std
---------------------------------
LogReg             0.983    0.011
microTabPFN        0.981    0.007  ← Matches LogReg!
RF                 0.979    0.011
KNN                0.978    0.016
Tree               0.863    0.018
```

**Training time**: ~9 min for 5000 steps on my laptop (Apple M4 Max). Slow due to no batching (educational clarity over speed).

## Quick Start

```bash
uv run python microtabpfn.py
```

## How It Works

```
┌──────────────────────────────────────────────────────────┐
│  1. PRIOR: Train on synthetic tasks                      │
│     X ~ SCM (causal features with dependencies)          │
│     y ~ BNN (random 2-layer neural net boundary)         │
├──────────────────────────────────────────────────────────┤
│  2. MODEL: Two-axis attention transformer                │
│     - Column attention: features talk to features        │
│     - Row attention: samples talk to samples             │
│       • train ↔ train (self-attention)                   │
│       • test → train (cross-attention, ICL happens here!)│
│     - No positional encoding → permutation invariant     │
├──────────────────────────────────────────────────────────┤
│  3. INFERENCE: In-context learning                       │
│     Pass (X_train, y_train, X_test) → get predictions    │
│     No gradients! Just a forward pass.                   │
└──────────────────────────────────────────────────────────┘
```

## Limitations

Educational implementation: binary classification, 4 features, ~100 samples. Full TabPFN supports multi-class, 500 features, 50K samples.

## Design Decisions (vs nanoTabPFN)

Compared to [nanoTabPFN](https://arxiv.org/abs/2511.03634), i made following changes for educational clarity:

| Change | Why |
|--------|-----|
| Generate SCM+BNN prior on-the-fly | No need for HDF5 data files, reader can see how prior works |
| Remove batch dimension | Simpler tensor shapes, easier to understand |
| Single 210-line file | Can read everything top-to-bottom |

There is no multi-class support and batching. And this is okay for educational / "micro" purpose.

**What i kept**:
- **SCM + BNN prior**: fake data that look like real tabular data
- **Two-axis attention**: attention on features and samples, both direction
- **Permutation invariance**: shuffle column or row, still work
- **In-context learning**: test look at train to guess answer

## Acknowledgments

Obviously, built with help from [Cursor](https://cursor.com) + [Claude Opus 4.5](https://www.anthropic.com/claude).

- [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](https://arxiv.org/abs/2207.01848) (ICLR 2023)
- [Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6) (Nature 2024)
- [nanoTabPFN: A Lightweight and Educational Reimplementation](https://arxiv.org/abs/2511.03634)
- [Awesome Tabular Foundation Models](https://github.com/jxucoder/Awesome-Tabular-Foundation-Models) - curated list of TFM resources
