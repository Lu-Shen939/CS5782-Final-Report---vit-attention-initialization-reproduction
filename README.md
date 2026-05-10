# From Mimetic to Structured Initialization: Reproducing Attention Biases for Small-Data Vision Transformers

Course project for CS 4782/5782. This repository contains our re-implementation
of attention-initialization methods for small-data Vision Transformers.

Authors: Lu Shen (ls2244), Xuanbo Jia (xj256), Yichen Li (yl3938)

## 1. Introduction

Vision Transformers can perform well at scale, but are harder to train from
scratch on small datasets because they lack the locality bias built into CNNs.
This project reproduces two initialization-based approaches for injecting useful
attention bias without changing the ViT architecture:

- Trockman & Kolter, **Mimetic Initialization of Self-Attention Layers**.
- Zheng et al., **Structured Initialization for Vision Transformers**.

## 2. Chosen Result

We reproduce the comparison between three ViT-Tiny initialization strategies:

- **Default**: standard truncated-normal initialization.
- **Mimetic**: initializes self-attention weights to mimic pretrained attention
  statistics.
- **Structured / Impulse**: initializes Q/K attention so early maps resemble
  local convolutional impulse filters.

The target result is the original paper's claim that Mimetic improves over
Default and Structured/Impulse usually improves further, especially when data is
limited.

## 3. GitHub Contents

- `code/`: training, summarization, plotting, and attention-visualization code.
- `notebooks/`: original Colab notebooks used to run full-data, seed-stability,
  and low-data experiments.
- `data/`: instructions for obtaining CIFAR-10/CIFAR-100.
- `results/`: processed figures, tables, and per-run `summary.csv` logs.
- `poster/`: final presentation poster PDF.
- `report/`: final report PDF and LaTeX source.

## 4. Re-implementation Details

We used the official Structured Initialization repository with a ViT-Tiny model
(`vit_tiny_patch16_224`) and CIFAR-10/CIFAR-100. Experiments include:

- Full-data CIFAR-10 and CIFAR-100 with seed 0.
- CIFAR-10 seed-stability runs over seeds 0, 1, and 2.
- Low-data CIFAR-10/CIFAR-100 runs using stratified 10% and 25% training
  subsets.

Metrics are best top-1 validation accuracy, training curves, seed stability, and
trained patch-to-patch attention maps.

## 5. Reproduction Steps

The main training script is:

```bash
python code/run_training_sweeps.py --suite all
```

Recommended environment:

- Google Colab or Linux machine with an A100-class GPU.
- Python 3.10+.
- PyTorch, torchvision, timm, pandas, matplotlib, and the official Structured
  Initialization dependencies.

Useful variants:

```bash
# Full-data CIFAR-10/CIFAR-100 seed-0 runs
python code/run_training_sweeps.py --suite full-seed0

# CIFAR-10 seed-stability runs
python code/run_training_sweeps.py --suite seed-stability

# 10% and 25% low-data runs
python code/run_training_sweeps.py --suite low-data
```

The script clones `https://github.com/osiriszjq/structured_initialization.git`
by default, installs its `pytorch-image-models-1.0.22` package, downloads CIFAR
datasets, and writes results to `/content/official_structured_results`. The
notebooks in `notebooks/` contain the original Colab workflow used for our runs.

To regenerate report figures from saved logs:

```bash
python code/make_report_results_figures.py
```

## 6. Results / Insights

Our reproduced results match the main qualitative trend of the papers. Mimetic
initialization consistently improves over the default ViT initialization, and
Structured/Impulse initialization usually gives the best accuracy.

Examples from our final summary:

- Full CIFAR-10: Default 89.12, Mimetic 91.99, Structured 92.47.
- Full CIFAR-100: Default 67.95, Mimetic 72.67, Structured 73.40.
- 10% CIFAR-10: Default 49.24, Mimetic 55.59, Structured 57.30.
- 25% CIFAR-100: Default 30.88, Mimetic 38.06, Structured 39.30.

Our absolute numbers are below the original paper's full reported accuracies,
likely because our course reproduction used fewer epochs, fewer seeds, and a
smaller hyperparameter sweep. The seed study also shows that Structured
initialization can have higher variance across runs, suggesting that a stronger
locality prior improves data efficiency but may increase sensitivity to early
training dynamics.

## 7. Conclusion

Initialization is a lightweight way to inject inductive bias into ViTs. Mimetic
initialization provides a robust pretrained-statistics prior, while Structured
initialization more directly programs local attention. Our experiments support
the paper's main trend while also highlighting practical sensitivity to training
budget and random seed.

## 8. References

1. Asher Trockman and J. Zico Kolter. *Mimetic Initialization of
   Self-Attention Layers*. ICML, 2023.
2. Jianqiao Zheng et al. *Structured Initialization for Vision Transformers*.
   NeurIPS, 2025.
3. Ross Wightman. *PyTorch Image Models / timm*. 2019.

## 9. Acknowledgements

This repository was created as the final project deliverable for CS 4782/5782.
We thank the course staff for the project framework and feedback.
