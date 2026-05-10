# Code

Main entry points:

- `run_training_sweeps.py`: merged training script from the two Colab notebooks.
  It runs full-data seed-0 experiments, CIFAR-10 seed-stability experiments, and
  10%/25% low-data CIFAR experiments using the official Structured
  Initialization codebase.
- `make_report_results_figures.py`: regenerates the report figures from local
  `summary.csv` logs and processed result tables.
- `summarize_official_results.py`: summarizes official-run CSV logs into
  poster/report tables.
- `make_cifar_attention_matrix_zoom.py`: creates the trained attention-map
  visualization used in the poster/report.

The remaining files are a compact local ViT implementation and helper scripts
used while developing the reproduction.

