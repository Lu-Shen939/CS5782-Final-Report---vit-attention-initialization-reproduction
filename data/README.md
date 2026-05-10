# Data

This project uses CIFAR-10 and CIFAR-100 through `torchvision`.

The datasets are not committed to this repository. The training script downloads
the full datasets automatically with the official `timm`/Structured
Initialization training pipeline. For low-data experiments, the script builds
stratified ImageFolder-style 10% and 25% subsets from the downloaded CIFAR
training split.

Default Colab paths used by `code/run_training_sweeps.py`:

- Raw/cache data: `/content/official_structured_data`
- Generated low-data ImageFolder subsets:
  `/content/official_structured_data/{cifar10,cifar100}_{10pct,25pct}_seed*_imagefolder`

