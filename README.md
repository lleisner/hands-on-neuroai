# hands-on-neuroai

A minimal project exploring superposition and catastrophic forgetting, inspired by Anthropic's "Superposition of Many Models in One." This repository implements baseline MLPs, PSP (parameter superposition) models, and Permuted-MNIST experiments for studying how different context types and hidden dimensions affect interference between tasks.

## Project Structure

~~~~
hands-on-neuroai/
│
├── scripts/
│   ├── download_data.py          # utility to download MNIST
│   └── run_perm_mnist.py         # main experiment entry point
│
├── src/
│   └── hands_on_neuroai/
│       ├── data/
│       │   ├── download.py       # data download helpers
│       │   └── mnist.py          # MNIST + PermutedMNIST datasets/loaders
│       │
│       ├── models/
│       │   ├── context.py        # binary/complex/rotation context generators
│       │   ├── factory.py        # model builder (baseline vs PSP)
│       │   ├── mlp.py            # baseline MLP models
│       │   └── psp.py            # PSPLinear + PSPMLP
│       │
│       ├── training/
│       │   ├── metrics.py        # evaluation helpers
│       │   └── perm_mnist.py     # training loop + experiment config
│       │
│       └── utils/
│           ├── cli.py            # argument parsing + sweep setup
│           ├── io.py             # result saving/path helpers
│           └── utils.py          # misc utilities (if needed)
│
├── tests/
│   └── test_psp.py               # PSP + context generator tests
│
├── notebooks/                    # exploratory notebooks (PCA, etc.)
├── configs/                      # optional config files
├── docs/                         # documentation
├── data/
│   ├── raw/                      # raw dataset (gitignored)
│   └── processed/                # processed dataset (gitignored)
│
└── TODO.md                       # optional task list

~~~~


## Running Experiments
### 1. Create environment
```
conda env create -f environment.yml
conda activate hands-on-neuroai
```
### 2. Run a single experiment
```
python scripts/run_perm_mnist.py \
  --hidden-dim 512 \
  --context-type binary \
  --num-tasks 10 \
  --steps-per-task 1000
```
### 3. Run sweeps (example)
```
python scripts/run_perm_mnist.py \
  --hidden-dim 128 256 512 1024 \
  --context-type none binary \
  --num-tasks 10 \
  --steps-per-task 1000
```