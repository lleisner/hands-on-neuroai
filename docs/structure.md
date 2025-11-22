# Project Structure

template layout.

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

