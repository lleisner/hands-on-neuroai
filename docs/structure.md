# Project Structure

template layout.

~~~~
project/
│
├── src/
│   └── project_name/        # Python package (rename per project)
│
├── notebooks/               # Jupyter notebooks
├── configs/                 # Experiment configuration files
├── scripts/                 # CLI entry points (train, eval, etc.)
│
├── data/
│   ├── raw/                 # Raw data (ignored by git)
│   └── processed/           # Processed data (ignored by git)
│
├── tests/                   # pytest tests
│
├── docs/                    # Project documentation
└── TODO.md                  # Optional project task list
~~~~

