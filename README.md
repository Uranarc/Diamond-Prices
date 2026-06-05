# Diamond Prices — Reproducibility and Setup

This repository contains the analysis for the Diamond Prices project (EDA, preprocessing, modelling, and report figures). The following notes document the software environment, repository layout, and reproducible steps to regenerate the figures and results used in the report.

## Software environment

The analysis was produced with Python 3 and the following core libraries:

- `numpy` — numerical computation
- `pandas` — data manipulation and profiling
- `matplotlib`, `seaborn` — visualisation
- `statsmodels` — OLS regression (smf.ols) and VIF computation
- `scikit-learn` (`sklearn`) — `train_test_split`, `StandardScaler`, `PCA`, `mean_squared_error`
- `scipy` — skewness and kurtosis statistics

A complete list of pinned dependency versions is available in `requirements.txt` at the repository root. Create a virtual environment and install dependencies from that file to reproduce the exact environment used for the analysis.

All random operations in the notebooks use `random_state=42` for reproducibility.

## Repository structure

Top-level layout for this project (relative to this folder):

```
Diamond_Prices/
│
├── data/
│   ├── raw/
│   │   └── DiamondsPrices.csv
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       ├── train_scaled.csv
│       ├── val_scaled.csv
│       └── test_scaled.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
│
├── reports/
│   └── figures/
│
├── README.md
├── requirements.txt
└── .gitignore
```

Notes:
- The canonical preprocessing pipeline is implemented in `notebooks/02_preprocessing.ipynb` and deterministically writes the processed CSVs under `data/processed/`.
- `notebooks/03_modeling.ipynb` consumes the processed CSVs and produces model outputs, metrics, and figures saved to `reports/figures/`.
