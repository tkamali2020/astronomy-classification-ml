# Astronomical Object Classification (STAR / GALAXY / QSO)

A machine learning pipeline that classifies astronomical objects — stars,
galaxies, and quasars (QSO) — from photometric and spectroscopic survey
data, built to scale to large datasets (~4.6M rows) using
[Dask](https://www.dask.org/) and [Dask-ML](https://ml.dask.org/).

## Overview
The pipeline is designed to efficiently process a large-scale astronomical dataset (approximately 5 million records) using Dask for scalable data loading and preprocessing.

The pipeline consist of: 

1. **Loads** the dataset with Dask for out-of-core / parallel processing.
2. **Cleans** the data: missing-value checks, IQR-based outlier detection,
   noise inspection via boxplots/histograms.
3. **Transforms** features with Min-Max normalization followed by
   Z-score standardization.
4. **Selects features** based on correlation with the target class and
   per-class distribution analysis.
5. **Handles class imbalance** by computing per-class sample weights
   (the dataset is heavily skewed toward STAR).
6. **Trains baseline models**: Logistic Regression, Random Forest, and
   XGBoost, all trained with sample weighting.
7. **Evaluates** every model comprehensively: accuracy, macro/weighted
   precision/recall/F1, per-class metrics, ROC-AUC (OvO/OvR), confusion
   matrices, and feature importance.
8. **Tunes hyperparameters** with `RandomizedSearchCV` using
   **Stratified K-Fold** cross-validation (important given the class
   imbalance).
9. **Compares** baseline vs. hyperparameter-tuned models and saves the
   final models, best hyperparameters, and comparison report.

## Dataset

Expected input: a CSV with the following columns (after the initial
identifier/coordinate columns are dropped, the model uses 5 features):

| Feature      | Description                              |
|--------------|-------------------------------------------|
| `redshift`   | Spectroscopic redshift                   |
| `u`          | Photometric magnitude (u-band)           |
| `r`          | Photometric magnitude (r-band)           |
| `petroRad_r` | Petrosian radius (r-band)                |
| `resolved_r` | Resolved/point-source flag (r-band)      |
| `class`      | Target: `STAR`, `GALAXY`, or `QSO`       |

Columns dropped before modeling (low correlation / high redundancy):
`g`, `i`, `mag_z`, plus identifiers `bestObjID`, `ra`, `dec`,
`w1mpro`, `w2mpro`, `w3mpro`, `w4mpro`.



## Installation

Clone the repository:

```bash
git clone https://github.com/tkamali2020/astronomy-classification-ml.git
cd <your-repo>
```
Install the required dependencies:

```bash
pip install -r requirements.txt
```


## Usage

The script exposes a `main(csv_path)` function that runs the full
pipeline end to end. Call it with the path to your dataset:

```bash
python -c "from astro_classification_pipeline import main; main('path/to/data.csv')"
```

Or import individual functions in your own notebook/script:

```python
from astro_classification_pipeline import (
    load_data, encode_target, drop_unused_columns,
    build_modeling_dataset, calculate_class_weights,
    train_baseline_models, evaluate_all_models, compare_models,
    tune_random_forest, tune_xgboost,
)

data = load_data("path/to/data.csv")
data = encode_target(data)
data = drop_unused_columns(data)
# ... continue with the pipeline steps you need
```

## Results summary

*Example run* — exact numbers depend on your dataset split and version of
the underlying libraries; re-run the pipeline on your own data to get
results for your specific run. On the full dataset, baseline models
achieved:

| Model               | Accuracy | F1 (macro) |
|---------------------|----------|------------|
| Random Forest       | 97.77%   | 0.9740     |
| XGBoost             | 97.39%   | 0.9700     |
| Logistic Regression | 69.47%   | 0.5126     |

Hyperparameter tuning via `RandomizedSearchCV` did **not** meaningfully
improve on the baseline Random Forest / XGBoost results, suggesting the
baseline configurations were already close to the ceiling achievable
with the current feature set. The weakest class is consistently `QSO`,
the minority class, due to its smaller sample size.

## Project structure

```
.
├── astro_classification_pipeline.py   # Full pipeline (single file)
├── astro_classification_pipeline.ipynb # Same pipeline as a Colab notebook
├── requirements.txt
├── LICENSE
└── README.md
```

## Notes

- This script assumes a Google Colab + Google Drive environment for data
  loading (the dataset path is passed as the `csv_path` argument to
  `main()`); adjust the path resolution as needed for other environments.
- A Dask `distributed.Client` is started inside the normalization,
  standardization, and correlation functions. For repeated runs in the
  same process, consider hoisting the client creation to a single
  long-lived client to avoid repeated cluster start/stop overhead.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE)
for details.
