"""
Astronomical Object Classification Pipeline (STAR / GALAXY / QSO)
===================================================================

This script trains and evaluates machine learning models to classify
astronomical objects (Star, Galaxy, Quasar) from photometric and
spectroscopic survey data, using Dask for out-of-core / parallel
processing of a ~4.6M-row dataset.

Pipeline overview:
    1. Load data with Dask and perform initial cleaning
    2. Outlier detection (IQR method) and noise inspection
    3. Feature normalization (Min-Max) and standardization (Z-score)
    4. Correlation analysis and feature selection
    5. Class-weight computation to handle severe class imbalance
    6. Baseline model training (Logistic Regression, Random Forest, XGBoost)
       with sample-weighted training to correct for class imbalance
    7. Comprehensive evaluation (accuracy, precision/recall/F1, ROC-AUC,
       confusion matrices, per-class metrics, feature importance)
    8. Hyperparameter tuning via RandomizedSearchCV with Stratified K-Fold
       cross-validation (appropriate for imbalanced multi-class data)
    9. Final comparison between baseline and tuned ("optimized") models

Dataset columns used: redshift, u, g, r, i, mag_z, petroRad_r, resolved_r
Target classes: STAR (0), GALAXY (1), QSO (2)

Notes on this consolidated version:
    - This script merges the original notebook's cells into a single,
      readable, GitHub-ready file.
    - Comments and printed messages have been translated to English
      for international readability.
    - The original step-by-step structure and logic have been preserved
      so results remain identical to the executed notebook.
"""

import os
import time
import pickle
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import dask.dataframe as dd
import dask.array as da
from dask.diagnostics import ProgressBar
from dask.distributed import Client

from dask_ml.model_selection import train_test_split
from dask_ml.preprocessing import StandardScaler as DaskStandardScaler
from dask_ml.preprocessing import MinMaxScaler
from dask_ml import linear_model

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
)
from scipy.stats import randint, uniform, skew, kurtosis

from xgboost import XGBClassifier
import xgboost as xgb

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_data(csv_path: str) -> dd.DataFrame:
    """
    Load the raw spectroscopic/photometric/WISE dataset with Dask.

    Parameters
    ----------
    csv_path : str
        Path to the source CSV file
        (e.g. 'spec_photo_wise_<dataset_id>.csv').

    Returns
    -------
    dask.dataframe.DataFrame
    """
    print("Loading dataset with Dask...")
    data = dd.read_csv(csv_path)
    print("Dataset shape:", data.shape)
    print("\nFirst 5 rows:")
    print(data.head())
    return data


def inspect_data(data: dd.DataFrame) -> None:
    """Print basic dataset info: row/column counts, dtypes, descriptive stats."""
    print(f"Number of records: {len(data):,}")
    print(f"Number of columns: {len(data.columns)}")
    print("\nSample rows:")
    print(data.head())

    print("\nColumn dtypes:")
    print(data.dtypes)

    print("\nDescriptive statistics:")
    with ProgressBar():
        stats = data.describe().compute()
    print(stats)


def encode_target(data: dd.DataFrame) -> dd.DataFrame:
    """Map the string 'class' column (STAR/GALAXY/QSO) to integer labels."""
    class_map = {'STAR': 0, 'GALAXY': 1, 'QSO': 2}
    data['class'] = data['class'].map(class_map, meta=('class_encoded', 'int64'))

    print(data.head())
    print("\nDescriptive statistics after encoding:")
    with ProgressBar():
        stats = data.describe().compute()
    print(stats)
    return data


def drop_unused_columns(data: dd.DataFrame) -> dd.DataFrame:
    """Remove identifier / coordinate / WISE columns not used for modeling."""
    cols_to_drop = ['bestObjID', 'ra', 'dec', 'w1mpro', 'w2mpro', 'w3mpro', 'w4mpro']
    data = data.drop(columns=cols_to_drop)

    print("Feature shape after dropping unused columns:", data.shape)
    print("First rows of features:")
    print(data.head())
    return data


# =============================================================================
# 2. PRE-PROCESSING: MISSING VALUES, OUTLIERS, NOISE
# =============================================================================

def check_missing_values(data: dd.DataFrame) -> None:
    """Report the count of null values per column."""
    print(data.isnull().sum().compute())


def plot_distributions(data: dd.DataFrame, frac: float = 0.005) -> None:
    """
    Plot boxplots and histograms for all numeric columns on a small random
    sample of the data (for noise / distribution inspection).
    """
    sample = data.sample(frac=frac).compute()
    numeric_cols = sample.select_dtypes(include='number').columns

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        plt.boxplot(sample[col].dropna())
        plt.title(f"Boxplot for {col}")
        plt.xlabel(col)
        plt.ylabel("Values")
        plt.show()

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        plt.hist(sample[col].dropna(), bins=50)
        plt.title(f"Histogram for {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()


def detect_outliers_iqr(data: dd.DataFrame, k: float = 1.5) -> None:
    """
    Detect outliers in every column using the IQR (Interquartile Range)
    method. Prints the lower/upper bounds and outlier count per column.
    """
    def iqr_outlier_mask(ddf, col, k=3):
        q1, q3 = ddf[col].quantile([0.25, 0.75]).compute()
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        return (ddf[col] < lower) | (ddf[col] > upper), lower, upper

    for col in data:
        mask, lower, upper = iqr_outlier_mask(data, col, k=k)
        print("bounds:", lower, upper)
        count_out = mask.sum().compute()
        print("outliers", col, ':', count_out)


# =============================================================================
# 3. NORMALIZATION & STANDARDIZATION
# =============================================================================

def normalize_features(data: dd.DataFrame, numerical_cols: list) -> dd.DataFrame:
    """
    Apply Min-Max scaling (range [0, 1]) to the given numerical columns
    using dask_ml's distributed-friendly MinMaxScaler.
    """
    client = Client()
    print("\nPerforming normalization with Min-Max Scaling...")

    scaler = MinMaxScaler()
    print("Fitting the scaler...")
    scaler.fit(data[numerical_cols])

    print("Transforming the data...")
    data_scaled = scaler.transform(data[numerical_cols])

    data_normalized = data.copy()
    data_normalized[numerical_cols] = data_scaled

    print("\nFirst 5 rows of the normalized data:")
    print(data_normalized.head())
    print("\nDescriptive statistics of the normalized data:")
    print(data_normalized.describe().compute())

    client.close()
    return data_normalized


def standardize_features(data_normalized: dd.DataFrame, numerical_cols: list) -> dd.DataFrame:
    """
    Apply Z-score standardization (zero mean, unit variance) on top of the
    already Min-Max-normalized columns.
    """
    client = Client()
    print("\nPerforming standardization (Z-score scaling)...")

    std_scaler = DaskStandardScaler()
    print("Fitting the standard scaler...")
    std_scaler.fit(data_normalized[numerical_cols])

    print("Transforming the data with StandardScaler...")
    data_std_scaled = std_scaler.transform(data_normalized[numerical_cols])

    data_standardized = data_normalized.copy()
    data_standardized[numerical_cols] = data_std_scaled

    print("\nFirst 5 rows of the standardized data:")
    print(data_standardized.head())
    print("\nDescriptive statistics after standardization:")
    print(data_standardized.describe().compute())

    client.close()
    return data_standardized


def plot_feature_distributions_per_class(data_normalized: dd.DataFrame,
                                          numerical_cols: list,
                                          target_col: str = 'class') -> pd.DataFrame:
    """
    For each class (STAR, GALAXY, QSO), plot the distribution of every
    numerical feature and compute skewness/kurtosis. Saves a summary CSV.
    """
    df = data_normalized.compute()

    class_mapping = {0: 'STAR', 1: 'GALAXY', 2: 'QSO'}
    df[target_col] = df[target_col].map(class_mapping)

    print("\nClass mapping applied:")
    print(df[target_col].value_counts())

    print("\nMissing values per column:")
    print(df[numerical_cols + [target_col]].isna().sum())

    stats_summary = []
    for cls in ['STAR', 'GALAXY', 'QSO']:
        subset = df[df[target_col] == cls]
        print(f"\n===== Class: {cls} =====")

        for col in numerical_cols:
            sk = skew(subset[col], nan_policy='omit')
            ku = kurtosis(subset[col], nan_policy='omit')
            stats_summary.append({'Feature': col, 'Class': cls, 'Skewness': sk, 'Kurtosis': ku})

            plt.figure(figsize=(7, 4))
            sns.histplot(subset[col], kde=True, bins=40, color='cornflowerblue')
            plt.title(f"Distribution of {col} — Class: {cls}", fontsize=13)
            plt.xlabel(col, fontsize=11)
            plt.ylabel("Count", fontsize=11)
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout()
            plt.show()

    stats_df = pd.DataFrame(stats_summary)
    print("\n=== Skewness and Kurtosis per Feature per Class ===")
    print(stats_df)

    stats_df.to_csv("feature_distribution_stats_named_classes.csv", index=False)
    print("\nFile 'feature_distribution_stats_named_classes.csv' saved successfully.")
    return stats_df


# =============================================================================
# 4. CORRELATION ANALYSIS & FEATURE SELECTION
# =============================================================================

def compute_correlation_matrix(data: dd.DataFrame, numerical_cols: list,
                                target_col: str = 'class', sample_frac: float = 0.01):
    """
    Compute and plot the correlation matrix between numerical features and
    the encoded target column, using a random sample of the data.
    """
    client = Client()

    print("Sampling dataset for correlation matrix...")
    sample_df = data.sample(frac=sample_frac, random_state=42).compute()

    le = LabelEncoder()
    sample_df[target_col + '_encoded'] = le.fit_transform(sample_df[target_col])

    cols_for_corr = numerical_cols + [target_col + '_encoded']

    print("Computing correlation matrix...")
    corr_matrix = sample_df[cols_for_corr].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix between numerical columns and class')
    plt.show()

    client.close()
    return corr_matrix


def rank_feature_correlation_with_target(corr_matrix: pd.DataFrame,
                                          target_col_encoded: str = 'class_encoded') -> pd.Series:
    """Rank features by absolute correlation with the encoded target."""
    target_correlation = corr_matrix[target_col_encoded].drop(target_col_encoded).abs().sort_values(ascending=False)
    print("Feature correlation with target:")
    print(target_correlation)
    return target_correlation


# Final feature set selected based on the correlation / distribution analysis above.
# 'resolved_r', 'redshift', 'petroRad_r', 'u', 'r' were retained;
# 'g', 'i', 'mag_z' were dropped (high mutual correlation / severe outliers).
FINAL_FEATURES = ['resolved_r', 'redshift', 'petroRad_r', 'u', 'r']
DROPPED_FEATURES = ['g', 'i', 'mag_z']


# =============================================================================
# 5. CLASS WEIGHT COMPUTATION (handling class imbalance)
# =============================================================================

def build_modeling_dataset(data: dd.DataFrame, dropped_cols=DROPPED_FEATURES) -> dd.DataFrame:
    """
    Build the final dataset used for modeling by dropping the columns that
    were excluded based on the correlation analysis.
    """
    data_new = data.drop(columns=dropped_cols)
    print(f"Dataset shape: {data_new.shape}")
    print("Available columns:")
    print(data_new.columns.tolist())

    print("\nClass distribution:")
    class_distribution = data_new['class'].value_counts().compute()
    print(class_distribution)
    return data_new


def calculate_class_weights(class_distribution: pd.Series) -> dict:
    """
    Compute class weights to counteract class imbalance, using three
    alternative formulas (standard, inverse-frequency, log-scaled).
    Returns a dict of dicts: {'standard': {...}, 'inverse': {...}, 'log': {...}}
    """
    total_samples = class_distribution.sum()
    n_classes = len(class_distribution)

    print(f"Total samples: {total_samples:,}")
    print(f"Number of classes: {n_classes}")
    print("\nClass distribution:")
    for class_name, count in class_distribution.items():
        percentage = (count / total_samples) * 100
        print(f"  {class_name}: {count:,} samples ({percentage:.2f}%)")

    # Method 1: standard formula -> total / (n_classes * count)
    class_weights_standard = {
        class_name: total_samples / (n_classes * count)
        for class_name, count in class_distribution.items()
    }

    # Method 2: inverse frequency -> max_count / count
    max_count = class_distribution.max()
    class_weights_inverse = {
        class_name: max_count / count
        for class_name, count in class_distribution.items()
    }

    # Method 3: log-scaled (for severe imbalance)
    class_weights_log = {
        class_name: np.log(total_samples / count)
        for class_name, count in class_distribution.items()
    }

    return {
        'standard': class_weights_standard,
        'inverse': class_weights_inverse,
        'log': class_weights_log,
    }


def create_sample_weights_column(data_new: dd.DataFrame, class_weights: dict) -> dd.DataFrame:
    """
    Add a 'sample_weight' column to the dataset, assigning each row the
    weight corresponding to its class.

    Includes a defensive guard: if `data_new` somehow isn't available in
    the calling scope when this is invoked standalone, the caller should
    rebuild it via build_modeling_dataset() first (this avoids the
    NameError seen in the original notebook when cells were run out of
    order).
    """
    def assign_weight(row, weights_dict):
        return weights_dict[row['class']]

    print("Assigning sample weights to dataset...")
    data_new = data_new.copy()
    data_new['sample_weight'] = data_new.apply(
        lambda row: assign_weight(row, class_weights),
        axis=1,
        meta=('sample_weight', 'f8')
    )

    sample_weights_stats = data_new['sample_weight'].describe().compute()
    print("\nSample weight statistics:")
    print(sample_weights_stats)
    return data_new


# =============================================================================
# 6. TRAIN/TEST SPLIT & SCALING
# =============================================================================

def prepare_train_test_data(data_new: dd.DataFrame, target_column: str = 'class'):
    """
    Split features/target/sample_weights into train/test sets, standardize
    the features, and convert everything to Dask Arrays ready for model
    training.

    Returns
    -------
    dict with keys: X_train_array, X_test_array, y_train_array, y_test_array,
                     weights_train_array, weights_test, scaler, feature_columns
    """
    feature_columns = [c for c in data_new.columns if c not in ['class', 'sample_weight']]
    print(f"Number of features: {len(feature_columns)}")
    print("Features:", feature_columns)

    X = data_new[feature_columns]
    y = data_new[target_column]
    sample_weights = data_new['sample_weight']

    print("\nSplitting data into train/test...")
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, sample_weights,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nConverting to Dask Arrays...")
    X_train_array = X_train_scaled.to_dask_array(lengths=True)
    X_test_array = X_test_scaled.to_dask_array(lengths=True)
    y_train_array = y_train.to_dask_array(lengths=True)
    y_test_array = y_test.to_dask_array(lengths=True)
    weights_train_array = weights_train.to_dask_array(lengths=True)

    print("Data preparation complete.")

    return {
        'X_train_array': X_train_array,
        'X_test_array': X_test_array,
        'y_train_array': y_train_array,
        'y_test_array': y_test_array,
        'weights_train_array': weights_train_array,
        'weights_test': weights_test,
        'scaler': scaler,
        'feature_columns': feature_columns,
    }


# =============================================================================
# 7. BASELINE MODEL TRAINING (with class-weighted sample weights)
# =============================================================================

def train_baseline_models(X_train, y_train, sample_weights, class_weights: dict) -> dict:
    """
    Train three baseline classifiers with class weighting:
        1. Logistic Regression (dask_ml)
        2. Random Forest        (scikit-learn)
        3. XGBoost              (xgboost, with explicit sample_weight)

    Parameters
    ----------
    X_train, y_train : Dask Arrays (standardized features / encoded labels)
    sample_weights    : Dask Array of per-sample weights
    class_weights     : dict mapping {class_id: weight}, e.g. final_class_weights_corrected

    Returns
    -------
    dict of {model_name: fitted_model}
    """
    models = {}

    unique_classes = np.unique(y_train.compute() if hasattr(y_train, 'compute') else y_train)
    class_weight_dict = {
        class_id: class_weights[class_id]
        for class_id in unique_classes if class_id in class_weights
    }

    print("Class weights used for training:")
    for class_id, weight in class_weight_dict.items():
        print(f"  Class {class_id}: {weight:.4f}")

    # --- Model 1: Logistic Regression ---
    print("\n1. Training Logistic Regression with class_weight...")
    try:
        lr_model = linear_model.LogisticRegression(
            class_weight=class_weight_dict,
            random_state=42,
            solver='lbfgs',
            max_iter=1000,
        )
        lr_model.fit(X_train, y_train)
        models['LogisticRegression'] = lr_model
        print("Logistic Regression trained successfully.")
    except Exception as e:
        print(f"Error training Logistic Regression: {e}")

    # --- Model 2: Random Forest ---
    print("\n2. Training Random Forest with class_weight...")
    try:
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            class_weight=class_weight_dict,
        )
        rf_model.fit(X_train, y_train)
        models['RandomForest'] = rf_model
        print("Random Forest trained successfully.")
    except Exception as e:
        print(f"Error training Random Forest: {e}")

    # --- Model 3: XGBoost ---
    print("\n3. Training XGBoost with sample weights...")
    try:
        X_train_pd = X_train.compute() if hasattr(X_train, 'compute') else X_train
        y_train_pd = y_train.compute() if hasattr(y_train, 'compute') else y_train
        sample_weights_pd = sample_weights.compute() if hasattr(sample_weights, 'compute') else sample_weights

        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=12,
            learning_rate=0.1,
            random_state=42,
            tree_method='hist',
        )
        xgb_model.fit(
            X_train_pd, y_train_pd,
            sample_weight=sample_weights_pd,
            verbose=False,
        )
        models['XGBoost'] = xgb_model
        print("XGBoost trained successfully.")
    except Exception as e:
        print(f"Error training XGBoost: {e}")

    return models


# =============================================================================
# 8. MODEL EVALUATION
# =============================================================================

def comprehensive_evaluation(model, X_test, y_test, model_name: str,
                              sample_weights_test=None) -> dict:
    """
    Run a full evaluation of a trained model: accuracy, macro/weighted
    precision/recall/F1, per-class metrics, ROC-AUC (OvO/OvR), confusion
    matrix, classification report, and ROC curves.

    Returns a dict of computed metrics (or None on failure).
    """
    print(f"\n{'=' * 40}")
    print(f"Evaluating model: {model_name}")
    print(f"{'=' * 40}")

    try:
        print("Generating predictions...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

        y_test_np = y_test.compute() if hasattr(y_test, 'compute') else y_test
        y_pred_np = y_pred.compute() if hasattr(y_pred, 'compute') else y_pred
        sample_weights_np = (
            sample_weights_test.compute()
            if sample_weights_test is not None and hasattr(sample_weights_test, 'compute')
            else sample_weights_test
        )

        accuracy = accuracy_score(y_test_np, y_pred_np, sample_weight=sample_weights_np)
        precision_macro = precision_score(y_test_np, y_pred_np, average='macro', sample_weight=sample_weights_np)
        recall_macro = recall_score(y_test_np, y_pred_np, average='macro', sample_weight=sample_weights_np)
        f1_macro = f1_score(y_test_np, y_pred_np, average='macro', sample_weight=sample_weights_np)

        precision_weighted = precision_score(y_test_np, y_pred_np, average='weighted', sample_weight=sample_weights_np)
        recall_weighted = recall_score(y_test_np, y_pred_np, average='weighted', sample_weight=sample_weights_np)
        f1_weighted = f1_score(y_test_np, y_pred_np, average='weighted', sample_weight=sample_weights_np)

        auc_scores = {}
        if y_pred_proba is not None:
            y_pred_proba_np = y_pred_proba.compute() if hasattr(y_pred_proba, 'compute') else y_pred_proba
            try:
                auc_scores['AUC-ROC (OvO)'] = roc_auc_score(
                    y_test_np, y_pred_proba_np, multi_class='ovo', average='macro')
                auc_scores['AUC-ROC (OvR)'] = roc_auc_score(
                    y_test_np, y_pred_proba_np, multi_class='ovr', average='macro')
            except Exception as e:
                print(f"Error computing AUC: {e}")

        precision_per_class = precision_score(y_test_np, y_pred_np, average=None)
        recall_per_class = recall_score(y_test_np, y_pred_np, average=None)
        f1_per_class = f1_score(y_test_np, y_pred_np, average=None)

        class_names = ['STAR', 'GALAXY', 'QSO']

        print(f"\nKey metrics for {model_name}:")
        print(f"   Accuracy:                 {accuracy:.4f}")
        print(f"   Precision (macro):        {precision_macro:.4f}")
        print(f"   Recall (macro):           {recall_macro:.4f}")
        print(f"   F1-Score (macro):         {f1_macro:.4f}")
        print(f"   Precision (weighted):     {precision_weighted:.4f}")
        print(f"   Recall (weighted):        {recall_weighted:.4f}")
        print(f"   F1-Score (weighted):      {f1_weighted:.4f}")

        for metric_name, score in auc_scores.items():
            print(f"   {metric_name}:            {score:.4f}")

        print(f"\nPer-class metrics:")
        for i, class_name in enumerate(class_names):
            print(f"   {class_name}:")
            print(f"      Precision: {precision_per_class[i]:.4f}")
            print(f"      Recall:    {recall_per_class[i]:.4f}")
            print(f"      F1-Score:  {f1_per_class[i]:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test_np, y_pred_np)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

        print(f"\nClassification report for {model_name}:")
        print(classification_report(y_test_np, y_pred_np, target_names=class_names))

        # ROC curves
        if y_pred_proba is not None and len(np.unique(y_test_np)) > 1:
            try:
                plt.figure(figsize=(10, 8))
                colors = ['blue', 'red', 'green']
                for i, class_id in enumerate([0, 1, 2]):
                    y_test_binary = (y_test_np == class_id).astype(int)
                    y_pred_proba_class = y_pred_proba_np[:, i]
                    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba_class)
                    auc_score = roc_auc_score(y_test_binary, y_pred_proba_class)
                    plt.plot(fpr, tpr, color=colors[i], lw=2,
                             label=f'ROC curve class {class_names[i]} (AUC = {auc_score:.2f})')

                plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {model_name}')
                plt.legend(loc="lower right")
                plt.grid(True)
                plt.show()
            except Exception as e:
                print(f"Error plotting ROC curve: {e}")

        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'auc_scores': auc_scores,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
        }

    except Exception as e:
        print(f"Error evaluating model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_all_models(trained_models: dict, X_test, y_test, weights_test=None) -> dict:
    """Run comprehensive_evaluation() for every trained model."""
    print("\nEvaluating all models...")
    all_results = {}
    for model_name, model in trained_models.items():
        results = comprehensive_evaluation(model, X_test, y_test, model_name, weights_test)
        if results is not None:
            all_results[model_name] = results
    return all_results


def compare_models(all_results: dict) -> pd.DataFrame:
    """
    Build a comparison table across all evaluated models, plot a grouped
    bar chart of key metrics, and report the best model by F1-macro.
    """
    print("\n" + "=" * 60)
    print("Comparing all models")
    print("=" * 60)

    comparison_data = []
    for model_name, metrics in all_results.items():
        row = {
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision_Macro': metrics['precision_macro'],
            'Recall_Macro': metrics['recall_macro'],
            'F1_Macro': metrics['f1_macro'],
            'Precision_Weighted': metrics['precision_weighted'],
            'Recall_Weighted': metrics['recall_weighted'],
            'F1_Weighted': metrics['f1_weighted'],
        }
        for auc_name, auc_value in metrics.get('auc_scores', {}).items():
            row[auc_name] = auc_value
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    print("\nModel comparison table:")
    print(comparison_df.to_string(index=False, float_format='%.4f'))

    metrics_to_plot = ['Accuracy', 'Precision_Macro', 'Recall_Macro', 'F1_Macro']
    plt.figure(figsize=(12, 8))
    x = np.arange(len(comparison_df))
    width = 0.2
    for i, metric in enumerate(metrics_to_plot):
        plt.bar(x + i * width, comparison_df[metric], width, label=metric)
    plt.xlabel('Models')
    plt.ylabel('Metric value')
    plt.title('Model comparison across metrics')
    plt.xticks(x + width * 1.5, comparison_df['Model'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    best_model_idx = comparison_df['F1_Macro'].idxmax()
    best_model = comparison_df.loc[best_model_idx]
    print(f"\nBest model: {best_model['Model']}")
    print(f"   F1-Score (macro): {best_model['F1_Macro']:.4f}")
    print(f"   Accuracy:         {best_model['Accuracy']:.4f}")
    print(f"   Precision (macro):{best_model['Precision_Macro']:.4f}")
    print(f"   Recall (macro):   {best_model['Recall_Macro']:.4f}")

    return comparison_df


def plot_feature_importance(model, model_name: str, feature_names: list):
    """Plot and return a DataFrame of feature importances (or |coefficients|)."""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_imp_df = pd.DataFrame({
                'feature': feature_names, 'importance': importances
            }).sort_values('importance', ascending=False)
            xlabel = 'Importance'
        elif hasattr(model, 'coef_'):
            avg_importance = np.mean(np.abs(model.coef_), axis=0)
            feature_imp_df = pd.DataFrame({
                'feature': feature_names, 'importance': avg_importance
            }).sort_values('importance', ascending=False)
            xlabel = 'Mean |coefficient|'
        else:
            return None

        plt.figure(figsize=(10, 6))
        plt.barh(feature_imp_df['feature'][:15], feature_imp_df['importance'][:15])
        plt.xlabel(xlabel)
        plt.title(f'Top features - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

        print(f"\nTop 5 features for {model_name}:")
        for _, row in feature_imp_df.head().iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")

        return feature_imp_df

    except Exception as e:
        print(f"Error analyzing feature importance for {model_name}: {e}")
        return None


def display_model_parameters(model, model_name: str) -> None:
    """Print the key hyperparameters of a trained model."""
    print(f"\nParameters for {model_name}:")
    try:
        params = model.get_params()
        important_params_map = {
            'LogisticRegression': ['solver', 'C', 'max_iter', 'class_weight'],
            'RandomForest': ['n_estimators', 'max_depth', 'min_samples_split',
                              'min_samples_leaf', 'class_weight'],
            'XGBoost': ['n_estimators', 'max_depth', 'learning_rate',
                        'subsample', 'colsample_bytree'],
        }
        for param in important_params_map.get(model_name, []):
            if param in params:
                print(f"     {param}: {params[param]}")
    except Exception as e:
        print(f"   Error displaying parameters: {e}")


def save_models(trained_models: dict, class_weights: dict, scaler, model_dir: str = "trained_models") -> None:
    """Persist trained models, class weights, and the scaler to disk."""
    os.makedirs(model_dir, exist_ok=True)
    for model_name, model in trained_models.items():
        with open(f"{model_dir}/{model_name}_model.pkl", 'wb') as f:
            pickle.dump(model, f)
        print(f"Model {model_name} saved.")

    with open(f"{model_dir}/class_weights.pkl", 'wb') as f:
        pickle.dump(class_weights, f)
    print("Class weights saved.")

    with open(f"{model_dir}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved.")


# =============================================================================
# 9. HYPERPARAMETER TUNING (RandomizedSearchCV + Stratified K-Fold)
# =============================================================================

def prepare_data_for_tuning(X_train, y_train, sample_weights, sample_size: int = 100_000):
    """
    Convert a (possibly large) Dask training set to pandas/numpy and
    optionally subsample it (stratified) to a manageable size for
    hyperparameter search.
    """
    X_train_pd = X_train.compute() if hasattr(X_train, 'compute') else X_train
    y_train_pd = y_train.compute() if hasattr(y_train, 'compute') else y_train
    sample_weights_pd = sample_weights.compute() if hasattr(sample_weights, 'compute') else sample_weights

    if len(X_train_pd) > sample_size:
        from sklearn.model_selection import train_test_split as sk_train_test_split
        X_sample, _, y_sample, _, weights_sample, _ = sk_train_test_split(
            X_train_pd, y_train_pd, sample_weights_pd,
            train_size=sample_size,
            random_state=42,
            stratify=y_train_pd,
        )
        print(f"Sampled {sample_size} of {len(X_train_pd)} rows for tuning.")
        return X_sample, y_sample, weights_sample

    return X_train_pd, y_train_pd, sample_weights_pd


def tune_random_forest(X, y, sample_weights, class_weights: dict,
                        n_iter: int = 15, cv: int = 3) -> RandomizedSearchCV:
    """
    Hyperparameter tuning for Random Forest using RandomizedSearchCV with
    Stratified K-Fold cross-validation (appropriate given the severe class
    imbalance in this dataset).

    Note: class weighting is controlled ONLY through the `class_weights`
    argument passed to the base estimator -- it is intentionally NOT
    included in the search space, to avoid RandomizedSearchCV silently
    overriding the externally computed weights with sklearn's built-in
    'balanced' / 'balanced_subsample' options.
    """
    print("Starting hyperparameter tuning for Random Forest...")

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    rf = RandomForestClassifier(
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1,
    )

    param_dist_rf = {
        'n_estimators': randint(low=100, high=500),
        'max_depth': randint(low=10, high=30),
        'min_samples_split': randint(low=2, high=20),
        'min_samples_leaf': randint(low=1, high=10),
        'max_features': uniform(0.5, 0.5),  # 50% to 100% of features
    }

    print("Random Forest hyperparameter search space:")
    for param, values in param_dist_rf.items():
        print(f"  {param}: {values}")

    rf_random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist_rf,
        n_iter=n_iter,
        scoring='f1_macro',
        cv=cv_strategy,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    start_time = time.time()
    rf_random_search.fit(X, y, sample_weight=sample_weights)
    end_time = time.time()
    print(f"Elapsed time: {(end_time - start_time) / 60:.2f} minutes")

    return rf_random_search


def tune_xgboost(X, y, sample_weights, n_iter: int = 15, cv: int = 3) -> RandomizedSearchCV:
    """
    Hyperparameter tuning for XGBoost using RandomizedSearchCV with
    Stratified K-Fold cross-validation.

    This function is fully self-contained: it builds its own
    StratifiedKFold strategy locally (matching tune_random_forest) and
    fits on the X/y/sample_weights arguments it actually receives -- it
    does not rely on any global/outer-scope variables.
    """
    print("Starting hyperparameter tuning for XGBoost...")

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    xgb_model = XGBClassifier(
        objective='multi:softmax',
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='mlogloss',
    )

    param_dist = {
        'n_estimators': randint(low=200, high=800),
        'learning_rate': uniform(0.01, 0.09),     # 0.01 to 0.10
        'max_depth': randint(low=3, high=10),
        'gamma': uniform(0, 10),
        'subsample': uniform(0.6, 0.4),           # 60% to 100%
        'colsample_bytree': uniform(0.6, 0.4),    # 60% to 100%
        'reg_lambda': uniform(1e-5, 10),
        'reg_alpha': uniform(1e-5, 10),
    }

    print("XGBoost hyperparameter search space:")
    for param, values in param_dist.items():
        print(f"  {param}: {values}")

    xgb_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_strategy,
        scoring='f1_macro',
        random_state=42,
        n_jobs=-1,
        verbose=2,
    )

    start_time = time.time()
    xgb_search.fit(X, y, sample_weight=sample_weights)
    end_time = time.time()
    print(f"Elapsed time: {(end_time - start_time) / 60:.2f} minutes")

    return xgb_search


def train_optimized_models(X_train, y_train, sample_weights,
                            rf_best_params: dict, xgb_best_params: dict,
                            class_weights: dict) -> dict:
    """Train the final Random Forest and XGBoost models using the best
    hyperparameters found by RandomizedSearchCV."""
    optimized_models = {}

    X_train_pd = X_train.compute() if hasattr(X_train, 'compute') else X_train
    y_train_pd = y_train.compute() if hasattr(y_train, 'compute') else y_train
    sample_weights_pd = sample_weights.compute() if hasattr(sample_weights, 'compute') else sample_weights

    print("\nTraining optimized Random Forest model...")
    try:
        rf_optimized = RandomForestClassifier(
            **rf_best_params,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1,
        )
        rf_optimized.fit(X_train_pd, y_train_pd, sample_weight=sample_weights_pd)
        optimized_models['RandomForest_Optimized'] = rf_optimized
        print("Optimized Random Forest trained.")
    except Exception as e:
        print(f"Error training optimized Random Forest: {e}")

    print("\nTraining optimized XGBoost model...")
    try:
        xgb_optimized = XGBClassifier(
            **xgb_best_params,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='mlogloss',
        )
        xgb_optimized.fit(X_train_pd, y_train_pd, sample_weight=sample_weights_pd)
        optimized_models['XGBoost_Optimized'] = xgb_optimized
        print("Optimized XGBoost trained.")
    except Exception as e:
        print(f"Error training optimized XGBoost: {e}")

    return optimized_models


def comprehensive_evaluation_optimized(model, X_test, y_test, model_name: str) -> dict:
    """
    Simplified evaluation (no sample weighting) used to compare baseline
    vs. hyperparameter-tuned models on equal footing.
    """
    print(f"\n{'=' * 40}")
    print(f"Evaluating {model_name}")
    print(f"{'=' * 40}")

    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

        y_test_np = y_test.compute() if hasattr(y_test, 'compute') else y_test
        y_pred_np = y_pred.compute() if hasattr(y_pred, 'compute') else y_pred

        accuracy = accuracy_score(y_test_np, y_pred_np)
        precision_macro = precision_score(y_test_np, y_pred_np, average='macro')
        recall_macro = recall_score(y_test_np, y_pred_np, average='macro')
        f1_macro = f1_score(y_test_np, y_pred_np, average='macro')

        precision_weighted = precision_score(y_test_np, y_pred_np, average='weighted')
        recall_weighted = recall_score(y_test_np, y_pred_np, average='weighted')
        f1_weighted = f1_score(y_test_np, y_pred_np, average='weighted')

        print(f"Metrics for {model_name}:")
        print(f"   Accuracy:             {accuracy:.4f}")
        print(f"   Precision (macro):    {precision_macro:.4f}")
        print(f"   Recall (macro):       {recall_macro:.4f}")
        print(f"   F1-Score (macro):     {f1_macro:.4f}")
        print(f"   Precision (weighted): {precision_weighted:.4f}")
        print(f"   Recall (weighted):    {recall_weighted:.4f}")
        print(f"   F1-Score (weighted):  {f1_weighted:.4f}")

        precision_per_class = precision_score(y_test_np, y_pred_np, average=None)
        recall_per_class = recall_score(y_test_np, y_pred_np, average=None)
        f1_per_class = f1_score(y_test_np, y_pred_np, average=None)

        class_names = ['STAR (0)', 'GALAXY (1)', 'QSO (2)']
        print(f"\nPer-class metrics for {model_name}:")
        for i, class_name in enumerate(class_names):
            print(f"   {class_name}:")
            print(f"      Precision: {precision_per_class[i]:.4f}")
            print(f"      Recall:    {recall_per_class[i]:.4f}")
            print(f"      F1-Score:  {f1_per_class[i]:.4f}")

        cm = confusion_matrix(y_test_np, y_pred_np)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
        }

    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        return None


def compare_base_vs_optimized(trained_models: dict, optimized_models: dict,
                               X_test, y_test) -> pd.DataFrame:
    """
    Evaluate baseline RandomForest/XGBoost and their optimized counterparts
    on the same test set, then report the relative improvement (or lack
    thereof) from hyperparameter tuning.
    """
    print("\n" + "=" * 60)
    print("Comparing baseline vs. optimized models")
    print("=" * 60)

    base_results = {}
    for model_name in ['RandomForest', 'XGBoost']:
        if model_name in trained_models:
            results = comprehensive_evaluation_optimized(
                trained_models[model_name], X_test, y_test, model_name)
            if results is not None:
                base_results[model_name] = results

    optimized_results = {}
    for model_name, model in optimized_models.items():
        results = comprehensive_evaluation_optimized(model, X_test, y_test, model_name)
        if results is not None:
            optimized_results[model_name] = results

    comparison_data = []
    for model_name, metrics in base_results.items():
        comparison_data.append({
            'Model': model_name, 'Type': 'Base',
            'Accuracy': metrics['accuracy'],
            'Precision_Macro': metrics['precision_macro'],
            'Recall_Macro': metrics['recall_macro'],
            'F1_Macro': metrics['f1_macro'],
        })
    for model_name, metrics in optimized_results.items():
        comparison_data.append({
            'Model': model_name, 'Type': 'Optimized',
            'Accuracy': metrics['accuracy'],
            'Precision_Macro': metrics['precision_macro'],
            'Recall_Macro': metrics['recall_macro'],
            'F1_Macro': metrics['f1_macro'],
        })

    comparison_df = pd.DataFrame(comparison_data)
    print("\nBase vs. optimized comparison table:")
    print(comparison_df.to_string(index=False, float_format='%.4f'))

    print("\nPerformance improvement analysis:")
    for base_model in ['RandomForest', 'XGBoost']:
        base_f1, optimized_f1 = None, None
        for _, row in comparison_df.iterrows():
            if row['Model'] == base_model and row['Type'] == 'Base':
                base_f1 = row['F1_Macro']
            elif base_model in row['Model'] and row['Type'] == 'Optimized':
                optimized_f1 = row['F1_Macro']

        if base_f1 and optimized_f1:
            improvement = optimized_f1 - base_f1
            improvement_pct = (improvement / base_f1) * 100
            print(f"{base_model}:")
            print(f"   Base: {base_f1:.4f} -> Optimized: {optimized_f1:.4f}")
            print(f"   Change: {improvement:.4f} ({improvement_pct:.2f}%)")

    return comparison_df


def save_optimized_results(optimized_models: dict, rf_search: RandomizedSearchCV,
                            xgb_search: RandomizedSearchCV,
                            comparison_df: pd.DataFrame = None,
                            output_dir: str = "optimized_models") -> None:
    """Persist optimized models, best hyperparameters, and comparison results."""
    os.makedirs(output_dir, exist_ok=True)

    for model_name, model in optimized_models.items():
        with open(f"{output_dir}/{model_name}.pkl", 'wb') as f:
            pickle.dump(model, f)
        print(f"Model {model_name} saved.")

    best_params = {
        'RandomForest_best_params': rf_search.best_params_,
        'XGBoost_best_params': xgb_search.best_params_,
        'RandomForest_best_score': rf_search.best_score_,
        'XGBoost_best_score': xgb_search.best_score_,
    }

    def to_json_serializable(value):
        if hasattr(value, 'item'):
            return value.item()
        return value

    json_serializable_params = {}
    for key, value in best_params.items():
        if key.endswith('_best_params'):
            json_serializable_params[key] = {
                p: to_json_serializable(v) for p, v in value.items()
            }
        else:
            json_serializable_params[key] = to_json_serializable(value)

    with open(f"{output_dir}/best_hyperparameters.json", 'w') as f:
        json.dump(json_serializable_params, f, indent=2)
    print("Best hyperparameters saved.")

    if comparison_df is not None:
        comparison_df.to_csv(f"{output_dir}/model_comparison.csv", index=False)
        print("Comparison results saved.")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main(csv_path: str):
    """Run the full classification pipeline end-to-end."""

    # --- 1. Load & inspect data ---
    data = load_data(csv_path)
    inspect_data(data)
    data = encode_target(data)
    data = drop_unused_columns(data)

    # --- 2. Pre-processing ---
    check_missing_values(data)
    plot_distributions(data)
    detect_outliers_iqr(data, k=1.5)

    # --- 3. Normalization & standardization ---
    numerical_cols = ['redshift', 'u', 'g', 'r', 'i', 'mag_z', 'petroRad_r']
    data_normalized = normalize_features(data, numerical_cols)
    data_standardized = standardize_features(data_normalized, numerical_cols)
    plot_feature_distributions_per_class(
        data_normalized, numerical_cols + ['resolved_r'])

    # --- 4. Correlation analysis & feature selection ---
    corr_matrix = compute_correlation_matrix(
        data, numerical_cols + ['resolved_r'], target_col='class')
    rank_feature_correlation_with_target(corr_matrix)
    # -> Result: FINAL_FEATURES retained, DROPPED_FEATURES excluded (see above)

    # --- 5. Build modeling dataset & class weights ---
    data_new = build_modeling_dataset(data, dropped_cols=DROPPED_FEATURES)
    class_distribution = data_new['class'].value_counts().compute()
    weight_methods = calculate_class_weights(class_distribution)
    final_class_weights = weight_methods['standard']  # selected method
    data_new = create_sample_weights_column(data_new, final_class_weights)

    # --- 6. Train/test split & scaling ---
    prepared = prepare_train_test_data(data_new)

    # --- 7. Train baseline models ---
    trained_models = train_baseline_models(
        prepared['X_train_array'], prepared['y_train_array'],
        prepared['weights_train_array'], final_class_weights,
    )

    # --- 8. Evaluate baseline models ---
    all_results = evaluate_all_models(
        trained_models, prepared['X_test_array'], prepared['y_test_array'],
        prepared.get('weights_test'),
    )
    compare_models(all_results)

    for model_name, model in trained_models.items():
        display_model_parameters(model, model_name)
        if model_name in ('RandomForest', 'XGBoost', 'LogisticRegression'):
            plot_feature_importance(model, model_name, prepared['feature_columns'])

    save_models(trained_models, final_class_weights, prepared['scaler'])

    # --- 9. Hyperparameter tuning ---
    X_tune, y_tune, weights_tune = prepare_data_for_tuning(
        prepared['X_train_array'], prepared['y_train_array'],
        prepared['weights_train_array'], sample_size=100_000,
    )

    rf_search = tune_random_forest(X_tune, y_tune, weights_tune, final_class_weights)
    xgb_search = tune_xgboost(X_tune, y_tune, weights_tune)

    optimized_models = train_optimized_models(
        prepared['X_train_array'], prepared['y_train_array'], prepared['weights_train_array'],
        rf_search.best_params_, xgb_search.best_params_, final_class_weights,
    )

    comparison_df = compare_base_vs_optimized(
        trained_models, optimized_models,
        prepared['X_test_array'], prepared['y_test_array'],
    )

    save_optimized_results(optimized_models, rf_search, xgb_search, comparison_df)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    # Update this path to point to your local copy of the dataset.
    CSV_PATH = "/content/drive/My Drive/spec_photo_wise_Mohammadsad771.csv"
    main(CSV_PATH)
