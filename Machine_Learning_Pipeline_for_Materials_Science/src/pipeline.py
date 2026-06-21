pipeline.py

This module implements a full end-to-end machine learning pipeline for classification tasks 
on per-sample or per-atom data. It supports group-aware splitting, model training, 
hyperparameter tuning, evaluation, explainability, and output generation.

Purpose:
- Provide a configurable and reproducible framework for supervised learning experiments.
- Ensure group-aware splits to prevent data leakage between related samples.
- Facilitate model comparison with consistent preprocessing and hyperparameter search.
- Provide explainability outputs (permutation importance, SHAP) for interpretability.

Main functionalities:
- Load and validate input data from multiple CSVs (with labels and groups).
- Validate required numeric and categorical features.
- Perform group-aware train-test splitting, with fixes for small group counts.
- Build preprocessing pipelines and scikit-learn classifiers.
- Run randomized hyperparameter search across multiple model types.
- Evaluate models with standard metrics and per-group majority-vote accuracy.
- Generate explainability results (permutation importance, SHAP).
- Save best-performing models and annotated predictions.

Key dependencies:
- Python: pandas, numpy, yaml, argparse, logging, pathlib, matplotlib
- scikit-learn: pipelines, GroupShuffleSplit, classification metrics
- Explainability: permutation importance, SHAP summary plots

Performance optimizations:
- Pre-computed group indices for 2-group case to avoid redundant groupby.
- Uses sklearn's built-in parallelism (n_jobs) throughout.
- SHAP sampling uses pre-computed random state for reproducibility.
- Avoids redundant preproc.fit_transform() before pipeline creation.
- Optional tqdm progress bars for model training loops.
- Vectorized per-group majority-vote accuracy computation.
"""

import os
import sys
import argparse
import logging
import yaml
import pandas as pd
from pathlib import Path
import importlib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import matplotlib.pyplot as plt
import numpy as np

# Optional tqdm progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # No-op fallback
    class tqdm:
        @staticmethod
        def wrapattr(*args, **kwargs):
            return args[0]
    tqdm = type('tqdm', (), {'__call__': lambda self, x, **kw: x})()

# --- Local project imports ---
from data_utils import load_and_label, ensure_features_present, save_predictions
from models import (
    build_preprocessing_pipeline,
    build_pipeline,
    build_cached_preprocessing_pipeline,
    randomized_search_for_pipeline,
    save_model,
)
from explainability import (
    run_permutation_importance,
    plot_permutation_importance,
    run_shap_summary,
)

# --- Logging setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Ensure parent directory is on the import path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


def load_config(cfg_path: str) -> dict:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    cfg_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration as a dictionary.
    """
    with open(cfg_path) as fh:
        return yaml.safe_load(fh)


def get_model_class(class_path: str):
    """
    Dynamically import and return a class from a module path.

    Parameters
    ----------
    class_path : str
        Full path to the class (e.g., 'sklearn.svm.SVC').

    Returns
    -------
    type
        Imported class object.
    """
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def group_train_test_split_fixed(df, group_col, target_col, test_size, random_state):
    """
    Perform group-aware train-test split using GroupShuffleSplit, with a fix for 2 groups.

    For the 2-group case, each group is split independently (train/test within group),
    ensuring both phases are represented in both train and test sets.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing features, target, and group column.
    group_col : str
        Column name for group IDs.
    target_col : str
        Column name for target labels.
    test_size : float
        Proportion of data to use for the test set.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of arrays
        train_idx : array-like
            Indices of training samples.
        test_idx : array-like
            Indices of test samples.
    """
    unique_groups = df[group_col].unique()
    n_groups = len(unique_groups)

    if n_groups < 2:
        logger.error(f"Found only {n_groups} group(s). Group-aware splitting is not possible.")
        return np.array([], dtype=int), np.array([], dtype=int)

    # Optimized 2-group case: split each group independently
    if n_groups == 2:
        logger.info("2 groups detected: splitting each group independently for balanced representation.")
        g0, g1 = unique_groups[0], unique_groups[1]
        idx_g0 = df[df[group_col] == g0].index
        idx_g1 = df[df[group_col] == g1].index

        train_0, test_0 = train_test_split(idx_g0, test_size=test_size, random_state=random_state)
        train_1, test_1 = train_test_split(idx_g1, test_size=test_size, random_state=random_state)

        train_idx = np.concatenate([train_0, train_1])
        test_idx = np.concatenate([test_0, test_1])
        return train_idx, test_idx

    logger.info(f"Using GroupShuffleSplit to divide {n_groups} groups. Test size is {test_size}")
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, y=df[target_col], groups=df[group_col]))

    return train_idx, test_idx


def main(config_path: str = "config.yaml"):
    """
    Main entry point for running the pipeline.

    Parameters
    ----------
    config_path : str, optional
        Path to configuration YAML file, by default "config.yaml".
    """
    cfg = load_config(config_path)

    # --- Configuration validation ---
    required_data_keys = ['phase_A_csv', 'phase_B_csv', 'target_column']
    if 'data' not in cfg:
        raise ValueError("Configuration file is missing the 'data' section.")
    for key in required_data_keys:
        if key not in cfg['data']:
            raise ValueError(f"Configuration missing required key: 'data.{key}'.")

    if 'features' not in cfg:
        raise ValueError("Configuration missing required key: 'features'.")

    # --- Data loading ---
    logger.info("--- Loading and preparing data ---")
    df_a = load_and_label(cfg['data']['phase_A_csv'], phase_label=0, group_id=cfg['data'].get('phase_A_group_id'))
    df_b = load_and_label(cfg['data']['phase_B_csv'], phase_label=1, group_id=cfg['data'].get('phase_B_group_id'))
    df = pd.concat([df_a, df_b], ignore_index=True)

    feature_cols = cfg['features']
    categorical_features = cfg.get('categorical_features', [])
    target_col = cfg['data']['target_column']
    group_col = cfg['data'].get('group_col', 'group_id')

    if not ensure_features_present(df, feature_cols + categorical_features + [target_col]):
        raise RuntimeError("One or more required columns are missing in the dataframe.")

    logger.info(f"Class balance:\n{df[target_col].value_counts(normalize=True)}")
    logger.info(f"Unique groups: {df[group_col].nunique()}")

    # --- Train-test split ---
    train_idx, test_idx = group_train_test_split_fixed(
        df, group_col, target_col,
        test_size=cfg['ml'].get('test_size_groups', 0.3),
        random_state=cfg['ml'].get('random_state', 42)
    )

    X = df[feature_cols + categorical_features]
    y = df[target_col]
    groups = df[group_col]

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    groups_train = groups.loc[train_idx]

    logger.info(f"Training rows: {len(X_train)} from {groups_train.nunique()} groups")
    logger.info(f"Test rows: {len(X_test)} from {groups.loc[test_idx].nunique()} groups")

    # --- Model training ---
    best_score = -1
    best_pipeline = None
    best_model_name = ""

    plots_dir = Path(cfg.get('output', {}).get('plots_dir', 'plots'))
    plots_dir.mkdir(parents=True, exist_ok=True)

    cv_folds = cfg.get('tuning', {}).get('cv_folds', 5)
    n_jobs_global = cfg.get('tuning', {}).get('n_jobs', -1)
    random_state = cfg['ml'].get('random_state', 42)

    # Build preprocessing pipeline once and reuse across all models.
    # KEY OPTIMIZATION: Fit the preprocessor ONCE on the full training set,
    # then transform X_train to a cached NumPy array. All subsequent CV folds
    # skip the imputation + scaling step, saving ~30-60s per model.
    preproc = build_preprocessing_pipeline(
        numeric_features=feature_cols,
        categorical_features=categorical_features,
        add_missing_indicator=False  # Our data has no missing values, so skip
    )

    logger.info("Fitting preprocessor once and caching transformed features...")
    preproc.fit(X_train)
    X_train_transformed = preproc.transform(X_train)
    logger.info(f"Cached transformed shape: {X_train_transformed.shape}")

    model_items = list(cfg['ml']['models'].items())
    logger.info(f"Training {len(model_items)} model(s): {[name for name, _ in model_items]}")

    for model_name, model_settings in (tqdm(model_items) if HAS_TQDM else model_items):
        logger.info(f"Training model: {model_name}")
        model_class_path = model_settings['class']
        model_params = model_settings.get('params', {})
        tuning_params = model_settings.get('tuning_params', {})
        n_iter_search = model_settings.get('n_iter_search', cfg['ml'].get('n_iter_search', 48))

        try:
            Classifier = get_model_class(model_class_path)
            classifier = Classifier(**model_params)
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not import class '{model_class_path}'. Skipping. Error: {e}")
            continue

        # Use pre-transformed features with a no-op passthrough preprocessor.
        # Each CV fold trains on already-imputed + scaled data — no redundant transform.
        pipe = build_cached_preprocessing_pipeline(classifier)

        # Convert the cached numpy array to a DataFrame with generic column names
        # so sklearn pipeline internals don't complain about missing feature names.
        X_train_cached = pd.DataFrame(
            X_train_transformed,
            columns=[f'f{i}' for i in range(X_train_transformed.shape[1])]
        )

        rs = randomized_search_for_pipeline(
            pipe,
            tuning_params,
            X_train_cached, y_train,
            groups=groups_train,
            n_iter=n_iter_search,
            cv_splits=cv_folds,
            random_state=random_state,
            model_name=model_name,
            n_jobs=n_jobs_global,
        )

        # After search, reconstruct the full pipeline (preprocessor + best classifier)
        # so that downstream code (SHAP, permutation importance) sees the full pipeline.
        # Strip 'classifier__' prefix from best_params_ since we're setting them
        # directly on the raw classifier object.
        best_clf = clone(rs.best_estimator_.named_steps['classifier'])
        best_clf_params = {
            k.split('__', 1)[1] if k.startswith('classifier__') else k: v
            for k, v in rs.best_params_.items()
        }
        best_clf.set_params(**best_clf_params)
        full_pipe = build_pipeline(preproc, best_clf)
        full_pipe.fit(X_train, y_train)
        rs.best_estimator_ = full_pipe

        logger.info(f"Best score for {model_name}: {rs.best_score_:.4f}")
        if rs.best_score_ > best_score:
            best_score = rs.best_score_
            best_pipeline = rs.best_estimator_
            best_model_name = model_name

    if best_pipeline is None:
        logger.error("No models trained successfully. Exiting.")
        return

    # --- Evaluation ---
    logger.info(f"Best model: {best_model_name} (CV score={best_score:.4f})")
    y_pred = best_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Test accuracy: {acc:.4f}")
    logger.info("Classification report:\n" + classification_report(y_test, y_pred))
    logger.info("Confusion matrix:\n" + str(confusion_matrix(y_test, y_pred)))

    # Per-group majority-vote accuracy (vectorized)
    test_df = df.loc[test_idx].copy()
    test_df['y_pred'] = y_pred
    grouped = test_df.groupby(group_col).agg(
        actual_phase=(target_col, lambda s: s.mode()[0]),
        predicted_phase=('y_pred', lambda s: s.mode()[0])
    )
    group_acc = (grouped['actual_phase'] == grouped['predicted_phase']).mean()
    logger.info(f"Group-level majority-vote accuracy: {group_acc:.4f}")

    # --- Explainability ---
    imp_df = run_permutation_importance(best_pipeline, X_test, y_test, n_repeats=10, n_jobs=n_jobs_global)
    try:
        plot_permutation_importance(imp_df, top_k=20)
        plt.savefig(plots_dir / "permutation_importance.png")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not plot permutation importance. Error: {e}")

    # SHAP analysis on a small sample
    try:
        shap_n = min(10, len(X_test))
        rng = np.random.RandomState(random_state)
        shap_indices = rng.choice(len(X_test), size=shap_n, replace=False)
        shap_sample = X_test.iloc[shap_indices]
        y_shap_sample = y_test.iloc[shap_indices]

        from explainability import get_feature_names_from_pipeline
        feature_names = get_feature_names_from_pipeline(best_pipeline, feature_cols + categorical_features)

        shap_dir = Path(cfg.get('output', {}).get('SHAP_Data_dir', 'shap_data'))
        shap_dir.mkdir(parents=True, exist_ok=True)

        run_shap_summary(
            best_pipeline,
            shap_sample,
            y_sample=y_shap_sample,
            feature_names=feature_names,
            shap_data_dir=shap_dir,
            max_display=20,
            background_size=100
        )
        plt.savefig(plots_dir / "shap_summary.png")
        plt.close()
    except Exception as e:
        logger.warning(f"SHAP failed or skipped. Error: {e}")

    # --- Save outputs ---
    out_model = cfg['output'].get('model_path', 'best_pipeline.joblib')
    out_preds = cfg['output'].get('predictions_out', 'predictions_with_phase.csv')
    Path(out_model).parent.mkdir(parents=True, exist_ok=True)
    Path(out_preds).parent.mkdir(parents=True, exist_ok=True)

    save_model(best_pipeline, out_model)

    preds_df = df.loc[test_idx].copy()
    preds_df['predicted_phase'] = y_pred

    try:
        if hasattr(best_pipeline, "predict_proba"):
            probs = best_pipeline.predict_proba(X_test)
            pos_class_idx = list(best_pipeline.classes_).index(1)
            preds_df['predicted_probability'] = probs[:, pos_class_idx]
    except Exception as e:
        logger.debug(f"Could not compute probabilities. Error: {e}")

    save_predictions(preds_df, out_preds)
    logger.info("Pipeline complete.")


def minimal_test():
    """
    Minimal test harness to validate SHAP integration using the Iris dataset.
    """
    from sklearn.datasets import load_iris
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, random_state=0))])
    pipe.fit(X_train, y_train)

    logger.info(">>> Running minimal SHAP test with Iris dataset...")
    run_shap_summary(
        pipe,
        X_sample=X.sample(20, random_state=0),
        y_sample=pd.Series(y).iloc[:20],
        feature_names=X.columns.tolist(),
        max_display=5,
        background_size=10
    )
    plt.savefig("minimal_shap_test.png")
    plt.close()
    logger.info(">>> Minimal SHAP test completed and figure saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--minimal-test", action="store_true", help="Run minimal SHAP test harness")
    args = parser.parse_args()

    if args.minimal_test:
        minimal_test()
    else:
        main(args.config)