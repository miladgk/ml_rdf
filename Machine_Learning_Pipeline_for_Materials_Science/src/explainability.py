"""
explainability.py

This module provides utilities for model interpretability using feature importance
and SHAP (SHapley Additive exPlanations). It supports:
- Computing permutation feature importance for scikit-learn pipelines.
- Generating bar plots of top-k important features.
- Running SHAP explainability analyses for tree-based and kernel-based models.
- Handling preprocessing pipelines to map transformed features back to their
  original names.
- Logging throughout for transparency and debugging.

Key libraries:
- scikit-learn: for permutation importance and pipeline management
- SHAP: for explainability and feature contribution visualization
- pandas, numpy: for data handling
- matplotlib: for visualization

Performance notes:
- Permutation importance uses n_jobs=-1 for full parallelism.
- SHAP background samples use numpy-based random state for reproducibility.
- Preprocessor detection handles multiple naming conventions.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def run_permutation_importance(
    trained_pipeline,
    X_test,
    y_test,
    n_repeats: int = 10,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Compute permutation feature importance for a trained pipeline.

    Parameters
    ----------
    trained_pipeline : sklearn.pipeline.Pipeline
        The fitted pipeline including preprocessing and model.
    X_test : pd.DataFrame or np.ndarray
        Test set features.
    y_test : pd.Series or np.ndarray
        Test set labels.
    n_repeats : int, default=10
        Number of random shuffles for permutation importance.
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = all cores).

    Returns
    -------
    pd.DataFrame
        DataFrame containing features, mean importance, and standard deviation,
        sorted by descending importance.
    """
    logger.info("Computing permutation importance (this can take time)...")
    result = permutation_importance(
        trained_pipeline,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=n_jobs
    )

    # Use raw feature names from X_test (permutation importance operates on raw input,
    # not on transformed features). The preprocessor's get_feature_names_out() includes
    # extra columns (e.g. MissingIndicator) which don't align with raw X_test columns.
    if hasattr(X_test, 'columns'):
        feature_names = list(X_test.columns)
    else:
        feature_names = None

    df_importance = pd.DataFrame({
        'feature': feature_names if feature_names is not None else np.arange(len(result.importances_mean)),
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    })

    df_importance = df_importance.sort_values('importance_mean', ascending=False)
    logger.info("Permutation importance computed successfully.")
    return df_importance


def plot_permutation_importance(
    df_importance: pd.DataFrame,
    top_k: int = 20,
    figsize: tuple = (8, 6)
) -> None:
    """
    Plot a horizontal bar chart of the top-k permutation feature importances.

    Parameters
    ----------
    df_importance : pd.DataFrame
        DataFrame containing features and their importance values.
    top_k : int, default=20
        Number of top features to display.
    figsize : tuple, default=(8, 6)
        Size of the figure.
    """
    df_top = df_importance.head(top_k)
    plt.figure(figsize=figsize)
    plt.barh(
        df_top['feature'].astype(str)[::-1],
        df_top['importance_mean'][::-1],
        xerr=df_top['importance_std'][::-1]
    )
    plt.xlabel("Mean Permutation Importance")
    plt.title("Top Feature Importances (Permutation)")
    plt.tight_layout()


def make_stratified_background(
    X,
    y,
    size: int = 100,
    random_state: int = 0
):
    """
    Create a stratified background sample for SHAP KernelExplainer.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature dataset.
    y : pd.Series or np.ndarray
        Labels aligned with X.
    size : int, default=100
        Desired number of rows (capped at len(X)).
    random_state : int, default=0
        Random seed.

    Returns
    -------
    pd.DataFrame or np.ndarray
        Stratified or random background sample.
    """
    size = int(min(size, len(X)))
    try:
        X_bg, _, _, _ = train_test_split(
            X,
            y,
            train_size=size,
            stratify=y,
            random_state=random_state
        )
    except Exception as exc:
        logger.warning(
            f"Stratified sampling failed ({exc}). Falling back to random sampling."
        )
        if hasattr(X, "sample"):
            X_bg = X.sample(n=size, random_state=random_state)
        else:
            rng = np.random.RandomState(random_state)
            idx = rng.choice(len(X), size=size, replace=False)
            X_bg = X[idx]
    return X_bg


def get_feature_names_from_pipeline(
    pipeline,
    fallback_names=None
) -> Optional[List[str]]:
    """
    Extract feature names from a pipeline if the preprocessor supports it.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline including preprocessing.
    fallback_names : list, optional
        Names to fall back on if extraction fails.

    Returns
    -------
    list or None
        Extracted or fallback feature names.
    """
    preprocessor = (pipeline.named_steps.get('preprocessor')
                    or pipeline.named_steps.get('preproc'))
    if preprocessor is not None:
        try:
            return list(preprocessor.get_feature_names_out())
        except Exception as exc:
            logger.warning(f"Could not extract feature names: {exc}")
            return fallback_names
    return fallback_names


def run_shap_summary(
    trained_pipeline,
    X_sample,
    y_sample=None,
    feature_names: List[str] = None,
    shap_data_dir=None,
    max_display: int = 20,
    background_size: int = 100
) -> None:
    """
    Compute and prepare a SHAP summary plot for a given model and data sample.

    Parameters
    ----------
    trained_pipeline : sklearn.pipeline.Pipeline
        Trained pipeline including preprocessing and model.
    X_sample : pd.DataFrame or np.ndarray
        Raw input data to explain.
    y_sample : pd.Series or np.ndarray, optional
        Labels aligned with X_sample (for stratified background).
    feature_names : list, optional
        Feature names corresponding to X_sample.
    shap_data_dir : str or Path, optional
        Directory to save SHAP results.
    max_display : int, default=20
        Maximum number of features to display in SHAP summary plot.
    background_size : int, default=100
        Number of rows used for background in KernelExplainer.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap is not installed. Skipping SHAP analysis.")
        return

    try:
        from pathlib import Path

        # Step 1: Transform features if preprocessor exists
        preprocessor = (trained_pipeline.named_steps.get('preprocessor')
                        or trained_pipeline.named_steps.get('preproc'))
        X_raw = X_sample
        X_trans = preprocessor.transform(X_raw) if preprocessor else (
            X_raw.values if hasattr(X_raw, 'values') else X_raw
        )
        names = get_feature_names_from_pipeline(trained_pipeline, feature_names)

        # Sanity check feature names
        if names is not None and len(names) != X_trans.shape[1]:
            logger.warning(
                f"Mismatch: {len(names)} feature names vs {X_trans.shape[1]} columns. Ignoring feature_names."
            )
            names = None

        # Ensure DataFrame format (handles numpy array or sparse matrix)
        if names is not None and not isinstance(X_trans, pd.DataFrame):
            try:
                X_trans = pd.DataFrame(X_trans, columns=list(names))
            except Exception as e:
                logger.warning(f"Could not create DataFrame with feature names: {e}. Using array format.")
                names = None

        # Step 2: Get classifier from pipeline
        clf = (trained_pipeline.named_steps.get("clf")
               or trained_pipeline.named_steps.get("classifier")
               or trained_pipeline)

        # Step 3: Choose SHAP explainer
        if "SVC" in str(type(clf)):
            logger.info("Using SHAP KernelExplainer for SVC.")

            if y_sample is not None:
                background_raw = make_stratified_background(
                    X_raw, y_sample, size=background_size, random_state=0
                )
            else:
                logger.warning(
                    "y_sample not provided for background. Falling back to random sampling."
                )
                if hasattr(X_raw, "sample"):
                    background_raw = X_raw.sample(
                        n=min(background_size, len(X_raw)), random_state=0
                    )
                else:
                    rng = np.random.RandomState(0)
                    idx = rng.choice(len(X_raw), size=min(background_size, len(X_raw)), replace=False)
                    background_raw = X_raw[idx]

            background = preprocessor.transform(background_raw) if preprocessor else background_raw
            explainer = shap.KernelExplainer(clf.predict_proba, background)
            shap_values = explainer.shap_values(X_trans)
        else:
            logger.info(f"Using SHAP Explainer for {type(clf).__name__}.")
            explainer = shap.Explainer(clf, X_trans)
            shap_values = explainer(X_trans, check_additivity=False)

        # Step 4: Handle multi-class SHAP outputs
        if isinstance(shap_values, list):
            sv = shap_values[1]
        elif hasattr(shap_values, "values") and shap_values.values.ndim == 3:
            sv = shap_values.values[:, :, 1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sv = shap_values[:, :, 1]
        elif hasattr(shap_values, "values"):
            # Explanation object with 2D values (e.g. binary TreeExplainer output)
            sv = shap_values.values
        else:
            sv = shap_values

        # Step 4b: Save SHAP outputs
        # Convert to numpy array if sv is still an Explanation or other non-array type
        if hasattr(sv, "values"):
            sv = sv.values
        if names is not None:
            shap_df = pd.DataFrame(sv, columns=names, index=getattr(X_trans, "index", None))
        else:
            shap_df = pd.DataFrame(sv, index=getattr(X_trans, "index", None))

        output_path = Path(shap_data_dir) / "shap_values.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shap_df.to_csv(output_path)
        logger.info(f"Saved SHAP values to {output_path}")

        import joblib
        joblib.dump(sv, Path(shap_data_dir) / "shap_values.pkl")
        logger.info("Saved raw SHAP object as pickle.")

        # Step 5: Create SHAP summary plot
        shap.summary_plot(
            sv,
            features=X_trans,
            feature_names=names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()

    except Exception as exc:
        logger.error(f"SHAP analysis failed: {exc}")
        raise