"""
models.py

This module provides utilities for constructing and managing machine learning pipelines 
with robust preprocessing, cross-validation, and hyperparameter optimization. 
"""

from typing import List, Dict, Any, Optional
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import RandomizedSearchCV, GroupKFold, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer

logger = logging.getLogger(__name__)


def build_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: Optional[List[str]] = None,
    add_missing_indicator: bool = False
) -> ColumnTransformer:
    """
    Build a preprocessing pipeline for numeric and categorical features.
    
    Args:
        numeric_features (List[str]): List of numeric feature column names.
        categorical_features (Optional[List[str]]): List of categorical feature column names.
        add_missing_indicator (bool): Whether to add MissingIndicator columns.

    Returns:
        ColumnTransformer: Preprocessing transformer for use in a scikit-learn pipeline.
    """
    if categorical_features is None:
        categorical_features = []

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    transformers = [('numeric', numeric_pipeline, numeric_features)]

    if add_missing_indicator:
        transformers.append(('missing', MissingIndicator(), numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('categorical', categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0)


def build_pipeline(preprocessor, classifier) -> Pipeline:
    """
    Construct a full scikit-learn pipeline.
    
    Args:
        preprocessor: A ColumnTransformer or any sklearn transformer.
        classifier: A scikit-learn compatible estimator.

    Returns:
        Pipeline: Complete pipeline combining preprocessing and classification.
    """
    return Pipeline([('preprocessor', preprocessor), ('classifier', classifier)])


def build_cached_preprocessing_pipeline(classifier) -> Pipeline:
    """
    Build a pipeline with a no-op passthrough preprocessor.
    Used when data has already been pre-transformed and cached.
    
    Args:
        classifier: A scikit-learn compatible estimator.

    Returns:
        Pipeline: A pipeline with a FunctionTransformer passthrough.
    """
    return Pipeline([
        ('preprocessor', FunctionTransformer(func=lambda X: X, validate=False)),
        ('classifier', classifier)
    ])


def randomized_search_for_pipeline(
    pipeline: Pipeline,
    param_distributions: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series] = None,
    n_iter: int = 48,
    cv_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 1,
    model_name: str = "Unknown"
) -> "RandomizedSearchCV":
    """
    Run randomized hyperparameter search with robust cross-validation.
    
    Args:
        pipeline (Pipeline): Pipeline containing preprocessing and classifier.
        param_distributions (Dict[str, Any]): Hyperparameter distributions.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        groups (Optional[pd.Series]): Group labels for group-aware CV.
        n_iter (int): Number of parameter settings to sample. Default=48.
        cv_splits (int): Number of CV splits. Default=5.
        random_state (int): Random seed. Default=42.
        n_jobs (int): Parallel jobs. Default=-1 (all available cores).
        verbose (int): Verbosity level for search. Default=1.
        model_name (str): Name of model (for logging context). Default="Unknown".

    Returns:
        RandomizedSearchCV: The fitted search object with the best pipeline.
    """
    classifier_name, classifier_estimator = pipeline.steps[-1]

    valid_classifier_params = classifier_estimator.get_params(deep=False)
    corrected_param_distributions = {}
    for param_name, param_values in param_distributions.items():
        if param_name.startswith('classifier__'):
            param_key = param_name.split('__', 1)[1]
            corrected_name = f"{classifier_name}__{param_key}"
        else:
            param_key = param_name
            corrected_name = param_name
        if param_key in valid_classifier_params:
            corrected_param_distributions[corrected_name] = param_values
        else:
            logger.warning(
                f"Skipping invalid parameter '{param_name}' for pipeline. "
                f"'{param_key}' is not valid for '{classifier_name}'."
            )

    n_unique_groups = groups.nunique() if groups is not None else 0
    n_unique_classes = y.nunique()

    if groups is not None and n_unique_groups >= cv_splits and n_unique_classes > 1:
        cv_strategy = GroupKFold(n_splits=cv_splits)
        logger.info(f"Using GroupKFold with {cv_splits} splits.")
        fit_params = {'groups': groups}
    else:
        if groups is None:
            logger.warning("No groups provided. Using StratifiedKFold.")
        elif n_unique_groups < cv_splits:
            logger.warning(
                f"[{model_name}] Unique groups ({n_unique_groups}) fewer than CV splits ({cv_splits}). "
                f"Using StratifiedKFold fallback."
            )
        cv_strategy = StratifiedKFold(n_splits=max(2, cv_splits), shuffle=True, random_state=random_state)
        logger.info(f"Using StratifiedKFold with {cv_strategy.n_splits} splits.")
        fit_params = {}

    scorer = make_scorer(f1_score, average='macro')

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=corrected_param_distributions,
        n_iter=n_iter,
        cv=cv_strategy,
        scoring=scorer,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
        return_train_score=False,
    )

    logger.info(f"Pipeline steps: {[name for name, _ in pipeline.steps]}")
    logger.info(f"Classifier: {classifier_name} ({type(classifier_estimator)})")
    logger.info(f"Parameters for search: {list(corrected_param_distributions.keys())}")
    logger.info(f"Feature count: {X.shape[1]}")
    logger.info(f"Features: {list(X.columns)}")
    if groups is not None:
        logger.info(f"Unique groups: {groups.nunique()}")
    logger.info(f"CV strategy: {cv_strategy}")

    search.fit(X, y, **fit_params)

    logger.info(f"Best pipeline: {search.best_estimator_.steps}")
    logger.info(f"Best score: {search.best_score_:.4f}")
    logger.info(f"Best params: {search.best_params_}")
    return search


def save_model(pipeline: Pipeline, out_path: str) -> None:
    """Save a trained pipeline/model to disk."""
    joblib.dump(pipeline, out_path)
    logger.info(f"Model saved to {out_path}")


def load_model(path: str) -> Pipeline:
    """Load a trained pipeline/model from disk."""
    pipeline = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return pipeline