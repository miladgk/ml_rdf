"""
apply_model_to_unlabeled.py

This module applies a pre-trained scikit-learn machine learning pipeline to an unlabeled 
per-atom CSV dataset (e.g., from a polyamorphous metallic glass sample) and produces an 
annotated CSV with predicted phase labels and associated class probabilities.

Main functionality:
- Load a trained scikit-learn pipeline (joblib format).
- Validate and prepare input features from the unlabeled CSV.
- Predict phase labels using the pipeline.
- Compute class probabilities via `predict_proba` or by converting decision function scores.
- Append predictions and probabilities to the original dataset.
- Save results to an output CSV file.
- Support configuration through YAML files and/or CLI arguments.
- Automatically infer feature column names from the pipeline if not explicitly provided.

This script is typically used in a larger data-analysis pipeline for identifying polyamorphic 
phases in metallic glasses, after training machine learning models on labeled datasets.
"""

import argparse
import logging
import joblib
import yaml
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List, Optional

# Configure logging for informative runtime output
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path: Optional[str]):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path : str or None
        Path to YAML config file. If None, returns empty dict.

    Returns
    -------
    dict
        Configuration dictionary (may be empty).
    """
    if config_path is None:
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_classifier_step(pipeline):
    """
    Locate the classifier object inside a scikit-learn pipeline.

    Parameters
    ----------
    pipeline : sklearn.Pipeline or estimator
        The trained pipeline or estimator.

    Returns
    -------
    (classifier, step_name) : tuple
        Classifier object and its step name (None if not found).
    """
    if hasattr(pipeline, "named_steps"):
        # Try common step names
        for candidate in ("classifier", "clf", "model", "estimator"):
            if candidate in pipeline.named_steps:
                return pipeline.named_steps[candidate], candidate
        # Fallback: use last step
        names = list(pipeline.named_steps.keys())
        if names:
            last = names[-1]
            return pipeline.named_steps[last], last
    return pipeline, None


def ensure_feature_columns(df: pd.DataFrame, required: List[str], fill_value=np.nan):
    """
    Ensure that DataFrame contains all required feature columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    required : list of str
        List of required feature column names.
    fill_value : scalar, default np.nan
        Value used to fill missing columns.

    Returns
    -------
    (df_out, used_features) : (pandas.DataFrame, list of str)
        DataFrame with missing features added, and list of ordered required features.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        logging.warning(
            f"Input data is missing {len(missing)} required feature columns. "
            f"These will be filled with {fill_value!s}: {missing}"
        )
        for c in missing:
            df[c] = fill_value

    extra = [c for c in df.columns if c not in required]
    if extra:
        logging.info(
            f"Input data contains {len(extra)} extra columns (these will be preserved): "
            f"{extra[:10]}{'...' if len(extra) > 10 else ''}"
        )

    ordered_cols = required + [c for c in df.columns if c not in required]
    return df[ordered_cols], required


def softmax(z):
    """
    Compute numerically stable softmax.

    Parameters
    ----------
    z : array-like
        Input scores.

    Returns
    -------
    numpy.ndarray
        Softmax probabilities.
    """
    z = np.array(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    z_max = np.max(z, axis=1, keepdims=True)
    exp = np.exp(z - z_max)
    return exp / np.sum(exp, axis=1, keepdims=True)


def sigmoid(x):
    """
    Compute the sigmoid function.

    Parameters
    ----------
    x : array-like
        Input values.

    Returns
    -------
    numpy.ndarray
        Sigmoid-transformed values.
    """
    x = np.array(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def compute_probabilities_from_decision_fn(decision_scores, classes):
    """
    Convert decision_function scores into probabilities.

    Parameters
    ----------
    decision_scores : array-like
        Scores from a decision function.
    classes : list or None
        Class labels (if available).

    Returns
    -------
    numpy.ndarray
        Probability estimates with shape (n_samples, n_classes).
    """
    scores = np.array(decision_scores)
    if scores.ndim == 1:
        prob_pos = sigmoid(scores)
        prob_neg = 1.0 - prob_pos
        return np.vstack([prob_neg, prob_pos]).T
    else:
        return softmax(scores)


def apply_model_to_unlabeled(
    model_path: str,
    unlabeled_csv: str,
    output_csv: str,
    feature_list: Optional[List[str]] = None,
    class_prob_prefix: str = "prob_class_",
):
    """
    Apply a trained pipeline to an unlabeled dataset.

    Parameters
    ----------
    model_path : str
        Path to saved joblib pipeline.
    unlabeled_csv : str
        Path to unlabeled CSV file.
    output_csv : str
        Path to save annotated output CSV.
    feature_list : list of str, optional
        List of feature column names. If None, attempts to infer from pipeline.
    class_prob_prefix : str, default "prob_class_"
        Prefix for probability columns in output.

    Returns
    -------
    pandas.DataFrame
        Annotated DataFrame including predictions and probabilities.
    """
    model_path = Path(model_path)
    assert model_path.exists(), f"Model file not found: {model_path}"

    logging.info(f"Loading pipeline from {model_path}")
    pipeline = joblib.load(model_path)

    logging.info(f"Loading unlabeled data from {unlabeled_csv}")
    df = pd.read_csv(unlabeled_csv)
    logging.info(
        f"Loaded dataframe with {len(df)} rows and columns: "
        f"{list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}"
    )

    # Determine required features
    if feature_list is None:
        inferred = None
        try:
            if hasattr(pipeline, "named_steps") and "preproc" in pipeline.named_steps:
                preproc = pipeline.named_steps["preproc"]
                if hasattr(preproc, "transformers_"):
                    cols = []
                    for _, _, cols_in in preproc.transformers_:
                        if isinstance(cols_in, (list, tuple)):
                            cols.extend(list(cols_in))
                    if cols:
                        inferred = cols
            if inferred is None and hasattr(pipeline, "feature_names_in_"):
                inferred = list(pipeline.feature_names_in_)
        except Exception as e:
            logging.debug(f"Could not infer features from pipeline: {e}")
            inferred = None

        if inferred:
            logging.info(f"Inferred feature list from pipeline with {len(inferred)} features.")
            feature_list = inferred
        else:
            raise RuntimeError(
                "Feature list not provided and could not be inferred from the pipeline. "
                "Pass --feature-list or include in config."
            )

    # Ensure features exist
    df_ordered, use_features = ensure_feature_columns(df.copy(), feature_list, fill_value=np.nan)
    X = df_ordered[use_features].copy()
    logging.info(f"Preparing input array of shape {X.shape} for prediction.")

    # Predict labels
    try:
        preds = pipeline.predict(X)
    except Exception as e:
        logging.error(f"pipeline.predict failed: {e}")
        raise

    # Predict probabilities
    probs = None
    try:
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X)
            logging.info("Used pipeline.predict_proba() to compute probabilities.")
        else:
            classifier, _ = find_classifier_step(pipeline)
            if hasattr(pipeline, "decision_function"):
                dec = pipeline.decision_function(X)
                probs = compute_probabilities_from_decision_fn(
                    dec, getattr(classifier, "classes_", None)
                )
                logging.info("Used pipeline.decision_function() for probabilities.")
            else:
                X_trans = X
                if hasattr(pipeline, "named_steps") and "preproc" in pipeline.named_steps:
                    try:
                        X_trans = pipeline.named_steps["preproc"].transform(X)
                    except Exception as ex:
                        logging.warning(f"Failed preprocessing: {ex}. Trying raw X.")
                classifier, _ = find_classifier_step(pipeline)
                if hasattr(classifier, "decision_function"):
                    dec = classifier.decision_function(X_trans)
                    probs = compute_probabilities_from_decision_fn(
                        dec, getattr(classifier, "classes_", None)
                    )
                    logging.info("Used classifier.decision_function() for probabilities.")
                else:
                    logging.warning("No probability methods available. Using NaN.")
    except Exception as e:
        logging.warning(f"Failed to compute probabilities: {e}")
        probs = None

    # Collect classes if available
    classes = None
    try:
        classifier, _ = find_classifier_step(pipeline)
        if hasattr(classifier, "classes_"):
            classes = list(classifier.classes_)
    except Exception:
        classes = None

    # Build output DataFrame
    df_out = df.copy()
    df_out["predicted_phase"] = preds

    if probs is not None:
        probs_arr = np.array(probs)
        if probs_arr.ndim == 1:
            probs_arr = probs_arr.reshape(-1, 1)

        if classes is not None and probs_arr.shape[1] == len(classes):
            prob_df = pd.DataFrame(
                probs_arr,
                columns=[f"{class_prob_prefix}{str(cls)}" for cls in classes],
                index=df_out.index,
            )
            df_out = pd.concat([df_out, prob_df], axis=1)

            cls_to_idx = {cls: i for i, cls in enumerate(classes)}
            pred_indices = [cls_to_idx.get(p, None) for p in preds]

            pred_probs = []
            for row_i, idx in enumerate(pred_indices):
                if idx is None:
                    pred_probs.append(np.nan)
                else:
                    try:
                        pred_probs.append(probs_arr[row_i, idx])
                    except Exception:
                        pred_probs.append(np.nan)
            df_out["predicted_probability"] = pred_probs
        else:
            max_probs = np.max(probs_arr, axis=1)
            df_out["predicted_probability"] = max_probs
            for j in range(probs_arr.shape[1]):
                df_out[f"{class_prob_prefix}{j}"] = probs_arr[:, j]
    else:
        logging.warning("No probability estimates available. Filling with NaN.")
        df_out["predicted_probability"] = np.nan

    # Save output
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    logging.info(f"Saved predictions to {out_path} (rows: {len(df_out)})")

    return df_out


def main():
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(description="Apply trained pipeline to unlabeled polyamorphous CSV.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config (optional).")
    parser.add_argument("--model-path", type=str, default=None, help="Path to saved joblib pipeline (overrides config).")
    parser.add_argument("--input-csv", type=str, default=None, help="Path to unlabeled CSV (overrides config).")
    parser.add_argument("--output-csv", type=str, default=None, help="Path to annotated CSV output (overrides config).")
    parser.add_argument(
        "--feature-list",
        type=str,
        default=None,
        help="Comma-separated list of feature columns. If omitted, will attempt inference.",
    )
    args = parser.parse_args()

    cfg = {}
    if args.config and os.path.exists(args.config):
        logging.info(f"Loading config from {args.config}")
        cfg = load_config(args.config) or {}

    # Resolve paths
    model_path = args.model_path or cfg.get("ml", {}).get("output_model_path") \
        or cfg.get("output_model_path") or cfg.get("model_path") \
        or cfg.get("output", {}).get("model_path")
    if model_path is None:
        raise RuntimeError("Model path not provided. Pass --model-path or add to config.")

    input_csv = args.input_csv or cfg.get("data", {}).get("unlabeled_csv") \
        or cfg.get("data", {}).get("polyamorphous_features.csv") \
        or cfg.get("data", {}).get("polyamorphous")
    if input_csv is None:
        raise RuntimeError("Input CSV not provided. Pass --input-csv or add to config.")

    output_csv = args.output_csv or cfg.get("output", {}).get("predicted_csv") \
        or Path("outputs/predicted_polyamorphous.csv").as_posix()

    # Resolve features
    feature_list = None
    if args.feature_list:
        feature_list = [s.strip() for s in args.feature_list.split(",") if s.strip()]
    elif cfg.get("features"):
        feature_list = list(cfg.get("features"))
    elif cfg.get("feature_columns"):
        feature_list = list(cfg.get("feature_columns"))

    df_out = apply_model_to_unlabeled(
        model_path=model_path,
        unlabeled_csv=input_csv,
        output_csv=output_csv,
        feature_list=feature_list,
    )
    # logging.info("Done. Example of annotated output:")
    # logging.info(df_out.head().to_string(index=False))


if __name__ == "__main__":
    main()
