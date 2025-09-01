"""
data_utils.py

This module provides utility functions for preparing tabular data in machine learning workflows, 
with emphasis on group-aware data management to prevent leakage between training and testing. 
It is tailored for use in per-atom or grouped datasets, where ensuring group integrity during 
splitting is critical.

Main functionalities:
- Load CSV files and attach consistent phase labels and optional group identifiers.
- Validate that required feature columns are present in a DataFrame.
- Perform group-aware stratified train/test splits while preventing leakage.
- Save prediction results to CSV files with logging support.

This module integrates with larger machine learning pipelines that require reliable preprocessing 
and data partitioning, particularly in contexts like materials informatics, physics-informed ML, 
and other scientific applications.
"""

from pathlib import Path
import pandas as pd
from typing import Tuple, List, Optional, Union
from sklearn.model_selection import train_test_split
import logging

# Configure module-specific logger
logger = logging.getLogger(__name__)


def load_and_label(
    csv_path: Union[str, Path, List[Union[str, Path]]],
    phase_label: int,
    group_id: Optional[int] = None
) -> pd.DataFrame:
    """
    Load one or more CSV files, attach a phase label, and assign (or generate) a group identifier.

    Args:
        csv_path (str | Path | list[str | Path]): Path(s) to one or more CSV files.
        phase_label (int): Integer label for the phase (e.g., 0 or 1).
        group_id (Optional[int]): Optional integer identifier for the group. If None, a 
            deterministic ID is generated from the filename(s) hash.

    Returns:
        pd.DataFrame: Concatenated DataFrame(s) with added 'phase_label' and 'group_id' columns.
    """
    # Normalize to list of Paths
    if isinstance(csv_path, (str, Path)):
        paths = [Path(csv_path)]
    elif isinstance(csv_path, list):
        paths = [Path(p) for p in csv_path]
    else:
        raise TypeError(f"csv_path must be str, Path, or list[str|Path], got {type(csv_path)}")

    dfs = []
    for path in paths:
        df = pd.read_csv(path).copy()
        df["phase_label"] = int(phase_label)

        # Assign group ID: provided or derived from hash of path
        gid = group_id if group_id is not None else abs(hash(str(path))) % (10**8)
        df["group_id"] = int(gid)

        logger.info(
            f"Loaded {len(df)} rows from {path.name}; "
            f"assigned phase_label={phase_label}, group_id={gid}"
        )
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]


def ensure_features_present(df: pd.DataFrame, feature_cols: List[str]) -> bool:
    """
    Check that all required feature columns exist in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        feature_cols (List[str]): List of expected feature column names.

    Returns:
        bool: True if all features are present, False otherwise (with warning logged).
    """
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        logger.warning(f"Missing expected features: {missing_features}")
        return False
    return True


def group_train_test_split(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[pd.Index, pd.Index]:
    """
    Perform a group-aware stratified train/test split.

    Ensures that entire groups are assigned to either the training or testing set, 
    and that the split is stratified by the majority class within each group.

    Args:
        df (pd.DataFrame): Full dataset containing groups and target labels.
        group_col (str): Column name representing group membership.
        target_col (str): Column name of the target labels.
        test_size (float): Fraction of groups to assign to the test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.Index, pd.Index]: Row indices for the training and testing sets.

    Raises:
        ValueError: If fewer than two distinct groups are present.
    """
    # Aggregate groups with their majority class label
    groups_df = df.groupby(group_col)[target_col].agg(lambda s: s.mode().iat[0]).reset_index()

    if groups_df.shape[0] < 2:
        raise ValueError("Need at least two distinct groups for group-aware split.")

    # Stratified split on majority class per group
    train_groups, test_groups = train_test_split(
        groups_df[group_col],
        test_size=test_size,
        stratify=groups_df[target_col],
        random_state=random_state
    )

    train_idx = df[df[group_col].isin(train_groups)].index
    test_idx = df[df[group_col].isin(test_groups)].index

    logger.info(
        f"Groups: total={groups_df.shape[0]}, "
        f"train_groups={train_groups.size}, test_groups={test_groups.size}"
    )
    return train_idx, test_idx


def save_predictions(df: pd.DataFrame, out_csv: str) -> None:
    """
    Save predictions DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame containing predictions.
        out_csv (str): Output CSV file path.

    Returns:
        None
    """
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved predictions to {out_csv}")
