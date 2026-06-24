"""
feature_table_builder.py

This module processes atomic-level CSV data to construct a machine learning-ready
feature table for polyamorphous materials analysis. It identifies key interatomic
distance peaks, computes normalized peak ratios, and incorporates structural
descriptors such as Voronoi volume and coordination numbers.

Key steps:
- Load per-atom CSV data and extract relevant physical/structural features.
- Identify/validate key interatomic peaks (sqrt3, sqrt4, sqrt7, sqrt12) relative to R1.
- Compute peak-to-R1 ratios with tolerance filtering. (VECTORIZED)
- Handle missing values via per-type medians.
- Save the processed feature table to CSV.
- Update YAML config with the final feature list.

Optimization notes:
- Replaced row-by-row iterrows() with fully vectorized Pandas/NumPy operations.
- Batch JSON parsing for peak lists.
- Vectorized entropy, ratio, and interaction computations.
- Numba-accelerated peak matching for the heaviest computation.
- Grouped per-type median imputation in a single pass.
"""

import os
import re
import json
import math
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

# =============================================================================
# Numba-accelerated peak matching (optional fallback to pure NumPy)
# =============================================================================
try:
    from numba import njit

    @njit(cache=True)
    def _find_closest_peak_numba(peaks_array, expected_val, tol):
        """Numba-accelerated: find closest peak to expected_val within tolerance."""
        if len(peaks_array) == 0:
            return np.nan
        best_val = np.nan
        best_diff = np.inf
        for v in peaks_array:
            diff = abs(v - expected_val)
            if diff < best_diff:
                best_diff = diff
                best_val = v
        return best_val if best_diff <= tol * expected_val else np.nan

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def _find_closest_peak_numba(peaks_array, expected_val, tol):
        """Pure NumPy fallback if numba is not available."""
        if len(peaks_array) == 0:
            return np.nan
        diffs = np.abs(np.asarray(peaks_array) - expected_val)
        idx = np.argmin(diffs)
        return peaks_array[idx] if diffs[idx] <= tol * expected_val else np.nan


# =============================================================================
# Vectorized helpers
# =============================================================================
EXPECTED_RATIOS = {
    "sqrt3": math.sqrt(3),
    "sqrt4": math.sqrt(4),
    "sqrt7": math.sqrt(7),
    "sqrt12": math.sqrt(12),
}


def _parse_peak_column(series: pd.Series) -> pd.Series:
    """
    Vectorized (per-element) parsing of JSON-list columns.
    Returns a Series where each element is a list of floats.
    """
    def parse(val):
        if isinstance(val, list):
            return [float(v) for v in val]
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return []
        return []
    return series.apply(parse)


def _concatenate_peak_lists(series_list: List[pd.Series]) -> pd.Series:
    """
    Element-wise concatenation of multiple Series of lists.
    Returns a Series where each element is a flat list of floats.
    """
    result = series_list[0].copy()
    for s in series_list[1:]:
        result = result + s  # list concatenation per element
    return result


def _find_peak_vectorized(all_sources_series: pd.Series, expected_vals: np.ndarray, tol: float) -> np.ndarray:
    """
    Find the closest peak for each row within tolerance.

    Parameters
    ----------
    all_sources_series : pd.Series of lists
        Each element is a list of candidate peak values for that row.
    expected_vals : np.ndarray
        Expected target values per row (e.g., R1 * sqrt(3)).
    tol : float
        Relative tolerance.

    Returns
    -------
    np.ndarray
        Best matching peak for each row, or NaN if none within tolerance.
    """
    n = len(all_sources_series)
    result = np.full(n, np.nan)

    for i in range(n):
        sources = all_sources_series.iloc[i]
        if not sources or np.isnan(expected_vals[i]) or expected_vals[i] <= 0:
            continue
        result[i] = _find_closest_peak_numba(
            np.asarray(sources, dtype=np.float64),
            expected_vals[i],
            tol
        )
    return result


def _calc_entropy_vectorized(row_vals: np.ndarray) -> np.ndarray:
    """
    Compute row-wise entropy for neighbor columns in a vectorized manner.
    row_vals: (n_rows, n_neighbor_cols) array.
    Returns (n_rows,) array of entropy values.
    """
    total = row_vals.sum(axis=1, keepdims=True)
    # Avoid division by zero
    total_safe = np.where(total == 0, 1, total)
    probs = row_vals / total_safe
    # Zero probability -> zero contribution
    entropy = -np.sum(np.where(probs > 0, probs * np.log(probs), 0), axis=1)
    # Where total was zero, entropy is zero
    entropy = np.where(total.squeeze() == 0, 0, entropy)
    return entropy


# =============================================================================
# Main builder
# =============================================================================
def build_ml_table(input_csv: str, tolerance: float = 0.1, atomic_radii: dict = None) -> pd.DataFrame:
    """Build a machine learning-ready feature table from atomic CSV input.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV file with per-atom data.
    tolerance : float
        Relative tolerance for peak matching.
    atomic_radii : dict, optional
        Dictionary mapping atom type to radius in angstroms.
        E.g., {1: 1.28, 2: 1.60}. If provided, free volume and
        packing fraction features are computed.
    """
    df = pd.read_csv(input_csv)

    # Detect neighbor columns dynamically
    neighbor_cols = [col for col in df.columns if col.startswith("neighbor_")]
    logging.info(f"Detected {len(neighbor_cols)} neighbor columns: {neighbor_cols}")

    # ---------------------------------------------------------------
    # 1. Extract scalar structural descriptors (vectorized)
    # ---------------------------------------------------------------
    R1 = df.get("R1_SG", np.nan).astype(float)
    vor_vol = df.get("voronoi_volume_temporal", np.nan).astype(float)
    cn = df.get("CN_temporal", np.nan).astype(float)
    q4 = df.get("q4", np.nan).astype(float)
    q6 = df.get("q6", np.nan).astype(float)
    w4 = df.get("w4", np.nan).astype(float)
    w6 = df.get("w6", np.nan).astype(float)
    q4_avg = df.get("q4_avg", np.nan).astype(float)
    q6_avg = df.get("q6_avg", np.nan).astype(float)
    n3 = df.get("n3_temporal", np.nan).astype(float)
    n4 = df.get("n4_temporal", np.nan).astype(float)
    n5 = df.get("n5_temporal", np.nan).astype(float)
    n6 = df.get("n6_temporal", np.nan).astype(float)
    pentagon_frac = df.get("pentagon_fraction_temporal", np.nan).astype(float)
    asphericity = df.get("asphericity_temporal", np.nan).astype(float)
    asphericity_std = df.get("asphericity_std_temporal", np.nan).astype(float)
    s2_entropy = df.get("s2_entropy_temporal", np.nan).astype(float)
    s2_entropy_avg = df.get("s2_entropy_avg_temporal", np.nan).astype(float)
    isb = df.get("isb_temporal", np.nan).astype(float)
    csro_unlike = df.get("csro_unlike_temporal", np.nan).astype(float)
    csro_like = df.get("csro_like_temporal", np.nan).astype(float)
    csro_unlike_std = df.get("csro_unlike_std_temporal", np.nan).astype(float)
    gaussian_peak2 = df.get("gaussian_peak2_center", np.nan).astype(float)
    gaussian_peak3 = df.get("gaussian_peak3_center", np.nan).astype(float)
    validated_sqrt3 = df.get("validated_sqrt3_r_SG", np.nan).astype(float)
    validated_sqrt4 = df.get("validated_sqrt4_r_SG", np.nan).astype(float)
    validated_sqrt7 = df.get("validated_sqrt7_r_SG", np.nan).astype(float)
    validated_sqrt12 = df.get("validated_sqrt12_r_SG", np.nan).astype(float)

    # ---------------------------------------------------------------
    # 2. Parse peak-list columns (vectorized apply - unavoidable for JSON)
    # ---------------------------------------------------------------
    logging.info("Parsing peak-list columns...")
    global_peaks = _parse_peak_column(df.get("global_peak_r_values_SG", pd.Series([[]]*len(df))))
    targeted_peaks = _parse_peak_column(df.get("targeted_peak_r_second_third", pd.Series([[]]*len(df))))

    # Filter peaks: keep only those > R1
    def filter_peaks_by_r1(peaks_list, r1_val):
        """Keep only peaks > R1. Returns empty list if R1 is invalid."""
        if not isinstance(peaks_list, list) or np.isnan(r1_val) or r1_val <= 0:
            return []
        return [p for p in peaks_list if p > r1_val]

    global_peaks_filtered = pd.Series([
        filter_peaks_by_r1(lst, r1)
        for lst, r1 in zip(global_peaks, R1)
    ], index=df.index)

    targeted_peaks_filtered = pd.Series([
        filter_peaks_by_r1(lst, r1)
        for lst, r1 in zip(targeted_peaks, R1)
    ], index=df.index)

    # ---------------------------------------------------------------
    # 3. Build all-source peak lists per sqrt type (vectorized per-row)
    # ---------------------------------------------------------------
    logging.info("Building candidate peak lists for each sqrt peak...")
    sqrt3_sources = _concatenate_peak_lists([
        global_peaks_filtered, targeted_peaks_filtered,
        gaussian_peak2.apply(lambda v: [v] if not np.isnan(v) else []),
        validated_sqrt3.apply(lambda v: [v] if not np.isnan(v) else []),
    ])
    sqrt4_sources = _concatenate_peak_lists([
        global_peaks_filtered, targeted_peaks_filtered,
        gaussian_peak3.apply(lambda v: [v] if not np.isnan(v) else []),
        validated_sqrt4.apply(lambda v: [v] if not np.isnan(v) else []),
    ])
    sqrt7_sources = _concatenate_peak_lists([
        global_peaks_filtered,
        validated_sqrt7.apply(lambda v: [v] if not np.isnan(v) else []),
    ])
    sqrt12_sources = _concatenate_peak_lists([
        global_peaks_filtered,
        validated_sqrt12.apply(lambda v: [v] if not np.isnan(v) else []),
    ])

    # ---------------------------------------------------------------
    # 4. Peak matching (vectorized + optional numba)
    # ---------------------------------------------------------------
    logging.info("Matching peaks (vectorized, numba-accelerated if available)...")
    exp3 = R1.values * math.sqrt(3)
    exp4 = R1.values * math.sqrt(4)
    exp7 = R1.values * math.sqrt(7)
    exp12 = R1.values * math.sqrt(12)

    sqrt3_peak = _find_peak_vectorized(sqrt3_sources, exp3, tolerance)
    sqrt4_peak = _find_peak_vectorized(sqrt4_sources, exp4, tolerance)
    sqrt7_peak = _find_peak_vectorized(sqrt7_sources, exp7, tolerance)
    sqrt12_peak = _find_peak_vectorized(sqrt12_sources, exp12, tolerance)

    # Ratios (vectorized)
    R1_safe = np.where(R1.values > 0, R1.values, np.nan)
    sqrt3_ratio = sqrt3_peak / R1_safe
    sqrt4_ratio = sqrt4_peak / R1_safe
    sqrt7_ratio = sqrt7_peak / R1_safe
    sqrt12_ratio = sqrt12_peak / R1_safe

    # ---------------------------------------------------------------
    # 5. Assemble base feature DataFrame (fully vectorized)
    # ---------------------------------------------------------------
    ml_df = pd.DataFrame({
        "id": df["id"],
        "type": df["type"],
        "R1": R1,
        "sqrt3_peak": sqrt3_peak,
        "sqrt4_peak": sqrt4_peak,
        "sqrt7_peak": sqrt7_peak,
        "sqrt12_peak": sqrt12_peak,
        "sqrt3_ratio": sqrt3_ratio,
        "sqrt4_ratio": sqrt4_ratio,
        "sqrt7_ratio": sqrt7_ratio,
        "sqrt12_ratio": sqrt12_ratio,
        "voronoi_volume_temporal": vor_vol,
        "CN_temporal": cn,
        "q4": q4,
        "q6": q6,
        "w4": w4,
        "w6": w6,
        "q4_avg": q4_avg,
        "q6_avg": q6_avg,
        "n3_temporal": n3,
        "n4_temporal": n4,
        "n5_temporal": n5,
        "n6_temporal": n6,
        "pentagon_fraction_temporal": pentagon_frac,
        "asphericity_temporal": asphericity,
        "asphericity_std_temporal": asphericity_std,
        "s2_entropy_temporal": s2_entropy,
        "s2_entropy_avg_temporal": s2_entropy_avg,
        "isb_temporal": isb,
        "csro_unlike_temporal": csro_unlike,
        "csro_like_temporal": csro_like,
        "csro_unlike_std_temporal": csro_unlike_std,
    })

    # Add neighbor columns
    for col in neighbor_cols:
        ml_df[col] = df[col].astype(float)

    # ---------------------------------------------------------------
    # 6. Fill missing with per-type medians (vectorized groupby)
    # ---------------------------------------------------------------
    fill_cols = [
        "R1", "sqrt3_peak", "sqrt4_peak", "sqrt7_peak", "sqrt12_peak",
        "sqrt3_ratio", "sqrt4_ratio", "sqrt7_ratio", "sqrt12_ratio",
        "voronoi_volume_temporal", "CN_temporal", "q4", "q6", "w4", "w6",
        "q4_avg", "q6_avg",
        "n3_temporal", "n4_temporal", "n5_temporal", "n6_temporal",
        "pentagon_fraction_temporal", "asphericity_temporal", "asphericity_std_temporal",
    ]
    ml_df[fill_cols] = ml_df.groupby("type")[fill_cols].transform(lambda x: x.fillna(x.median()))

    # ---------------------------------------------------------------
    # 7. Derived features (all vectorized)
    # ---------------------------------------------------------------
    ml_df["R3_minus_R1"] = ml_df["sqrt3_peak"] - ml_df["R1"]
    ml_df["R4_minus_R1"] = ml_df["sqrt4_peak"] - ml_df["R1"]
    ml_df["R7_minus_R1"] = ml_df["sqrt7_peak"] - ml_df["R1"]
    ml_df["R12_minus_R1"] = ml_df["sqrt12_peak"] - ml_df["R1"]
    ml_df["R4_minus_R3"] = ml_df["sqrt4_peak"] - ml_df["sqrt3_peak"]
    ml_df["R7_minus_R4"] = ml_df["sqrt7_peak"] - ml_df["sqrt4_peak"]
    ml_df["R12_minus_R7"] = ml_df["sqrt12_peak"] - ml_df["sqrt7_peak"]

    # ---------------------------------------------------------------
    # 8. Row entropy (fully vectorized)
    # ---------------------------------------------------------------
    if neighbor_cols:
        neighbor_array = ml_df[neighbor_cols].fillna(0).astype(float).values
        ml_df["entropy"] = _calc_entropy_vectorized(neighbor_array)
    else:
        ml_df["entropy"] = 0.0

    # ---------------------------------------------------------------
    # 9. Ratios and interactions (all vectorized)
    # ---------------------------------------------------------------
    ml_df["q4_divide_q6"] = ml_df["q4"] / ml_df["q6"].replace(0, np.nan)
    ml_df["q4_power_2"] = ml_df["q4"] ** 2
    ml_df["q6_power_2"] = ml_df["q6"] ** 2
    ml_df["R1_time_CN_temporal"] = ml_df["R1"] * ml_df["CN_temporal"]
    ml_df["R3_minus_R1_time_q4"] = ml_df["R3_minus_R1"] * ml_df["q4"]
    ml_df["sqrt3_ratio_divide_sqrt4_ratio"] = ml_df["sqrt3_ratio"] / ml_df["sqrt4_ratio"].replace(0, np.nan)
    ml_df["CN_time_q4"] = ml_df["CN_temporal"] * ml_df["q4"]
    ml_df["CN_time_q6"] = ml_df["CN_temporal"] * ml_df["q6"]
    ml_df["entropy_time_q4"] = ml_df["entropy"] * ml_df["q4"]
    ml_df["entropy_time_q6"] = ml_df["entropy"] * ml_df["q6"]

    ml_df["CN_density"] = ml_df["CN_temporal"] / ml_df["R1"].replace(0, np.nan)
    ml_df["log_CN"] = np.log1p(ml_df["CN_temporal"])
    ml_df["log_voronoi"] = np.log1p(ml_df["voronoi_volume_temporal"])

    for col in neighbor_cols:
        ml_df[f"{col}_ratio"] = ml_df[col] / ml_df["CN_temporal"].replace(0, np.nan)

    # ---------------------------------------------------------------
    # 10. Free volume, packing fraction, and composition features
    # ---------------------------------------------------------------
    if atomic_radii is not None:
        # Map atom type to radius from config dictionary
        ml_df['atomic_radius'] = ml_df['type'].map({int(k): float(v) for k, v in atomic_radii.items()})
        ml_df['atomic_sphere_volume'] = (4.0 / 3.0) * np.pi * (ml_df['atomic_radius'] ** 3)

        # Free volume: Voronoi volume minus the atomic sphere volume
        ml_df['free_volume_temporal'] = ml_df['voronoi_volume_temporal'] - ml_df['atomic_sphere_volume']

        # Composition fractions: neighbor_1/CN and neighbor_2/CN
        cn_safe = ml_df["CN_temporal"].replace(0, np.nan)
        ml_df['neighbor_1_fraction_temporal'] = ml_df['neighbor_1'] / cn_safe
        ml_df['neighbor_2_fraction_temporal'] = ml_df['neighbor_2'] / cn_safe
        if ml_df['neighbor_1_fraction_temporal'].isna().any():
            logging.warning(f"Found {ml_df['neighbor_1_fraction_temporal'].isna().sum()} atoms with CN_temporal == 0")

        # Interaction features
        ml_df['volume_q6_interaction'] = ml_df['voronoi_volume_temporal'] * ml_df['q6']
        ml_df['volume_per_neighbor'] = ml_df['voronoi_volume_temporal'] / cn_safe

        # Add to fill_cols for per-type median imputation
        extra_fill = ['free_volume_temporal',
                      'neighbor_1_fraction_temporal', 'volume_q6_interaction',
                      'volume_per_neighbor']
        fill_cols.extend(extra_fill)
        ml_df[extra_fill] = ml_df.groupby("type")[extra_fill].transform(lambda x: x.fillna(x.median()))

    return ml_df


# === Build and save ML table ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Load atomic_radii from config if available
    import yaml
    _atomic_radii = None
    try:
        with open("config.yaml", "r") as f:
            _cfg = yaml.safe_load(f)
        _atomic_radii = _cfg.get("atomic_radii")
        if _atomic_radii:
            logging.info(f"Loaded atomic_radii from config: {_atomic_radii}")
    except Exception as e:
        logging.warning(f"Could not load atomic_radii from config: {e}")

    ml_df = build_ml_table(
        "../Atomic_Simulation_Post-Processing_Pipeline/outputs/features_polyamorphous.csv",
        tolerance=0.2,
        atomic_radii=_atomic_radii
    )
    os.makedirs("data", exist_ok=True)
    ml_df.to_csv("data/features_polyamorphous.csv", index=False)

    # Sanity check
    df = pd.read_csv("data/features_polyamorphous.csv")
    print(df.isna().sum())

    # === Update config.yaml with feature names ===
    exclude_cols = ["id", "type", "total_neighbors"]
    feature_names = [col for col in ml_df.columns if col not in exclude_cols]

    config_file = "config.yaml"
    with open(config_file, "r") as f:
        lines = f.readlines()

    new_lines = []
    features_written = False

    for line in lines:
        if re.match(r'^\s*features\s*:', line) and not features_written:
            indent = re.match(r'^(\s*)features', line).group(1)
            inline_features = ", ".join(feature_names)
            new_lines.append(f"{indent}features: [{inline_features}]\n")
            features_written = True
            continue

        if features_written and re.match(r'^\s*-\s*\S+', line):
            continue

        new_lines.append(line)

    with open(config_file, "w") as f:
        f.writelines(new_lines)