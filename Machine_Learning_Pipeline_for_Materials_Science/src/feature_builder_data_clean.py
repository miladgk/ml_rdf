"""
feature_table_builder.py

This module processes atomic-level CSV data to construct a machine learning-ready
feature table for polyamorphous materials analysis. It identifies key interatomic
distance peaks, computes normalized peak ratios, and incorporates structural
descriptors such as Voronoi volume and coordination numbers.

Key steps:
- Load per-atom CSV data and extract relevant physical/structural features.
- Identify/validate key interatomic peaks (sqrt3, sqrt4, sqrt7, sqrt12) relative to R1.
- Compute peak-to-R1 ratios with tolerance filtering.
- Handle missing values via per-type medians.
- Save the processed feature table to CSV.
- Update YAML config with the final feature list.
"""

import os
import re
import json
import math
import logging
import numpy as np
import pandas as pd

# Expected interatomic ratios
EXPECTED_RATIOS = {
    "sqrt3": math.sqrt(3),   # Peak 2
    "sqrt4": math.sqrt(4),   # Peak 3
    "sqrt7": math.sqrt(7),   # Peak 4
    "sqrt12": math.sqrt(12), # Peak 5
}


def build_ml_table(input_csv: str, tolerance: float = 0.1) -> pd.DataFrame:
    """Build a machine learning-ready feature table from atomic CSV input."""
    df = pd.read_csv(input_csv)

    # Detect neighbor columns dynamically
    neighbor_cols = [col for col in df.columns if col.startswith("neighbor_")]
    features = []

    for _, row in df.iterrows():
        try:
            atom_id = row["id"]
            atom_type = row["type"]

            # Structural descriptors
            vor_vol = row.get("voronoi_volume_temporal", np.nan)
            cn = row.get("CN_temporal", np.nan)
            q4 = row.get("q4", np.nan)
            q6 = row.get("q6", np.nan)
            R1 = row.get("R1_SG", np.nan)

            # Peak sources
            gaussian_peak2 = row.get("gaussian_peak2_center", np.nan)
            gaussian_peak3 = row.get("gaussian_peak3_center", np.nan)
            validated_sqrt3 = row.get("validated_sqrt3_r_SG", np.nan)
            validated_sqrt4 = row.get("validated_sqrt4_r_SG", np.nan)
            validated_sqrt7 = row.get("validated_sqrt7_r_SG", np.nan)
            validated_sqrt12 = row.get("validated_sqrt12_r_SG", np.nan)

            def parse_peak_list(col: str):
                """Parse JSON list from CSV field."""
                val = row.get(col, "[]")
                if isinstance(val, str):
                    try:
                        return json.loads(val)
                    except json.JSONDecodeError:
                        return []
                if isinstance(val, list):
                    return val
                return []

            global_peaks_r = parse_peak_list("global_peak_r_values_SG")
            targeted_peaks_r = parse_peak_list("targeted_peak_r_second_third")

            if pd.isna(R1) or R1 <= 0:
                global_peaks_r, targeted_peaks_r = [], []

            global_peaks_r = [p for p in global_peaks_r if p > R1]
            targeted_peaks_r = [p for p in targeted_peaks_r if p > R1]

            def find_best_match(all_sources, expected_val, tol):
                """Find closest peak within tolerance."""
                all_vals = [v for v in all_sources if not pd.isna(v)]
                if not all_vals:
                    return np.nan
                diffs = np.abs(np.array(all_vals) - expected_val)
                idx = np.argmin(diffs)
                return all_vals[idx] if diffs[idx] <= tol * expected_val else np.nan

            # Initialize peaks & ratios
            sqrt3_peak = sqrt4_peak = sqrt7_peak = sqrt12_peak = np.nan
            sqrt3_ratio = sqrt4_ratio = sqrt7_ratio = sqrt12_ratio = np.nan

            if not pd.isna(R1) and R1 > 0:
                # Expected values
                exp3, exp4 = R1 * math.sqrt(3), R1 * math.sqrt(4)
                exp7, exp12 = R1 * math.sqrt(7), R1 * math.sqrt(12)

                # Candidate sources
                sqrt3_sources = global_peaks_r + targeted_peaks_r + [gaussian_peak2, validated_sqrt3]
                sqrt4_sources = global_peaks_r + targeted_peaks_r + [gaussian_peak3, validated_sqrt4]
                sqrt7_sources = global_peaks_r + [validated_sqrt7]
                sqrt12_sources = global_peaks_r + [validated_sqrt12]

                # Match peaks
                sqrt3_peak = find_best_match(sqrt3_sources, exp3, tolerance)
                sqrt4_peak = find_best_match(sqrt4_sources, exp4, tolerance)
                sqrt7_peak = find_best_match(sqrt7_sources, exp7, tolerance)
                sqrt12_peak = find_best_match(sqrt12_sources, exp12, tolerance)

                # Ratios
                sqrt3_ratio = sqrt3_peak / R1 if not np.isnan(sqrt3_peak) else np.nan
                sqrt4_ratio = sqrt4_peak / R1 if not np.isnan(sqrt4_peak) else np.nan
                sqrt7_ratio = sqrt7_peak / R1 if not np.isnan(sqrt7_peak) else np.nan
                sqrt12_ratio = sqrt12_peak / R1 if not np.isnan(sqrt12_peak) else np.nan

            # Collect feature row
            feature_row = {
                "id": atom_id,
                "type": atom_type,
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
            }

            # Add neighbor columns
            for col in neighbor_cols:
                feature_row[col] = row.get(col, np.nan)

            features.append(feature_row)

        except Exception as e:
            logging.error(f"Error processing atom {row.get('id', 'unknown')}: {e}")
            continue

    ml_df = pd.DataFrame(features)

    # Fill missing with per-type medians
    fill_cols = [
        "R1", "sqrt3_peak", "sqrt4_peak", "sqrt7_peak", "sqrt12_peak",
        "sqrt3_ratio", "sqrt4_ratio", "sqrt7_ratio", "sqrt12_ratio",
        "voronoi_volume_temporal", "CN_temporal", "q4", "q6"
    ]
    ml_df[fill_cols] = ml_df.groupby("type")[fill_cols].transform(lambda x: x.fillna(x.median()))

    # Derived features
    ml_df["R3_minus_R1"] = ml_df["sqrt3_peak"] - ml_df["R1"]
    ml_df["R4_minus_R1"] = ml_df["sqrt4_peak"] - ml_df["R1"]
    ml_df["R7_minus_R1"] = ml_df["sqrt7_peak"] - ml_df["R1"]
    ml_df["R12_minus_R1"] = ml_df["sqrt12_peak"] - ml_df["R1"]

    ml_df["R4_minus_R3"] = ml_df["sqrt4_peak"] - ml_df["sqrt3_peak"]
    ml_df["R7_minus_R4"] = ml_df["sqrt7_peak"] - ml_df["sqrt4_peak"]
    ml_df["R12_minus_R7"] = ml_df["sqrt12_peak"] - ml_df["sqrt7_peak"]

    # Row entropy
    def calc_entropy(row):
        vals = row[neighbor_cols].astype(float).values
        total = vals.sum()
        if total == 0:
            return 0
        probs = vals / total
        return -np.sum(probs * np.log(probs + 1e-12))

    ml_df["entropy"] = ml_df.apply(calc_entropy, axis=1)

    # Ratios & interactions
    ml_df["q4_divide_q6"] = ml_df["q4"] / ml_df["q6"].replace([0, np.inf, -np.inf], np.nan)
    ml_df["q4_power_2"] = ml_df["q4"] ** 2
    ml_df["q6_power_2"] = ml_df["q6"] ** 2
    ml_df["R1_time_CN_temporal"] = ml_df["R1"] * ml_df["CN_temporal"]
    ml_df["R3_minus_R1_time_q4"] = ml_df["R3_minus_R1"] * ml_df["q4"]
    ml_df["sqrt3_ratio_divide_sqrt4_ratio"] = ml_df["sqrt3_ratio"] / ml_df["sqrt4_ratio"].replace([0, np.inf, -np.inf], np.nan)

    ml_df["CN_time_q4"] = ml_df["CN_temporal"] * ml_df["q4"]
    ml_df["CN_time_q6"] = ml_df["CN_temporal"] * ml_df["q6"]
    ml_df["entropy_time_q4"] = ml_df["entropy"] * ml_df["q4"]
    ml_df["entropy_time_q6"] = ml_df["entropy"] * ml_df["q6"]

    ml_df["CN_norm_type"] = ml_df["CN_temporal"] / ml_df.groupby("type")["CN_temporal"].transform("mean").replace([0, np.inf, -np.inf], np.nan)
    ml_df["Voronoi_norm_type"] = ml_df["voronoi_volume_temporal"] / ml_df.groupby("type")["voronoi_volume_temporal"].transform("mean").replace([0, np.inf, -np.inf], np.nan)
    ml_df["CN_density"] = ml_df["CN_temporal"] / ml_df["R1"].replace([0, np.inf, -np.inf], np.nan)

    ml_df["log_CN"] = np.log1p(ml_df["CN_temporal"])
    ml_df["log_voronoi"] = np.log1p(ml_df["voronoi_volume_temporal"])

    for col in neighbor_cols:
        ml_df[f"{col}_ratio"] = ml_df[col] / ml_df["CN_temporal"].replace([0, np.inf, -np.inf], np.nan)

    return ml_df


# === Build and save ML table ===
ml_df = build_ml_table(
    "../Atomic_Simulation_Post-Processing_Pipeline/outputs/output_polyamorphous.csv",
    tolerance=0.2
)
# Create the outputs/ directory if it doesn't exist
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
