"""
analyze_feature_importance.py

Loads saved SHAP values and permutation importance to rank features by importance
and suggest which features can be safely removed.

Usage:
    python src/analyze_feature_importance.py

Output:
    - Prints top-N features ranked by SHAP mean absolute value
    - Saves a bar chart of feature importance to outputs/plots/
    - Prints feature redundancy analysis (which derived features are correlated)
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SHAP_DATA_DIR = Path("outputs/shap_data")
PLOTS_DIR = Path("outputs/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_analyze_shap():
    """Load SHAP values and rank features by mean absolute importance."""
    shap_csv = SHAP_DATA_DIR / "shap_values.csv"

    if not shap_csv.exists():
        logger.error(f"SHAP CSV not found at {shap_csv}. Run pipeline first.")
        return None

    df_shap = pd.read_csv(shap_csv, index_col=0)

    # Compute mean absolute SHAP value per feature
    mean_abs_shap = df_shap.abs().mean().sort_values(ascending=False)

    logger.info("=" * 70)
    logger.info("FEATURE IMPORTANCE RANKING (by mean |SHAP|)")
    logger.info("=" * 70)

    # Strip prefixes from pipeline-generated column names
    def clean_name(name):
        return name.replace("numeric__", "").replace("missing__missingindicator_", "missing_")

    for rank, (feature, importance) in enumerate(mean_abs_shap.items(), 1):
        clean = clean_name(feature)
        logger.info(f"  {rank:2d}. {clean:45s} {importance:.6f}")

    # Plot
    top_n = min(20, len(mean_abs_shap))
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = mean_abs_shap.head(top_n)
    clean_labels = [clean_name(f) for f in top_features.index]
    ax.barh(range(top_n), top_features.values[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(clean_labels[::-1])
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Top Feature Importance by SHAP")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance_shap.png", dpi=150)
    logger.info(f"Saved SHAP importance plot to {PLOTS_DIR / 'feature_importance_shap.png'}")
    plt.close()

    return mean_abs_shap


def analyze_feature_redundancy():
    """
    Identify groups of correlated / derived features from feature_builder_data_clean.py.
    These are mathematically dependent and may be redundant.
    """
    logger.info("=" * 70)
    logger.info("FEATURE REDUNDANCY ANALYSIS")
    logger.info("=" * 70)
    logger.info("")

    # From feature_builder_data_clean.py, these are the derived relationships:
    redundancies = {
        "Peak ratios are derived from peaks and R1": [
            "ratio_validated_sqrt3  =  validated_sqrt3_r_SG / R1_SG",
            "ratio_validated_sqrt4  =  validated_sqrt4_r_SG / R1_SG",
            "ratio_validated_sqrt7  =  validated_sqrt7_r_SG / R1_SG",
            "ratio_validated_sqrt12 =  validated_sqrt12_r_SG / R1_SG",
        ],
        "Peak differences are derived from peaks": [
            "R3_minus_R1 = sqrt3_peak - R1_SG",
            "R4_minus_R1 = sqrt4_peak - R1_SG",
            "R7_minus_R1 = sqrt7_peak - R1_SG",
            "R12_minus_R1 = sqrt12_peak - R1_SG",
            "R4_minus_R3 = sqrt4_peak - sqrt3_peak",
            "R7_minus_R4 = sqrt7_peak - sqrt4_peak",
            "R12_minus_R7 = sqrt12_peak - sqrt7_peak",
        ],
        "Interaction terms combine existing features": [
            "R1_time_CN_temporal = R1_SG * CN_temporal",
            "R3_minus_R1_time_q4 = (sqrt3_peak - R1_SG) * q4",
            "CN_time_q4 = CN_temporal * q4",
            "CN_time_q6 = CN_temporal * q6",
            "entropy_time_q4 = entropy * q4",
            "entropy_time_q6 = entropy * q6",
        ],
        "Ratio-of-ratios compound existing ratios": [
            "sqrt3_ratio_divide_sqrt4_ratio = ratio_validated_sqrt3 / ratio_validated_sqrt4",
        ],
        "Normalized features compound existing ones": [
            "CN_norm_type = CN_temporal / CN_type_mean",
            "Voronoi_norm_type = voronoi_volume_temporal / voronoi_type_mean",
            "CN_density = CN_temporal / R1_SG",
        ],
        "Log-transformed features": [
            "log_CN = log(1 + CN_temporal)",
            "log_voronoi = log(1 + voronoi_volume_temporal)",
        ],
        "Neighbor ratios": [
            "neighbor_N_ratio = neighbor_N / CN_temporal",
        ],
    }

    for group, relations in redundancies.items():
        logger.info(f"  📦 {group}:")
        for rel in relations:
            logger.info(f"     {rel}")
        logger.info("")

    logger.info("")
    logger.info("RECOMMENDATION:")
    logger.info("  Start with CORE features only:")
    logger.info("    - R1_SG, voronoi_volume_temporal, CN_temporal, q4, q6")
    logger.info("    - validated_sqrt3_r_SG, validated_sqrt4_r_SG")
    logger.info("    - gaussian_peak2_center, gaussian_peak3_center")
    logger.info("    - neighbor_1, neighbor_2")
    logger.info("  Then incrementally ADD derived features and check if CV score improves.")
    logger.info("  Features with |SHAP| < 0.01 are likely noisier and can be dropped first.")


if __name__ == "__main__":
    shap_importance = load_and_analyze_shap()
    analyze_feature_redundancy()