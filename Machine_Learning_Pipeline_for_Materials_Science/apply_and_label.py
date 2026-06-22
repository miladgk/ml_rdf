"""
apply_and_label.py
==================
Apply trained model to polyamorphous data, write LAMMPS dump with phase labels,
prediction CSV, and report statistics.
"""
import os
import sys
import logging
import yaml
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from data_utils import load_and_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

confidence_threshold = cfg.get("prediction", {}).get("confidence_threshold", 0.90)
model_path = cfg["output"]["model_path"]
unlabeled_csv = cfg["data"]["unlabeled_csv"]
feature_cols = cfg["features"]

# Phase names for reporting
phase_names = {0: "Cu50Zr50", 1: "Cu46Zr54", 2: "Cu64Zr36"}

# ============================================================
# Step 2: Load model and data
# ============================================================
logger.info(f"Loading model from {model_path}")
model = joblib.load(model_path)

logger.info(f"Loading unlabeled data from {unlabeled_csv}")
df = pd.read_csv(unlabeled_csv)
logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")

# Load original pipeline output for x, y, z coordinates (feature builder drops them)
orig_pipeline_csv = "../Atomic_Simulation_Post-Processing_Pipeline/outputs/features_polyamorphous.csv"
df_orig = pd.read_csv(orig_pipeline_csv)
logger.info(f"Loaded original pipeline output from {orig_pipeline_csv} with {len(df_orig.columns)} columns")

# Ensure all feature columns exist
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    logger.warning(f"Missing {len(missing)} features, filling with NaN: {missing[:5]}...")
    for c in missing:
        df[c] = np.nan

X = df[feature_cols].copy()

# ============================================================
# Step 2: Predict with probabilities
# ============================================================
logger.info("Predicting...")
preds = model.predict(X)
probs = model.predict_proba(X)
classes = model.classes_

logger.info(f"Classes: {classes}")
logger.info(f"Predictions shape: {preds.shape}, Probs shape: {probs.shape}")

# Build probability columns
for i, cls in enumerate(classes):
    df[f"prob_class{cls}"] = probs[:, i]

# Confidence = max probability
df["confidence"] = np.max(probs, axis=1)

# Phase label with threshold
df["phase"] = np.where(df["confidence"] >= confidence_threshold, preds, -1)

# ============================================================
# Step 3: Write LAMMPS dump file
# ============================================================
dump_path = "../Atomic_Simulation_Post-Processing_Pipeline/files/snapshots_polyamorphous/combined_final_dump.lammpstrj"
logger.info(f"Reading original LAMMPS dump from {dump_path}")

with open(dump_path, 'r') as f:
    dump_lines = f.readlines()

# Parse header
header = dump_lines[:9]
atom_lines = dump_lines[9:]

# Parse column names from ITEM: ATOMS line
atoms_header = header[8].strip()
if atoms_header.startswith("ITEM:"):
    atoms_header = atoms_header.replace("ITEM: ATOMS", "").strip()
orig_col_names = atoms_header.split()

# Parse atom data
atom_data = []
for line in atom_lines:
    parts = line.strip().split()
    if len(parts) >= len(orig_col_names):
        atom_data.append(parts)

atom_df = pd.DataFrame(atom_data, columns=orig_col_names)
atom_df['id'] = atom_df['id'].astype(int)

# Sort by id to match
atom_df = atom_df.sort_values('id').reset_index(drop=True)

# Also sort prediction data by id
df_sorted = df.sort_values('id').reset_index(drop=True)

# Verify matching
if len(atom_df) != len(df_sorted):
    logger.error(f"Atom count mismatch: {len(atom_df)} vs {len(df_sorted)}")
    sys.exit(1)

# Check ids match
id_match = (atom_df['id'].values == df_sorted['id'].values).all()
if not id_match:
    logger.warning("Atom IDs don't match between dump and feature data, matching by id...")
    # Reindex df_sorted to match atom_df order
    id_map = dict(zip(df_sorted['id'].values, df_sorted.index))
    df_sorted = df_sorted.loc[[id_map[aid] for aid in atom_df['id'].values]].reset_index(drop=True)

# Write new dump
output_dump = "polyamorphous_phase_labeled.lammpstrj"
with open(output_dump, 'w') as f:
    # Write header (first 8 lines — skip line 8 which is original ITEM: ATOMS)
    for line in dump_lines[:8]:
        f.write(line)
    
    # Write new ITEM: ATOMS line with phase and confidence appended
    new_header = "ITEM: ATOMS " + " ".join(orig_col_names) + " phase confidence\n"
    f.write(new_header)
    
    # Write atom data with phase and confidence
    for i in range(len(atom_df)):
        orig_vals = atom_data[i]  # original order
        phase_val = int(df_sorted.iloc[i]['phase'])
        conf_val = f"{df_sorted.iloc[i]['confidence']:.4f}"
        f.write(" ".join(orig_vals) + f" {phase_val} {conf_val}\n")

logger.info(f"Written labeled LAMMPS dump to {output_dump}")

# ============================================================
# Step 4: Write prediction CSV
# ============================================================
pred_csv = "polyamorphous_phase_predictions.csv"
# Get coordinates from original pipeline output (feature builder drops x,y,z)
df_orig_sorted = df_orig.sort_values('id').reset_index(drop=True)
pred_df = pd.DataFrame({
    'atom_id': df_sorted['id'],
    'x': df_orig_sorted['x'],
    'y': df_orig_sorted['y'],
    'z': df_orig_sorted['z'],
    'type': df_orig_sorted['type'],
    'prob_class0': df_sorted['prob_class0'],
    'prob_class1': df_sorted['prob_class1'],
    'prob_class2': df_sorted['prob_class2'],
    'confidence': df_sorted['confidence'],
    'phase': df_sorted['phase'],
})
pred_df.to_csv(pred_csv, index=False)
logger.info(f"Written prediction CSV to {pred_csv}")

# ============================================================
# Step 5: Report statistics
# ============================================================
print("\n" + "="*60)
print("PREDICTION STATISTICS")
print("="*60)

total = len(df_sorted)
for cls in [0, 1, 2]:
    count = (df_sorted['phase'] == cls).sum()
    pct = count / total * 100
    name = phase_names[cls]
    print(f"  Class {cls} ({name}): {count:>6d} atoms ({pct:5.2f}%)")

uncertain_count = (df_sorted['phase'] == -1).sum()
uncertain_pct = uncertain_count / total * 100
print(f"  Uncertain (-1):       {uncertain_count:>6d} atoms ({uncertain_pct:5.2f}%)")

below_threshold = (df_sorted['confidence'] < confidence_threshold).sum()
below_pct = below_threshold / total * 100
print(f"\n  Below confidence threshold ({confidence_threshold:.0%}): {below_threshold:>6d} atoms ({below_pct:5.2f}%)")

# Check for class 1 (Cu46Zr54) above threshold
class1_high_conf = ((df_sorted['phase'] == 1) & (df_sorted['confidence'] >= confidence_threshold)).sum()
if class1_high_conf > 0:
    print(f"\n  *** WARNING: {class1_high_conf} atoms assigned to Class 1 (Cu46Zr54) above threshold! ***")
    print(f"      This may indicate model leakage or feature distribution mismatch.")
else:
    print(f"\n  No atoms assigned to Class 1 (Cu46Zr54) above threshold. ✓")

# Confidence distribution
conf = df_sorted['confidence']
print(f"\n  Confidence distribution:")
print(f"    Mean:   {conf.mean():.4f}")
print(f"    Std:    {conf.std():.4f}")
print(f"    Min:    {conf.min():.4f}")
print(f"    5th %:  {conf.quantile(0.05):.4f}")
print(f"    25th %: {conf.quantile(0.25):.4f}")
print(f"    50th %: {conf.quantile(0.50):.4f}")
print(f"    75th %: {conf.quantile(0.75):.4f}")
print(f"    Max:    {conf.max():.4f}")

# ============================================================
# Step 6: Check Voronoi_norm_type computation
# ============================================================
print("\n" + "="*60)
print("Voronoi_norm_type ANALYSIS")
print("="*60)

# Check how Voronoi_norm_type is computed in feature_builder_data_clean.py
# It's: ml_df["Voronoi_norm_type"] = ml_df["voronoi_volume_temporal"] / type_mean_voronoi
# where type_mean_voronoi = ml_df.groupby("type")["voronoi_volume_temporal"].transform("mean")
# So each atom's Voronoi volume is divided by the mean Voronoi volume of its atom type

print("  Voronoi_norm_type = voronoi_volume_temporal / type_mean_voronoi")
print("  where type_mean_voronoi = mean(voronoi_volume_temporal) per atom type")
print()
print("  packing_fraction_temporal = atomic_sphere_volume / voronoi_volume_temporal")
print("  where atomic_sphere_volume = (4/3)*pi*r^3 per atom type")
print()

# Check correlation
corr = df_sorted['Voronoi_norm_type'].corr(df_sorted['packing_fraction_temporal'])
print(f"  Correlation between Voronoi_norm_type and packing_fraction_temporal: r = {corr:.4f}")

if abs(corr) > 0.95:
    print("  *** FLAG: |r| > 0.95 — these features are highly correlated.")
    print("      Candidate for removal in next model simplification pass.")
else:
    print("  Features are not highly correlated — both can be kept.")

print()
print("  Voronoi_norm_type is mathematically INDEPENDENT of packing_fraction_temporal:")
print("  - Voronoi_norm_type normalizes by TYPE MEAN (statistical normalization)")
print("  - packing_fraction_temporal uses ATOMIC RADIUS (physical normalization)")
print("  They capture different information even if empirically correlated.")