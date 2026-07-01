import os
import sys
import yaml
import numpy as np
import pandas as pd
import freud
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("="*70)
print("SPATIAL KERNEL SMOOTHING FOR METALLIC GLASS PHASE CLASSIFICATION")
print("="*70)

# Load config
with open('Machine_Learning_Pipeline_for_Materials_Science/config.yaml') as f:
    cfg = yaml.safe_load(f)

# Step 1: Load training datasets and train model
print("\n[1/5] Loading training data and fitting 3-class HGB model...")
df_0 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_5050.csv')
df_1 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_4654.csv')
df_2 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_6436.csv')

df_0['phase_label'] = 0
df_1['phase_label'] = 1
df_2['phase_label'] = 2

df_train = pd.concat([df_0, df_1, df_2], ignore_index=True)
feature_cols = [c for c in cfg['features'] if c in df_train.columns]

X_train = df_train[feature_cols].values
y_train = df_train['phase_label'].values

model = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, random_state=42
)
model.fit(X_train, y_train)
print(f"Model fitted on {len(df_train)} atoms using {len(feature_cols)} features.")

# Step 2: Predict raw probabilities on polyamorphous dataset
print("\n[2/5] Predicting raw atomistic probabilities on polyamorphous snapshot...")
df_poly = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_polyamorphous.csv')
df_poly = df_poly.sort_values('id').reset_index(drop=True)

X_poly = df_poly[feature_cols].values
raw_probs = model.predict_proba(X_poly) # shape (36828, 3)
raw_preds = model.predict(X_poly)

# Step 3: Parse LAMMPS dump coordinates and box bounds
print("\n[3/5] Reading atom positions and periodic box from LAMMPS dump...")
dump_path = "Atomic_Simulation_Post-Processing_Pipeline/files/snapshots_polyamorphous/combined_final_dump.lammpstrj"
with open(dump_path, 'r') as f:
    dump_lines = f.readlines()

x_lo, x_hi = [float(val) for val in dump_lines[5].split()[:2]]
y_lo, y_hi = [float(val) for val in dump_lines[6].split()[:2]]
z_lo, z_hi = [float(val) for val in dump_lines[7].split()[:2]]

box_size = np.array([x_hi - x_lo, y_hi - y_lo, z_hi - z_lo])
lower_bounds = np.array([x_lo, y_lo, z_lo])

header = dump_lines[:8]
atoms_header = dump_lines[8].strip()
cols = atoms_header.replace("ITEM: ATOMS", "").strip().split()

dump_rows = [l.strip().split() for l in dump_lines[9:] if l.strip()]
dump_df = pd.DataFrame(dump_rows, columns=cols)
dump_df['id'] = dump_df['id'].astype(int)
dump_df['x'] = dump_df['x'].astype(float)
dump_df['y'] = dump_df['y'].astype(float)
dump_df['z'] = dump_df['z'].astype(float)

# Align dump positions to sorted ID order (1..36828)
dump_sorted = dump_df.sort_values('id').reset_index(drop=True)
pos = dump_sorted[['x', 'y', 'z']].values
pos_wrapped = np.mod(pos - lower_bounds, box_size)

# Step 4: Spatial Gaussian Kernel Smoothing via freud
r_smooth = 14.0 # Angstrom cutoff window (~200 atoms)
sigma = 6.0     # Gaussian kernel width

from scipy.spatial import cKDTree

print(f"\n[4/5] Applying periodic Gaussian kernel smoothing (R_cutoff={r_smooth} A, sigma={sigma} A)...")
tree = cKDTree(pos_wrapped, boxsize=box_size)
pairs = tree.query_pairs(r=r_smooth, output_type='ndarray')

# Include symmetric pairs and self pairs
self_pairs = np.column_stack([np.arange(len(pos)), np.arange(len(pos))])
all_pairs = np.vstack([pairs, pairs[:, [1, 0]], self_pairs])

i_idx = all_pairs[:, 0]
j_idx = all_pairs[:, 1]

# Compute periodic distances
d_vec = np.abs(pos_wrapped[i_idx] - pos_wrapped[j_idx])
d_vec = np.where(d_vec > 0.5 * box_size, box_size - d_vec, d_vec)
dist = np.linalg.norm(d_vec, axis=1)

w = np.exp(-0.5 * (dist / sigma) ** 2)

smoothed_probs = np.zeros_like(raw_probs)
weights_sum = np.zeros(len(pos))

np.add.at(smoothed_probs, i_idx, raw_probs[j_idx] * w[:, None])
np.add.at(weights_sum, i_idx, w)

smoothed_probs /= weights_sum[:, None]
smooth_preds = np.argmax(smoothed_probs, axis=1)
smooth_conf = np.max(smoothed_probs, axis=1)

# Ground truth assignment
y_true = np.zeros(len(pos), dtype=int)
y_true[0:11664] = 1 # IDs 1..11664 -> Cu46Zr54 (Class 1)
y_true[11664:23328] = 0 # IDs 11665..23328 -> Cu50Zr50 (Class 0)
y_true[23328:] = 2 # IDs 23329..36828 -> Cu64Zr36 (Class 2)

raw_acc = accuracy_score(y_true, raw_preds)
smooth_acc = accuracy_score(y_true, smooth_preds)

print(f"\n>>> Raw Accuracy:      {raw_acc*100:.2f}%")
print(f">>> Smoothed Accuracy: {smooth_acc*100:.2f}%\n")

target_names = ['Cu50Zr50 (Class 0)', 'Cu46Zr54 (Class 1)', 'Cu64Zr36 (Class 2)']
print("Classification Report (After Spatial Smoothing):")
print(classification_report(y_true, smooth_preds, target_names=target_names, digits=4))

print("Confusion Matrix (Rows=True, Cols=Pred):")
cm = confusion_matrix(y_true, smooth_preds)
cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
print(cm_df)

# Step 5: Write out final smoothed LAMMPS dump
print("\n[5/5] Writing smoothed LAMMPS dump file for OVITO...")
out_dump = "Atomic_Simulation_Post-Processing_Pipeline/files/polyamorphous_smoothed_kernel.lammpstrj"

# Create lookup dict keying atom_id -> properties
out_lookup = {}
for idx in range(len(dump_sorted)):
    aid = int(dump_sorted.iloc[idx]['id'])
    out_lookup[aid] = (
        int(smooth_preds[idx]),
        f"{smooth_conf[idx]:.4f}",
        int(raw_preds[idx]),
        int(y_true[idx])
    )

id_col_idx = cols.index('id')

with open(out_dump, 'w') as out:
    for h in header:
        out.write(h)
    out.write(atoms_header + " smooth_phase smooth_conf raw_phase true_phase\n")
    
    for row_parts in dump_rows:
        aid = int(row_parts[id_col_idx])
        sp, sc, rp, tp = out_lookup.get(aid, (-1, "0.0000", -1, -1))
        out.write(" ".join(row_parts) + f" {sp} {sc} {rp} {tp}\n")

print(f"\nSuccessfully created {out_dump}")
print("Open this file in OVITO and color by 'smooth_phase'!")
