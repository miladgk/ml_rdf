import os
import sys
import yaml
import subprocess
import numpy as np
import pandas as pd

print("="*70)
print("MASTER WORKFLOW AUTOMATION: LEVEL 3 MRO & SPATIAL KERNEL SMOOTHING")
print("="*70)

config_path = "Atomic_Simulation_Post-Processing_Pipeline/config.yaml"
datasets = ["5050", "4654", "6436", "polyamorphous"]

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

cfg["MRO_DEPTH"] = 3
cfg["COMPUTE_SECOND_LEVEL_MRO"] = True

# Step 1: Extract snapshots & temporal averaging for each amorphous box + polyamorphous box
print("\n[Step 1/4] Extracting atomistic features (MRO Level 3 & Chemical MRO) across all folders...")
for ds in datasets:
    target_csv = f"Atomic_Simulation_Post-Processing_Pipeline/outputs/features_{ds}.csv"
    if os.path.exists(target_csv):
        print(f"  -> {ds} already extracted at {target_csv}. Skipping calculation.")
        continue
    print(f"  -> Running pipeline on {ds}...")
    cfg["SNAPSHOT_DIRECTORY"] = f"files/snapshots_{ds}/"
    cfg["output_csv"] = f"outputs/features_{ds}.csv"
    cfg["PLOT_OUTPUT_FILE"] = f"outputs/sample_peak_{ds}.png"
    
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
        
    cmd = ["conda", "run", "-n", "materials-sim-ml", "python3", "Atomic_Simulation_Post-Processing_Pipeline/src/pipeline_orchestrator.py", "--config", config_path]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"!!! ERROR executing pipeline on {ds} !!!")
        sys.exit(1)

# Restore config path to default
cfg["SNAPSHOT_DIRECTORY"] = "files/snapshots_polyamorphous/"
cfg["output_csv"] = "outputs/features_polyamorphous.csv"
cfg["PLOT_OUTPUT_FILE"] = "outputs/sample_peak_polyamorphous.png"
with open(config_path, "w") as f:
    yaml.dump(cfg, f)

# Step 2: Clean datasets & copy to Machine_Learning_Pipeline_for_Materials_Science/data/
print("\n[Step 2/4] Cleaning tables and migrating to ML data directory...")
sys.path.append('Machine_Learning_Pipeline_for_Materials_Science/src')
from feature_builder_data_clean import build_ml_table

ml_cfg_path = "Machine_Learning_Pipeline_for_Materials_Science/config.yaml"
with open(ml_cfg_path, "r") as f:
    ml_cfg = yaml.safe_load(f)
atomic_radii = ml_cfg.get('atomic_radii', {1: 1.28, 2: 1.60})

for ds in datasets:
    in_csv = f"Atomic_Simulation_Post-Processing_Pipeline/outputs/features_{ds}.csv"
    out_csv = f"Machine_Learning_Pipeline_for_Materials_Science/data/features_{ds}.csv"
    df_clean = build_ml_table(in_csv, tolerance=0.2, atomic_radii=atomic_radii)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_clean.to_csv(out_csv, index=False)
    print(f"  Migrated {ds}: {len(df_clean)} rows, {len(df_clean.columns)} cols -> {out_csv}")

# Step 3: Train 3-Class Classifier (HistGB) & Predict on Polyamorphous
print("\n[Step 3/4] Training 3-Class HistGradientBoosting Classifier...")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df_0 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_5050.csv')
df_1 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_4654.csv')
df_2 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_6436.csv')

chem_cols = ['cu_fraction_2nd_shell_temporal', 'zr_fraction_2nd_shell_temporal',
             'cu_fraction_3rd_shell_temporal', 'zr_fraction_3rd_shell_temporal']

for phase_name, df_phase, expected_cu in [('5050', df_0, 0.50), ('4654', df_1, 0.46), ('6436', df_2, 0.64)]:
    print(f"\n=== {phase_name} (expected Cu={expected_cu}) ===")
    for col in chem_cols:
        if col in df_phase:
            print(f"  {col}: mean={df_phase[col].mean():.4f}, std={df_phase[col].std():.4f}")

for col in ['cu_fraction_2nd_shell_temporal', 'cu_fraction_3rd_shell_temporal']:
    if col in df_0 and col in df_1:
        v5050 = df_0[col]
        v4654 = df_1[col]
        q5 = v4654.quantile(0.05)
        q95 = v4654.quantile(0.95)
        overlap = ((v5050 >= q5) & (v5050 <= q95)).mean()
        print(f"\n{col}:")
        print(f"  5050 mean={v5050.mean():.4f}, std={v5050.std():.4f}")
        print(f"  4654 mean={v4654.mean():.4f}, std={v4654.std():.4f}")
        print(f"  Signal (difference in means): {abs(v5050.mean()-v4654.mean()):.4f}")
        print(f"  Overlap fraction: {overlap:.3f}")

df_0['phase_label'] = 0
df_1['phase_label'] = 1
df_2['phase_label'] = 2

df_train = pd.concat([df_0, df_1, df_2], ignore_index=True)
feature_cols = [c for c in ml_cfg['features'] if c in df_train.columns]

X_train = df_train[feature_cols].fillna(0).values
y_train = df_train['phase_label'].values

model = HistGradientBoostingClassifier(random_state=42, early_stopping=True, max_iter=300)
model.fit(X_train, y_train)

df_poly = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_polyamorphous.csv')
df_poly = df_poly.sort_values('id').reset_index(drop=True)
X_poly = df_poly[feature_cols].fillna(0).values
raw_probs = model.predict_proba(X_poly)
raw_preds = np.argmax(raw_probs, axis=1)

# Step 4: Spatial Kernel Smoothing & Export to LAMMPS Dump
print("\n[Step 4/4] Applying Periodic Gaussian Spatial Kernel Smoothing...")
from scipy.spatial import cKDTree

dump_path = "Atomic_Simulation_Post-Processing_Pipeline/files/snapshots_polyamorphous/combined_final_dump.lammpstrj"
with open(dump_path, 'r') as f:
    lines = f.readlines()

box_x = [float(x) for x in lines[5].strip().split()]
box_y = [float(x) for x in lines[6].strip().split()]
box_z = [float(x) for x in lines[7].strip().split()]
Lx = box_x[1] - box_x[0]
Ly = box_y[1] - box_y[0]
Lz = box_z[1] - box_z[0]

cols = lines[8].strip().split()[2:]
data = [l.strip().split() for l in lines[9:] if l.strip()]
dump_df = pd.DataFrame(data, columns=cols)
dump_df['id'] = dump_df['id'].astype(int)
for c in ['x', 'y', 'z']:
    dump_df[c] = dump_df[c].astype(float)

dump_sorted = dump_df.sort_values('id').reset_index(drop=True)
pos = dump_sorted[['x', 'y', 'z']].values

y_true = np.zeros(len(pos), dtype=int)
y_true[0:11664] = 1        # Cu46Zr54 (Class 1)
y_true[11664:23328] = 0    # Cu50Zr50 (Class 0)
y_true[23328:] = 2         # Cu64Zr36 (Class 2)

print(f">>> Raw ML Accuracy (Before Spatial Smoothing): {accuracy_score(y_true, raw_preds)*100:.2f}%")

pos_origin = np.array([box_x[0], box_y[0], box_z[0]])
box_dims = np.array([Lx, Ly, Lz])
pos_wrapped = (pos - pos_origin) % box_dims

tree = cKDTree(pos_wrapped, boxsize=box_dims)
R_cutoff = 14.0
sigma = 6.0
smoothed_probs = np.zeros_like(raw_probs)

for i in range(len(pos_wrapped)):
    idxs = tree.query_ball_point(pos_wrapped[i], r=R_cutoff)
    neigh_pos = pos_wrapped[idxs]
    diff = neigh_pos - pos_wrapped[i]
    diff = diff - np.round(diff / box_dims) * box_dims
    dists = np.linalg.norm(diff, axis=1)
    weights = np.exp(-0.5 * (dists / sigma)**2)
    weights /= weights.sum()
    smoothed_probs[i] = np.dot(weights, raw_probs[idxs])

smooth_preds = np.argmax(smoothed_probs, axis=1)
smooth_conf = np.max(smoothed_probs, axis=1)

acc_smooth = accuracy_score(y_true, smooth_preds)
print(f">>> Final Smoothed Accuracy (After Spatial Smoothing): {acc_smooth*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_true, smooth_preds, target_names=['Cu50Zr50 (Class 0)', 'Cu46Zr54 (Class 1)', 'Cu64Zr36 (Class 2)']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, smooth_preds))

out_dump = "Atomic_Simulation_Post-Processing_Pipeline/files/polyamorphous_mro3_FINAL.lammpstrj"
with open(out_dump, 'w') as out:
    out.writelines(lines[:9])
    header = lines[8].strip() + " raw_phase smooth_phase smooth_conf true_phase\n"
    out.write(header)
    for idx, row in dump_df.iterrows():
        atom_id = int(row['id'])
        arr_idx = atom_id - 1
        line_str = " ".join(str(row[c]) for c in cols)
        out.write(f"{line_str} {raw_preds[arr_idx]} {smooth_preds[arr_idx]} {smooth_conf[arr_idx]:.4f} {y_true[arr_idx]}\n")

print(f"\nSUCCESS! Exported complete validated trajectory to {out_dump}")

low_conf_mask = (smooth_conf > 0.50) & (smooth_conf < 0.70)
low_conf_x = dump_df.loc[low_conf_mask, 'x'].astype(float)
print(f"\nLow-confidence atoms (0.50-0.70): {len(low_conf_x)}")
if len(low_conf_x) > 0:
    print(f"x distribution: mean={low_conf_x.mean():.1f}, std={low_conf_x.std():.1f}")
    print(f"x percentiles: 10th={low_conf_x.quantile(0.10):.1f}, "
          f"50th={low_conf_x.quantile(0.50):.1f}, "
          f"90th={low_conf_x.quantile(0.90):.1f}")
