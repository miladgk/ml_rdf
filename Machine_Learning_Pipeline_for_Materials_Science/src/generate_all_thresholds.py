"""Generate labeled LAMMPS files for all thresholds for visual inspection."""
import os, sys, pandas as pd, numpy as np, joblib, yaml
sys.path.insert(0, 'Machine_Learning_Pipeline_for_Materials_Science/src')
from feature_builder_data_clean import build_ml_table

with open('Machine_Learning_Pipeline_for_Materials_Science/config.yaml') as f:
    cfg = yaml.safe_load(f)
atomic_radii = cfg.get('atomic_radii')

# Rebuild ML tables
for ds in ['5050', '4654', '6436', 'polyamorphous']:
    in_path = f'Atomic_Simulation_Post-Processing_Pipeline/outputs/features_{ds}.csv'
    out_path = f'Machine_Learning_Pipeline_for_Materials_Science/data/features_{ds}.csv'
    df = build_ml_table(in_path, tolerance=0.2, atomic_radii=atomic_radii)
    df.to_csv(out_path, index=False)

df_5050 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_5050.csv')
df_4654 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_4654.csv')
df_6436 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_6436.csv')
df_poly = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_polyamorphous.csv')

feature_cols = [c for c in cfg['features'] if c in df_5050.columns]

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Train 3-class model
dfL = pd.concat([df_5050.assign(phase_label=0), df_4654.assign(phase_label=1), df_6436.assign(phase_label=2)], ignore_index=True)
X, y = dfL[feature_cols].values, dfL['phase_label'].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

m3 = HistGradientBoostingClassifier(random_state=42, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=10, max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20)
m3.fit(X_tr, y_tr)

# Predict polyamorphous
Xp = df_poly[feature_cols].values
probs = m3.predict_proba(Xp)
preds = m3.predict(Xp)
conf = np.max(probs, axis=1)

# Read original dump
dump_path = 'Machine_Learning_Pipeline_for_Materials_Science/data/polyamorphous/combined_final_dump.lammpstrj'
with open(dump_path, 'r') as f:
    dump_lines = f.readlines()

n_atoms = int(dump_lines[3].strip())
header_atoms_line = dump_lines[8].strip()
cols_str = header_atoms_line.replace('ITEM: ATOMS', '').strip()
orig_col_names = cols_str.split()
dump_data = [l.strip().split() for l in dump_lines[9:] if l.strip()]

atom_df = pd.DataFrame(dump_data, columns=orig_col_names)
atom_df['id'] = atom_df['id'].astype(int)
atom_df = atom_df.sort_values('id').reset_index(drop=True)

out_dir = 'Machine_Learning_Pipeline_for_Materials_Science/outputs'
os.makedirs(out_dir, exist_ok=True)

# Summary table
print(f"{'Threshold':>10s} {'Cu50Zr50':>12s} {'Cu46Zr54':>12s} {'Cu64Zr36':>12s} {'Uncertain':>12s}")
print("-" * 60)

for th in [0.50, 0.60, 0.70, 0.80, 0.90]:
    phase = np.where(conf >= th, preds, -1)
    t = len(phase)
    c0 = (phase == 0).sum()
    c1 = (phase == 1).sum()
    c2 = (phase == 2).sum()
    unc = (phase == -1).sum()
    print(f"{th:.2f}        {c0:8d} {c0/t*100:5.1f}%   {c1:8d} {c1/t*100:5.1f}%   {c2:8d} {c2/t*100:5.1f}%   {unc:8d} {unc/t*100:5.1f}%")

    # Write LAMMPS dump
    th_str = f"{th:.2f}"
    output_dump = os.path.join(out_dir, f'polyamorphous_phase_labeled_{th_str}.lammpstrj')
    with open(output_dump, 'w') as f:
        for line in dump_lines[:8]:
            f.write(line)
        new_header = "ITEM: ATOMS " + " ".join(orig_col_names) + " phase confidence\n"
        f.write(new_header)
        for i in range(len(dump_data)):
            orig_vals = dump_data[i]
            phase_val = int(phase[i])
            conf_val = f"{conf[i]:.4f}"
            f.write(" ".join(orig_vals) + f" {phase_val} {conf_val}\n")

    # Write prediction CSV
    pred_csv = os.path.join(out_dir, f'polyamorphous_phase_predictions_{th_str}.csv')
    pred_df = pd.DataFrame({
        'atom_id': atom_df['id'],
        'x': atom_df['x'].astype(float),
        'y': atom_df['y'].astype(float),
        'z': atom_df['z'].astype(float),
        'type': atom_df['type'],
        'prob_class0': probs[:, 0],
        'prob_class1': probs[:, 1],
        'prob_class2': probs[:, 2],
        'confidence': conf,
        'phase': phase,
    })
    pred_df.to_csv(pred_csv, index=False)

print(f"\nDone. Generated {len([0.50, 0.60, 0.70, 0.80, 0.90])} threshold files in {out_dir}")