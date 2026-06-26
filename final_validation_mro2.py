"""Complete validation of second-level MRO — Steps 1-5."""
import sys, os, pandas as pd, numpy as np, joblib, yaml
sys.path.insert(0, 'Machine_Learning_Pipeline_for_Materials_Science/src')
from feature_builder_data_clean import build_ml_table

with open('Machine_Learning_Pipeline_for_Materials_Science/config.yaml') as f:
    cfg = yaml.safe_load(f)
atomic_radii = cfg.get('atomic_radii')

# Rebuild all ML tables from raw CSVs
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
print(f"Total features in pipeline: {len(feature_cols)}")

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

# =============================================================================
# Step 1: 3-class model (already trained, load and predict polyamorphous)
# =============================================================================
print("\n" + "="*60)
print("STEP 1: 3-CLASS MODEL — FULL POLYAMORPHOUS PREDICTION")
print("="*60)

dfL = pd.concat([df_5050.assign(phase_label=0), df_4654.assign(phase_label=1), df_6436.assign(phase_label=2)], ignore_index=True)
X, y = dfL[feature_cols].values, dfL['phase_label'].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

m3 = HistGradientBoostingClassifier(random_state=42, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=10, max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20)
cv_scores = cross_val_score(m3, X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='f1_macro')
m3.fit(X_tr, y_tr)
yp = m3.predict(X_te)

print(f"CV F1 macro: {cv_scores.mean():.4f}")
print(f"Test accuracy: {accuracy_score(y_te, yp):.4f}")
print(f"Cu50Zr50 F1: {classification_report(y_te, yp, output_dict=True)['0']['f1-score']:.4f}")
print(f"Cu46Zr54 F1: {classification_report(y_te, yp, output_dict=True)['1']['f1-score']:.4f}")
print(f"Cu64Zr36 F1: {classification_report(y_te, yp, output_dict=True)['2']['f1-score']:.4f}")

cm = confusion_matrix(y_te, yp)
print(f"\nConfusion matrix:\n{cm}")

joblib.dump(m3, 'Machine_Learning_Pipeline_for_Materials_Science/outputs/models/model_3class.pkl')

# Polyamorphous prediction
Xp = df_poly[feature_cols].values
probs = m3.predict_proba(Xp)
preds = m3.predict(Xp)
conf = np.max(probs, axis=1)

with open('Atomic_Simulation_Post-Processing_Pipeline/files/snapshots_polyamorphous/combined_final_dump.lammpstrj') as f:
    ll = f.readlines()
cols = ll[8].strip().replace('ITEM: ATOMS', '').strip().split()
atoms = pd.DataFrame([l.strip().split() for l in ll[9:] if l.strip()], columns=cols)
atoms['id'] = atoms['id'].astype(int)
atoms['x'] = atoms['x'].astype(float)
atoms_sorted = atoms.sort_values('id').reset_index(drop=True)

# Build prediction DataFrame
df_pred = pd.DataFrame({'id': atoms_sorted['id'],
                         'x': atoms_sorted['x'],
                         'phase': preds,
                         'confidence': conf,
                         'class_0_prob': probs[:, 0],
                         'class_1_prob': probs[:, 1],
                         'class_2_prob': probs[:, 2]})

print(f"\n{'Thresh':>7s} {'Class 0':>20s} {'Class 1':>20s} {'Class 2':>20s} {'Uncert':>12s}")
for th in [0.50, 0.60, 0.70, 0.80, 0.90]:
    phase = np.where(conf >= th, preds, -1)
    t = len(phase)
    def fmt(mask):
        cnt = mask.sum()
        if cnt == 0: return f'{cnt:6d}  0.0%'
        return f'{cnt:6d} {cnt/t*100:5.1f}%'
    print(f'{th:.2f}   c0:{fmt(phase==0):>18s}   c1:{fmt(phase==1):>18s}   c2:{fmt(phase==2):>18s}   {(phase==-1).sum():5d} {(phase==-1).sum()/t*100:.1f}%')

print(f"\n--- Confidence distribution ---")
print(f"Mean confidence: {conf.mean():.4f}")
print(f"Std confidence:  {conf.std():.4f}")
print(f"% above 0.90:    {(conf > 0.90).mean():.3f}")
print(f"% above 0.80:    {(conf > 0.80).mean():.3f}")
print(f"% above 0.60:    {(conf > 0.60).mean():.3f}")

# =============================================================================
# Step 2: Binary model retrain
# =============================================================================
print("\n" + "="*60)
print("STEP 2: BINARY MODEL (5050 vs 6436) WITH 2ND-LEVEL MRO")
print("="*60)

df_bin = pd.concat([df_5050.assign(phase_label=0), df_6436.assign(phase_label=2)], ignore_index=True)
Xb, yb = df_bin[feature_cols].values, df_bin['phase_label'].values
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.3, random_state=42, stratify=yb)

m2 = HistGradientBoostingClassifier(random_state=42, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=10, max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20)
cv_bin = cross_val_score(m2, Xb_tr, yb_tr, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring='f1_macro')
m2.fit(Xb_tr, yb_tr)
ypb = m2.predict(Xb_te)
print(f"CV F1 macro: {cv_bin.mean():.4f}")
print(f"Test accuracy: {accuracy_score(yb_te, ypb):.4f}")
print(f"Confusion matrix:\n{confusion_matrix(yb_te, ypb)}")

joblib.dump(m2, 'Machine_Learning_Pipeline_for_Materials_Science/outputs/models/model_binary_0_2.pkl')

# =============================================================================
# Step 3: LAMMPS output file
# =============================================================================
print("\n" + "="*60)
print("STEP 3: GENERATE LAMMPS OUTPUT FILE")
print("="*60)

# Use threshold from config
threshold = cfg.get('prediction', {}).get('confidence_threshold', 0.60)
poly_preds_clean = np.where(conf >= threshold, preds, -1)

# Build output
output_lines = []
output_lines.append("ITEM: TIMESTEP")
output_lines.append("0")
output_lines.append("ITEM: NUMBER OF ATOMS")
output_lines.append(str(len(df_pred)))
output_lines.append("ITEM: BOX BOUNDS pp pp pp")
output_lines.append("-500.0 500.0")
output_lines.append("-500.0 500.0")
output_lines.append("-500.0 500.0")
output_lines.append("ITEM: ATOMS id type x y z phase_label confidence")

# Get original atomic data from dump
dump_path = 'Atomic_Simulation_Post-Processing_Pipeline/files/snapshots_polyamorphous/combined_final_dump.lammpstrj'
with open(dump_path, 'r') as f:
    dump_lines = f.readlines()
cols = dump_lines[8].strip().replace('ITEM: ATOMS', '').strip().split()
x_col = cols.index('x')
y_col = cols.index('y')
z_col = cols.index('z')
type_col = cols.index('type')
id_col = cols.index('id')

df_sorted = atoms_sorted.copy()
for i, line in enumerate(dump_lines[9:]):
    if not line.strip(): continue
    parts = line.strip().split()
    atom_id = int(parts[id_col])
    atom_type = parts[type_col]
    ax, ay, az = parts[x_col], parts[y_col], parts[z_col]
    
    # Find prediction for this atom
    idx_in_pred = df_pred[df_pred['id'] == atom_id].index
    if len(idx_in_pred) > 0:
        pred_phase = int(poly_preds_clean[idx_in_pred[0]])
        pred_conf = conf[idx_in_pred[0]]
    else:
        pred_phase = -2
        pred_conf = 0.0
    
    output_lines.append(f"{atom_id} {atom_type} {ax} {ay} {az} {pred_phase} {pred_conf:.4f}")

out_path = 'Atomic_Simulation_Post-Processing_Pipeline/outputs/polyamorphous_phase_labeled.lammpstrj'
with open(out_path, 'w') as f:
    f.write('\n'.join(output_lines) + '\n')

print(f"File written: {out_path}")

# Validate
total_atoms = len(df_pred)
with open(out_path, 'r') as f:
    out_text = f.read()
    num_lines = out_text.strip().split('\n')
    atom_data_lines = [l for l in num_lines if l and not l.startswith('ITEM:')]

# Verify atom count
n_atoms_check = int(num_lines[3])
print(f"\n--- Validation ---")
print(f"Total atoms in header: {n_atoms_check}")
print(f"Total data lines: {len(atom_data_lines)}")
print(f"Match: {n_atoms_check == len(atom_data_lines)}")

# Verify phase values
phases = set()
for line in num_lines:
    if not line or line.startswith('ITEM:'): continue
    parts = line.strip().split()
    if len(parts) >= 7:
        phases.add(parts[6])
phases_int = sorted([int(p) for p in phases])
print(f"Phase values found: {phases_int}")
valid = set(phases_int).issubset({-2, -1, 0, 1, 2})
print(f"Valid phase range (all -2,-1,0,1,2): {valid}")

# Verify confidence values
conf_values = []
for line in num_lines:
    if not line or line.startswith('ITEM:'): continue
    parts = line.strip().split()
    if len(parts) >= 7:
        conf_values.append(float(parts[6]))
conf_arr = np.array(conf_values)
print(f"Confidence range: [{conf_arr.min():.4f}, {conf_arr.max():.4f}]")
print(f"All in [0,1]: {conf_arr.min() >= 0 and conf_arr.max() <= 1}")

import os as os_mod
file_size = os_mod.path.getsize(out_path)
print(f"File size: {file_size/1e6:.2f} MB")

# =============================================================================
# Step 4: Spatial validation
# =============================================================================
print("\n" + "="*60)
print("STEP 4: SPATIAL VALIDATION AT THRESHOLD 0.60")
print("="*60)

phase_labels = {0: 'Cu50Zr50', 1: 'Cu46Zr54', 2: 'Cu64Zr36', -1: 'Uncertain'}
for phase_label, phase_name in phase_labels.items():
    mask = poly_preds_clean == phase_label
    cnt = mask.sum()
    pct = cnt / total_atoms * 100
    if cnt == 0:
        print(f"{phase_name:15s}: 0 atoms")
        continue
    x_vals = df_pred.loc[mask, 'x'].values
    print(f"{phase_name:15s}: {cnt:5d} atoms ({pct:5.1f}%), "
          f"x_mean={x_vals.mean():.1f}, x_std={x_vals.std():.1f}, "
          f"x_min={x_vals.min():.1f}, x_max={x_vals.max():.1f}")

# =============================================================================
# Step 5: Final summary
# =============================================================================
print("\n" + "="*60)
print("STEP 5: FINAL MODEL SUMMARY")
print("="*60)

print(f"Total features in pipeline: {len(feature_cols)}")
print(f"3-class model CV F1: {cv_scores.mean():.4f}")
print(f"3-class model test accuracy: {accuracy_score(y_te, yp):.4f}")
print(f"Binary model CV F1: {cv_bin.mean():.4f}")
print(f"Binary model test accuracy: {accuracy_score(yb_te, ypb):.4f}")
print(f"\nModel files saved:")
print(f"  - 3-class: outputs/models/model_3class.pkl")
print(f"  - binary:  outputs/models/model_binary_0_2.pkl")
print(f"\nOutput files generated:")
print(f"  - polyamorphous_phase_labeled.lammpstrj")
print(f"  - features_polyamorphous.csv (in data/ and outputs/)")
print(f"\nKey validation metric:")
c1_at_060 = (poly_preds_clean == 1).sum()
print(f"  Class 1 (Cu46Zr54) false-positive at thresh 0.60: {c1_at_060} ({c1_at_060/total_atoms*100:.1f}%)")
if c1_at_060 / total_atoms < 0.05:
    print("  ✅ Below 5% threshold — pipeline scientifically complete")
else:
    print("  ❌ Above 5% threshold")

print("\nDone.")