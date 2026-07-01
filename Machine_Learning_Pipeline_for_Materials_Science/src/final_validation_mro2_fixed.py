"""Complete validation of second-level MRO — corrected with proper box bounds and output paths."""
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

# =============================================================================
# Step 1: 3-class model
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
print(f"Confusion matrix:\n{cm}")

joblib.dump(m3, 'Machine_Learning_Pipeline_for_Materials_Science/outputs/models/model_3class.pkl')

# =============================================================================
# Polyamorphous prediction
# =============================================================================
Xp = df_poly[feature_cols].values
probs = m3.predict_proba(Xp)
preds = m3.predict(Xp)
conf = np.max(probs, axis=1)

# Read original dump file from ML data directory
dump_path = 'Machine_Learning_Pipeline_for_Materials_Science/data/polyamorphous/combined_final_dump.lammpstrj'
with open(dump_path, 'r') as f:
    dump_lines = f.readlines()

# Parse header
n_atoms = int(dump_lines[3].strip())
# Box bounds (lines 6-8)
box_x_lo, box_x_hi = [float(x) for x in dump_lines[5].strip().split()]
box_y_lo, box_y_hi = [float(x) for x in dump_lines[6].strip().split()]
box_z_lo, box_z_hi = [float(x) for x in dump_lines[7].strip().split()]
print(f"\nBox bounds from dump: x=[{box_x_lo:.2f}, {box_x_hi:.2f}], "
      f"y=[{box_y_lo:.2f}, {box_y_hi:.2f}], z=[{box_z_lo:.2f}, {box_z_hi:.2f}]")

# Parse atom data
header_atoms_line = dump_lines[8].strip()
cols_str = header_atoms_line.replace('ITEM: ATOMS', '').strip()
orig_col_names = cols_str.split()
dump_data = [l.strip().split() for l in dump_lines[9:] if l.strip()]

atom_df = pd.DataFrame(dump_data, columns=orig_col_names)
atom_df['id'] = atom_df['id'].astype(int)
atom_df = atom_df.sort_values('id').reset_index(drop=True)

# Build prediction DataFrame aligned by id
df_pred = pd.DataFrame({
    'id': atom_df['id'].values,
    'x': atom_df['x'].astype(float).values,
    'y': atom_df['y'].astype(float).values,
    'z': atom_df['z'].astype(float).values,
    'phase': preds,
    'confidence': conf,
    'class_0_prob': probs[:, 0],
    'class_1_prob': probs[:, 1],
    'class_2_prob': probs[:, 2],
})

# Threshold sweep
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
# Step 2: Binary model
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
# Step 3: LAMMPS output file + prediction CSV
# =============================================================================
print("\n" + "="*60)
print("STEP 3: GENERATE LABELED OUTPUT FILES")
print("="*60)

# Use threshold from config
confidence_threshold = cfg.get('prediction', {}).get('confidence_threshold', 0.60)
poly_preds_clean = np.where(conf >= confidence_threshold, preds, -1)

out_dir = 'Machine_Learning_Pipeline_for_Materials_Science/outputs'
os.makedirs(out_dir, exist_ok=True)

# --- Write LAMMPS dump ---
output_dump = os.path.join(out_dir, 'polyamorphous_phase_labeled.lammpstrj')
with open(output_dump, 'w') as f:
    # Write header (8 lines, with actual box bounds)
    for line in dump_lines[:8]:
        f.write(line)
    # Write new ITEM: ATOMS line with phase and confidence appended
    new_header = "ITEM: ATOMS " + " ".join(orig_col_names) + " phase confidence\n"
    f.write(new_header)
    # Write atom data with phase and confidence
    for i in range(len(atom_df)):
        orig_vals = dump_data[i]
        phase_val = int(poly_preds_clean[i])
        conf_val = f"{conf[i]:.4f}"
        f.write(" ".join(orig_vals) + f" {phase_val} {conf_val}\n")

print(f"Written: {output_dump}")

# --- Write prediction CSV ---
pred_csv = os.path.join(out_dir, 'polyamorphous_phase_predictions.csv')
pred_df = pd.DataFrame({
    'atom_id': df_pred['id'],
    'x': df_pred['x'],
    'y': df_pred['y'],
    'z': df_pred['z'],
    'type': atom_df['type'].values,
    'prob_class0': df_pred['class_0_prob'],
    'prob_class1': df_pred['class_1_prob'],
    'prob_class2': df_pred['class_2_prob'],
    'confidence': df_pred['confidence'],
    'phase': poly_preds_clean,
})
pred_df.to_csv(pred_csv, index=False)
print(f"Written: {pred_csv}")

# --- Validation ---
print(f"\n--- Validation ---")
# Atom count
n_data_lines = len(dump_data)
print(f"Total atoms in header: {n_atoms}")
print(f"Total data lines: {n_data_lines}")
print(f"Match: {n_atoms == n_data_lines}")

# Phase values
phases_set = set(int(poly_preds_clean[i]) for i in range(n_atoms))
print(f"Phase values found: {sorted(phases_set)}")
valid = phases_set.issubset({-2, -1, 0, 1, 2})
print(f"Valid phase range: {valid}")

# Confidence range
print(f"Confidence range: [{conf.min():.4f}, {conf.max():.4f}]")
print(f"All in [0,1]: {conf.min() >= 0 and conf.max() <= 1}")

# File size
file_size = os.path.getsize(output_dump)
print(f"File size: {file_size/1e6:.2f} MB")

# =============================================================================
# Step 4: Spatial validation
# =============================================================================
print("\n" + "="*60)
print(f"STEP 4: SPATIAL VALIDATION AT THRESHOLD {confidence_threshold}")
print("="*60)

total = n_atoms
for pl, pn in [(0, 'Cu50Zr50'), (1, 'Cu46Zr54'), (2, 'Cu64Zr36'), (-1, 'Uncertain')]:
    mask = poly_preds_clean == pl
    cnt = mask.sum()
    if cnt == 0:
        print(f"{pn:15s}: 0 atoms")
        continue
    xv = df_pred.loc[mask, 'x'].values
    print(f"{pn:15s}: {cnt:5d} atoms ({cnt/total*100:5.1f}%), "
          f"x_mean={xv.mean():.1f}, x_std={xv.std():.1f}, "
          f"x_min={xv.min():.1f}, x_max={xv.max():.1f}")

c1 = (poly_preds_clean == 1).sum()
print(f"\nClass 1 FP at threshold {confidence_threshold}: {c1} ({c1/total*100:.1f}%)")
if c1/total < 0.05:
    print("✅ Below 5% — pipeline scientifically complete")
else:
    print("❌ Above 5%")

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
print(f"  - 3-class: Machine_Learning_Pipeline_for_Materials_Science/outputs/models/model_3class.pkl")
print(f"  - binary:  Machine_Learning_Pipeline_for_Materials_Science/outputs/models/model_binary_0_2.pkl")
print(f"\nOutput files generated:")
print(f"  - Machine_Learning_Pipeline_for_Materials_Science/outputs/polyamorphous_phase_labeled.lammpstrj")
print(f"  - Machine_Learning_Pipeline_for_Materials_Science/outputs/polyamorphous_phase_predictions.csv")

print("\nDone.")