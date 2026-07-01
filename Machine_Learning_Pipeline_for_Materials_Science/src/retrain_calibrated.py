"""
retrain_calibrated.py
=====================
Retrain the 3-class model with improvements to fix polyamorphous classification:

1. Probability calibration via CalibratedClassifierCV
2. Class weighting to reduce bias toward majority-like class
3. Feature importance analysis with domain-shift awareness
4. Apply to polyamorphous with spatial validation
5. Generate labeled LAMMPS dumps at multiple thresholds

Key insight from diagnostics:
- Per-atom local features (voronoi_volume, R1, pentagon_fraction, q4, q6) are RELIABLE
  (preserve their signal in the polyamorphous system)
- Composition-dependent features (neighbor_1, neighbor_2, neighbor_1_fraction) have DOMAIN SHIFT
  (the mixed neighborhoods change these features)
- MRO features (mean_neighbor_*) shift moderately — 2nd-shell averages get diluted
"""
import sys, os, warnings
import pandas as pd
import numpy as np
import joblib
import yaml

warnings.filterwarnings('ignore')
sys.path.insert(0, 'Machine_Learning_Pipeline_for_Materials_Science/src')
from feature_builder_data_clean import build_ml_table

with open('Machine_Learning_Pipeline_for_Materials_Science/config.yaml') as f:
    cfg = yaml.safe_load(f)

# ============================================================
# Step 0: Load data
# ============================================================
print('Loading data...')
df_5050 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_5050.csv')
df_4654 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_4654.csv')
df_6436 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_6436.csv')
df_poly = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_polyamorphous.csv')

for name, df in [('5050', df_5050), ('4654', df_4654), ('6436', df_6436), ('poly', df_poly)]:
    print(f'  {name}: {len(df)} atoms, {len(df.columns)} cols')

# ============================================================
# Step 1: Feature selection — prioritize domain-robust features
# ============================================================
feature_cols = [c for c in cfg['features'] if c in df_5050.columns]

# Categorize features by domain shift behavior
GOOD_features = [  # <3% domain shift — most reliable
    'voronoi_volume_temporal', 'R1', 'pentagon_fraction_temporal',
    'q4', 'q6', 'CN_temporal', 'asphericity_temporal',
    'q4_avg', 'q6_avg', 'w4', 'w6',
    'n3_temporal', 'n4_temporal', 'n5_temporal', 'n6_temporal',
    'asphericity_std_temporal',
]

OK_features = [  # 3-10% domain shift — usable with caution
    'free_volume_temporal',
    'mean_neighbor_volume_temporal', 'std_neighbor_volume_temporal',
    'mean_2nd_neighbor_volume_temporal', 'std_2nd_neighbor_volume_temporal',
    'mean_neighbor_free_volume_temporal',
    'mean_2nd_neighbor_free_volume_temporal',
    'mean_neighbor_pentagon_fraction_temporal', 'std_neighbor_pentagon_fraction_temporal',
    'mean_neighbor_CN_temporal', 'mean_neighbor_asphericity_temporal',
    'mean_2nd_neighbor_pentagon_fraction_temporal', 'mean_2nd_neighbor_CN_temporal',
]

BAD_features = [  # >10% domain shift — unreliable in polyamorphous
    'neighbor_1_fraction_temporal', 'neighbor_1', 'neighbor_2',
]

# Use all features for training but the model will learn to weight them
print(f'\nFeatures: {len(feature_cols)} total')
print(f'  GOOD (domain-robust): {sum(1 for f in GOOD_features if f in feature_cols)}')
print(f'  OK (moderate shift): {sum(1 for f in OK_features if f in feature_cols)}')
print(f'  BAD (high shift): {sum(1 for f in BAD_features if f in feature_cols)}')

# ============================================================
# Step 2: Train model with class weighting + calibration
# ============================================================
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.inspection import permutation_importance

# Prepare data
df_5050['phase_label'] = 0
df_4654['phase_label'] = 1
df_6436['phase_label'] = 2
df_labelled = pd.concat([df_5050, df_4654, df_6436], ignore_index=True)

X = df_labelled[feature_cols].values
y = df_labelled['phase_label'].values

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print('\n' + '='*70)
print('MODEL A: Standard HistGradientBoosting (baseline)')
print('='*70)

model_baseline = HistGradientBoostingClassifier(
    random_state=42, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=10, max_iter=500, learning_rate=0.05,
    max_leaf_nodes=31, min_samples_leaf=20
)
model_baseline.fit(X_train, y_train)
y_pred_base = model_baseline.predict(X_test)
print(f'Test accuracy: {accuracy_score(y_test, y_pred_base):.4f}')
print(f'F1 macro:      {f1_score(y_test, y_pred_base, average="macro"):.4f}')
print('Confusion matrix:')
cm = confusion_matrix(y_test, y_pred_base)
print(pd.DataFrame(cm, index=['True 5050', 'True 4654', 'True 6436'],
                   columns=['Pred 5050', 'Pred 4654', 'Pred 6436']))

# Check calibration
probs_base = model_baseline.predict_proba(X_test)
conf_base = np.max(probs_base, axis=1)
print(f'\nMean confidence (all):     {conf_base.mean():.4f}')
print(f'Mean confidence (correct): {conf_base[y_pred_base == y_test].mean():.4f}')
print(f'Mean confidence (wrong):   {conf_base[y_pred_base != y_test].mean():.4f}')

print('\n' + '='*70)
print('MODEL B: Class-weighted HistGradientBoosting')
print('='*70)

# Compute sample weights to upweight the harder class (4654)
# 4654 is harder to separate from 5050, so give it more weight
from sklearn.utils.class_weight import compute_sample_weight

sample_weights = compute_sample_weight('balanced', y_train)
# Further boost 4654 (class 1)
# Since 4654 is hardest, give it 2x the balanced weight
boost_mask = y_train == 1
sample_weights[boost_mask] *= 2.0

model_weighted = HistGradientBoostingClassifier(
    random_state=42, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=10, max_iter=500, learning_rate=0.05,
    max_leaf_nodes=31, min_samples_leaf=20
)
model_weighted.fit(X_train, y_train, sample_weight=sample_weights)
y_pred_w = model_weighted.predict(X_test)
print(f'Test accuracy: {accuracy_score(y_test, y_pred_w):.4f}')
print(f'F1 macro:      {f1_score(y_test, y_pred_w, average="macro"):.4f}')
print('Confusion matrix:')
cm_w = confusion_matrix(y_test, y_pred_w)
print(pd.DataFrame(cm_w, index=['True 5050', 'True 4654', 'True 6436'],
                   columns=['Pred 5050', 'Pred 4654', 'Pred 6436']))

print('\n' + '='*70)
print('MODEL C: Calibrated + Class-weighted')
print('='*70)

# Train calibrated model using isotonic regression
# CalibratedClassifierCV will refit with CV and calibrate probabilities
calibrated_model = CalibratedClassifierCV(
    estimator=HistGradientBoostingClassifier(
        random_state=42, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=10, max_iter=500, learning_rate=0.05,
        max_leaf_nodes=31, min_samples_leaf=20
    ),
    method='isotonic', cv=5
)

# For CalibratedClassifierCV, we need to use fit with sample_weight
calibrated_model.fit(X_train, y_train, sample_weight=sample_weights)
y_pred_cal = calibrated_model.predict(X_test)
probs_cal = calibrated_model.predict_proba(X_test)
conf_cal = np.max(probs_cal, axis=1)

print(f'Test accuracy: {accuracy_score(y_test, y_pred_cal):.4f}')
print(f'F1 macro:      {f1_score(y_test, y_pred_cal, average="macro"):.4f}')
print('Confusion matrix:')
cm_cal = confusion_matrix(y_test, y_pred_cal)
print(pd.DataFrame(cm_cal, index=['True 5050', 'True 4654', 'True 6436'],
                   columns=['Pred 5050', 'Pred 4654', 'Pred 6436']))

print(f'\nCalibrated Mean confidence (all):     {conf_cal.mean():.4f}')
print(f'Calibrated Mean confidence (correct): {conf_cal[y_pred_cal == y_test].mean():.4f}')
print(f'Calibrated Mean confidence (wrong):   {conf_cal[y_pred_cal != y_test].mean():.4f}')

# Compare calibration quality
print('\n' + '='*70)
print('CALIBRATION COMPARISON')
print('='*70)
print(f'{"Metric":<35s} {"Baseline":>12s} {"Calibrated":>12s}')
print('-'*60)
print(f'{"Mean confidence (all)":<35s} {conf_base.mean():12.4f} {conf_cal.mean():12.4f}')
print(f'{"Mean confidence (correct)":<35s} {conf_base[y_pred_base==y_test].mean():12.4f} {conf_cal[y_pred_cal==y_test].mean():12.4f}')
print(f'{"Mean confidence (wrong)":<35s} {conf_base[y_pred_base!=y_test].mean():12.4f} {conf_cal[y_pred_cal!=y_test].mean():12.4f}')
print(f'{"Confidence gap":<35s} {conf_base[y_pred_base==y_test].mean()-conf_base[y_pred_base!=y_test].mean():12.4f} {conf_cal[y_pred_cal==y_test].mean()-conf_cal[y_pred_cal!=y_test].mean():12.4f}')

# ============================================================
# Step 3: Apply all models to polyamorphous + spatial validation
# ============================================================
print('\n' + '='*70)
print('POLYAMORPHOUS PREDICTIONS — SPATIAL VALIDATION')
print('='*70)

X_poly = df_poly[feature_cols].values
df_poly_sorted = df_poly.sort_values('id').reset_index(drop=True)

# Load x-positions from original dump
dump_path = 'Atomic_Simulation_Post-Processing_Pipeline/files/snapshots_polyamorphous/combined_final_dump.lammpstrj'
with open(dump_path, 'r') as f:
    dump_lines = f.readlines()
cols_str = dump_lines[8].strip().replace('ITEM: ATOMS', '').strip()
orig_col_names = cols_str.split()
dump_data = [l.strip().split() for l in dump_lines[9:] if l.strip()]
atom_df = pd.DataFrame(dump_data, columns=orig_col_names)
atom_df['id'] = atom_df['id'].astype(int)
atom_df['x'] = atom_df['x'].astype(float)
atom_df = atom_df.sort_values('id').reset_index(drop=True)

models = {
    'Baseline': (model_baseline, model_baseline.predict_proba(X_poly), model_baseline.predict(X_poly)),
    'Weighted': (model_weighted, model_weighted.predict_proba(X_poly), model_weighted.predict(X_poly)),
    'Calibrated': (calibrated_model, calibrated_model.predict_proba(X_poly), calibrated_model.predict(X_poly)),
}

# Define expected phase regions based on Cu% composition profile
# x < 60: Cu46Zr54 (expected class 1)
# 60 < x < 119: Cu50Zr50 (expected class 0)  
# x > 119: Cu64Zr36 (expected class 2)
region_bins = [0, 60, 119, 200]
region_expected = {0: 1, 1: 0, 2: 2}  # region_idx -> expected_class
region_names = ['Cu46Zr54 (x<60)', 'Cu50Zr50 (60<x<119)', 'Cu64Zr36 (x>119)']

for model_name, (model, probs, preds) in models.items():
    conf = np.max(probs, axis=1)
    print(f'\n--- {model_name} ---')
    
    for thresh in [0.50, 0.60, 0.70, 0.80, 0.90]:
        phase = np.where(conf >= thresh, preds, -1)
        
        # Per-region analysis
        regions = pd.cut(atom_df['x'], bins=region_bins, labels=[0, 1, 2])
        
        print(f'\n  Threshold={thresh:.2f}:')
        for ridx, rname in enumerate(region_names):
            rmask = regions == ridx
            n = rmask.sum()
            expected_cls = region_expected[ridx]
            correct = (phase[rmask] == expected_cls).sum()
            c0 = (phase[rmask] == 0).sum()
            c1 = (phase[rmask] == 1).sum()
            c2 = (phase[rmask] == 2).sum()
            cu = (phase[rmask] == -1).sum()
            print(f'    {rname:<25s} N={n:5d} | C0={c0:5d} C1={c1:5d} C2={c2:5d} Unc={cu:5d} | '
                  f'Expected(C{expected_cls})={correct/n*100:5.1f}%')

# ============================================================
# Step 4: Save best model + generate LAMMPS dumps
# ============================================================
print('\n' + '='*70)
print('SAVING OUTPUTS')
print('='*70)

# Save calibrated model
model_path = 'Machine_Learning_Pipeline_for_Materials_Science/outputs/models/model_3class_calibrated.pkl'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(calibrated_model, model_path)
print(f'Saved calibrated model to {model_path}')

# Save weighted model too
model_path_w = 'Machine_Learning_Pipeline_for_Materials_Science/outputs/models/model_3class_weighted.pkl'
joblib.dump(model_weighted, model_path_w)
print(f'Saved weighted model to {model_path_w}')

# Generate LAMMPS dumps for calibrated model
probs_cal_poly = calibrated_model.predict_proba(X_poly)
preds_cal_poly = calibrated_model.predict(X_poly)
conf_cal_poly = np.max(probs_cal_poly, axis=1)

out_dir = 'Machine_Learning_Pipeline_for_Materials_Science/outputs'
for thresh in [0.50, 0.60, 0.70, 0.80, 0.90]:
    phase = np.where(conf_cal_poly >= thresh, preds_cal_poly, -1)
    
    # Write LAMMPS dump
    th_str = f"{thresh:.2f}"
    output_dump = os.path.join(out_dir, f'polyamorphous_calibrated_{th_str}.lammpstrj')
    with open(output_dump, 'w') as f:
        for line in dump_lines[:8]:
            f.write(line)
        new_header = "ITEM: ATOMS " + " ".join(orig_col_names) + " phase confidence\n"
        f.write(new_header)
        for i in range(len(dump_data)):
            orig_vals = dump_data[i]
            phase_val = int(phase[i])
            conf_val = f"{conf_cal_poly[i]:.4f}"
            f.write(" ".join(orig_vals) + f" {phase_val} {conf_val}\n")
    
    # Summary
    total = len(phase)
    c0 = (phase == 0).sum()
    c1 = (phase == 1).sum()
    c2 = (phase == 2).sum()
    unc = (phase == -1).sum()
    print(f'  {th_str}: C0={c0:6d}({c0/total*100:5.1f}%)  C1={c1:6d}({c1/total*100:5.1f}%)  '
          f'C2={c2:6d}({c2/total*100:5.1f}%)  Unc={unc:6d}({unc/total*100:5.1f}%)')
    
    # Also write CSV
    pred_csv = os.path.join(out_dir, f'polyamorphous_calibrated_predictions_{th_str}.csv')
    pred_df = pd.DataFrame({
        'atom_id': atom_df['id'],
        'x': atom_df['x'].values,
        'type': atom_df['type'] if 'type' in atom_df.columns else df_poly_sorted['type'],
        'prob_class0': probs_cal_poly[:, 0],
        'prob_class1': probs_cal_poly[:, 1],
        'prob_class2': probs_cal_poly[:, 2],
        'confidence': conf_cal_poly,
        'phase': phase,
    })
    pred_df.to_csv(pred_csv, index=False)

print('\nDone.')
