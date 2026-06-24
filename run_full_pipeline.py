"""
Run full pipeline: rebuild feature tables, train binary model, run diagnostics.
"""
import sys, os, pandas as pd, numpy as np, joblib, yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

sys.path.insert(0, 'Machine_Learning_Pipeline_for_Materials_Science/src')
from feature_builder_data_clean import build_ml_table
from data_utils import load_and_label

with open('Machine_Learning_Pipeline_for_Materials_Science/config.yaml') as f:
    cfg = yaml.safe_load(f)
atomic_radii = cfg.get('atomic_radii')

# Step 1: Rebuild all feature tables
print("="*60)
print("Rebuilding feature tables...")
print("="*60)
for ds in ['5050', '4654', '6436', 'polyamorphous']:
    in_path = f'Atomic_Simulation_Post-Processing_Pipeline/outputs/features_{ds}.csv'
    out_path = f'Machine_Learning_Pipeline_for_Materials_Science/data/features_{ds}.csv'
    df = build_ml_table(in_path, tolerance=0.2, atomic_radii=atomic_radii)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    has_new = [c for c in df.columns if 's2' in c or 'isb' in c]
    print(f'  {ds}: {len(df)} rows, {len(df.columns)} cols, new: {has_new}')

# Step 2: Train binary model (class 0 vs class 2 only)
print()
print("="*60)
print("Training binary model: class 0 (5050) vs class 2 (6436)")
print("="*60)

df_0 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_5050.csv')
df_2 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_6436.csv')
df_0['phase_label'] = 0
df_2['phase_label'] = 2
df = pd.concat([df_0, df_2], ignore_index=True)

feature_cols = [c for c in cfg['features'] if c in df.columns]
print(f'Using {len(feature_cols)} features')

X = df[feature_cols].values
y = df['phase_label'].values

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# HistGB with CV
hgb = HistGradientBoostingClassifier(random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)
param_dist = {
    'max_iter': [100, 200, 300, 500],
    'learning_rate': [0.1, 0.05, 0.01],
    'max_leaf_nodes': [15, 31, 63],
    'min_samples_leaf': [10, 20, 30],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rs = RandomizedSearchCV(hgb, param_dist, n_iter=48, cv=cv, scoring='f1_macro', random_state=42, n_jobs=-1)
rs.fit(X_train, y_train)

best_model = rs.best_estimator_
print(f'Best CV F1: {rs.best_score_:.4f}')
y_pred = best_model.predict(X_test)
print(f'Test accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(classification_report(y_test, y_pred, target_names=['Cu50Zr50(0)', 'Cu64Zr36(2)']))
print(confusion_matrix(y_test, y_pred))

# Permutation importance
r = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
sorted_idx = r.importances_mean.argsort()[::-1]
print(f'\n{"Rank":>5s} {"Feature":40s} {"Importance":>10s} {"Std":>10s}')
print('-' * 67)
for rank, i in enumerate(sorted_idx[:20], 1):
    print(f'{rank:5d} {feature_cols[i]:40s} {r.importances_mean[i]:10.4f} {r.importances_std[i]:10.4f}')

# Save model
joblib.dump(best_model, 'Machine_Learning_Pipeline_for_Materials_Science/outputs/models/model_binary_0_2.pkl')
print('\nModel saved to outputs/models/model_binary_0_2.pkl')

# Step 3: Apply to polyamorphous
print()
print("="*60)
print("Applying binary model to polyamorphous data")
print("="*60)

df_poly = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_polyamorphous.csv')
X_poly = df_poly[feature_cols].values
probs = best_model.predict_proba(X_poly)
preds = best_model.predict(X_poly)
confidence = np.max(probs, axis=1)

# Load dump for x coords
dump_path = 'Atomic_Simulation_Post-Processing_Pipeline/files/snapshots_polyamorphous/combined_final_dump.lammpstrj'
with open(dump_path, 'r') as f:
    lines = f.readlines()
header = lines[8].strip().replace('ITEM: ATOMS', '').strip()
cols = header.split()
atom_data = [line.strip().split() for line in lines[9:] if line.strip()]
atoms = pd.DataFrame(atom_data, columns=cols)
atoms['id'] = atoms['id'].astype(int)
atoms['x'] = atoms['x'].astype(float)
atoms_sorted = atoms.sort_values('id').reset_index(drop=True)

df_poly_sorted = df_poly.sort_values('id').reset_index(drop=True)

# Threshold sweep
print(f'\n{"Thresh":>7s} {"Class 0":>30s} {"Class 2":>30s} {"Uncertain":>12s}')
print(f'{"":7s} {"count":>6s} {"pct":>5s} {"x_mean":>7s} {"x_std":>7s} {"count":>6s} {"pct":>5s} {"x_mean":>7s} {"x_std":>7s} {"count":>6s} {"pct":>5s}')
print('-' * 86)

for thresh in [0.50, 0.60, 0.70, 0.80, 0.90, 0.99]:
    phase = np.where(confidence >= thresh, preds, -1)
    total = len(phase)
    
    c0 = phase == 0
    c2 = phase == 2
    unc = phase == -1
    
    c0_x = atoms_sorted.loc[c0, 'x'].values if c0.sum() > 0 else []
    c2_x = atoms_sorted.loc[c2, 'x'].values if c2.sum() > 0 else []
    
    def fmt(arr):
        if len(arr) > 0:
            return f'{len(arr):6d} {len(arr)/total*100:5.1f}% {np.mean(arr):7.1f} {np.std(arr):7.1f}'
        return f'{0:6d} {"0.0%":>5s} {"-":>7s} {"-":>7s}'
    
    print(f'{thresh:5.2f}   {fmt(c0_x)}   {fmt(c2_x)}   {unc.sum():6d} {unc.sum()/total*100:5.1f}%')

# S2 distribution check
print()
print("="*60)
print("S2 entropy distribution by class")
print("="*60)
for label, csv_path in [('5050 (class 0)', 'data/features_5050.csv'), ('6436 (class 2)', 'data/features_6436.csv'), ('POLY', 'data/features_polyamorphous.csv')]:
    d = pd.read_csv(f'Machine_Learning_Pipeline_for_Materials_Science/{csv_path}')
    s2 = d['s2_entropy_avg_temporal'].dropna()
    print(f'{label:20s}: mean={s2.mean():.4f}, std={s2.std():.4f}, min={s2.min():.4f}, max={s2.max():.4f}')