"""
Train binary model 0 vs 2 and apply to polyamorphous.
"""
import pandas as pd, numpy as np, joblib, yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import os

# Load data
df_0 = pd.read_csv('data/features_5050.csv')
df_2 = pd.read_csv('data/features_6436.csv')
df_0['phase_label'] = 0
df_2['phase_label'] = 2
df = pd.concat([df_0, df_2], ignore_index=True)

with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
feature_cols = [c for c in cfg['features'] if c in df.columns]
print(f'Using {len(feature_cols)} features')

X = df[feature_cols].values
y = df['phase_label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

hgb = HistGradientBoostingClassifier(random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)
param_dist = {'max_iter': [100,200,300,500], 'learning_rate': [0.1,0.05,0.01], 'max_leaf_nodes': [15,31,63], 'min_samples_leaf': [10,20,30]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rs = RandomizedSearchCV(hgb, param_dist, n_iter=48, cv=cv, scoring='f1_macro', random_state=42, n_jobs=-1)
rs.fit(X_train, y_train)

best = rs.best_estimator_
print(f'Best CV F1: {rs.best_score_:.4f}')
y_pred = best.predict(X_test)
print(f'Test accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(classification_report(y_test, y_pred, target_names=['Cu50Zr50(0)', 'Cu64Zr36(2)']))
print(confusion_matrix(y_test, y_pred))

# Permutation importance
r = permutation_importance(best, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
sorted_idx = r.importances_mean.argsort()[::-1]
print(f'\n{"Rank":>5s} {"Feature":40s} {"Importance":>10s} {"Std":>10s}')
print('-' * 67)
for rank, i in enumerate(sorted_idx[:20], 1):
    print(f'{rank:5d} {feature_cols[i]:40s} {r.importances_mean[i]:10.4f} {r.importances_std[i]:10.4f}')

# S2 check
print('\nS2 entropy by class:')
for label, csv in [('5050 (c0)','data/features_5050.csv'), ('6436 (c2)','data/features_6436.csv')]:
    d = pd.read_csv(csv)
    s2 = d['s2_entropy_avg_temporal'].dropna()
    print(f'  {label:15s}: mean={s2.mean():.4f}, std={s2.std():.4f}')

# Save model
os.makedirs('outputs/models', exist_ok=True)
joblib.dump(best, 'outputs/models/model_binary_0_2.pkl')
print('\nModel saved to outputs/models/model_binary_0_2.pkl')

# --- Apply to polyamorphous ---
print('\n' + '='*70)
print('DIAGNOSTIC D: Threshold Sweep (Binary Model 0 vs 2)')
print('='*70)

df_poly = pd.read_csv('data/features_polyamorphous.csv')
X_poly = df_poly[feature_cols].values
probs = best.predict_proba(X_poly)
preds = best.predict(X_poly)
confidence = np.max(probs, axis=1)

dump_path = '../Atomic_Simulation_Post-Processing_Pipeline/files/snapshots_polyamorphous/combined_final_dump.lammpstrj'
with open(dump_path, 'r') as f:
    lines = f.readlines()
header = lines[8].strip().replace('ITEM: ATOMS', '').strip()
cols = header.split()
atom_data = [line.strip().split() for line in lines[9:] if line.strip()]
atoms = pd.DataFrame(atom_data, columns=cols)
atoms['id'] = atoms['id'].astype(int)
atoms['x'] = atoms['x'].astype(float)
atoms_sorted = atoms.sort_values('id').reset_index(drop=True)

print(f'{"Thresh":>7s} {"Class 0 (Cu50Zr50)":>30s} {"Class 2 (Cu64Zr36)":>30s} {"Uncertain":>12s}')
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