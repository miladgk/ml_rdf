"""3-class retrain with MRO — fast version without grid search."""
import sys, os, pandas as pd, numpy as np, joblib, yaml
sys.path.insert(0, 'Machine_Learning_Pipeline_for_Materials_Science/src')
from feature_builder_data_clean import build_ml_table

with open('Machine_Learning_Pipeline_for_Materials_Science/config.yaml') as f:
    cfg = yaml.safe_load(f)

# Rebuild ML tables
for ds in ['5050', '4654', '6436', 'polyamorphous']:
    in_path = f'Atomic_Simulation_Post-Processing_Pipeline/outputs/features_{ds}.csv'
    out_path = f'Machine_Learning_Pipeline_for_Materials_Science/data/features_{ds}.csv'
    df = build_ml_table(in_path, tolerance=0.2, atomic_radii=cfg.get('atomic_radii'))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'{ds}: {len(df)} rows, {len(df.columns)} cols')

df_5050 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_5050.csv')
df_4654 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_4654.csv')
df_6436 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_6436.csv')
df_poly = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_polyamorphous.csv')

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

# ====== Step 1: 3-class model (fixed params, 5-fold CV) ======
print('\n' + '='*60)
print('3-CLASS MODEL: 5050(0) vs 4654(1) vs 6436(2) — WITH MRO')
print('='*60)

feature_cols = [c for c in cfg['features'] if c in df_5050.columns]
df_labelled = pd.concat([
    df_5050.assign(phase_label=0),
    df_4654.assign(phase_label=1),
    df_6436.assign(phase_label=2)
], ignore_index=True)

print(f'Features: {len(feature_cols)}, Samples: {len(df_labelled)}')

X = df_labelled[feature_cols].values
y = df_labelled['phase_label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Best params from binary model, retrained on 3-class
model = HistGradientBoostingClassifier(
    random_state=42, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=10, max_iter=500, learning_rate=0.05,
    max_leaf_nodes=31, min_samples_leaf=20
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
print(f'CV F1 macro: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f'Test accuracy: {accuracy_score(y_test, y_pred):.4f}')

print('\nPer-class F1:')
report = classification_report(y_test, y_pred, target_names=['Cu50Zr50(0)', 'Cu46Zr54(1)', 'Cu64Zr36(2)'], output_dict=True)
for c in ['Cu50Zr50(0)', 'Cu46Zr54(1)', 'Cu64Zr36(2)']:
    print(f'  {c}: {report[c]["f1-score"]:.4f}')

print('\nConfusion matrix (3x3):')
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=['True 0', 'True 1', 'True 2'], columns=['Pred 0', 'Pred 1', 'Pred 2']))

# Permutation importance top 15
r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=4)
sorted_idx = r.importances_mean.argsort()[::-1]
mro_set = {'mean_neighbor_volume_temporal','std_neighbor_volume_temporal','mean_neighbor_pentagon_fraction_temporal',
           'std_neighbor_pentagon_fraction_temporal','mean_neighbor_CN_temporal','mean_neighbor_free_volume_temporal',
           'mean_neighbor_asphericity_temporal'}
print(f'\n{"Rank":>5s} {"Feature":40s} {"Importance":>10s}')
print('-' * 57)
for rank, i in enumerate(sorted_idx[:15], 1):
    fn = feature_cols[i]
    tag = ' <<< MRO' if fn in mro_set else ''
    print(f'{rank:5d} {fn:40s} {r.importances_mean[i]:10.4f}{tag}')

joblib.dump(model, 'Machine_Learning_Pipeline_for_Materials_Science/outputs/models/model_3class.pkl')
print('\nModel saved')

# ====== Step 2: MRO volume by phase and atom type ======
print('\n' + '='*60)
print('mean_neighbor_volume_temporal by phase and atom type')
print('='*60)
for phase_name, df in [('5050', df_5050), ('4654', df_4654), ('6436', df_6436)]:
    col = 'mean_neighbor_volume_temporal'
    print(f'\n=== {phase_name} ===')
    print(f'All: mean={df[col].mean():.4f}, std={df[col].std():.4f}')
    for t, tname in [(1, 'Cu'), (2, 'Zr')]:
        sub = df[df['type'] == t]
        print(f'  {tname}: mean={sub[col].mean():.4f}, std={sub[col].std():.4f}')

# ====== Step 3: 5050 vs 4654 overlap ======
print('\n' + '='*60)
print('Overlap: 5050 vs 4654 in mean_neighbor_volume_temporal')
print('='*60)
v0 = df_5050['mean_neighbor_volume_temporal']
v1 = df_4654['mean_neighbor_volume_temporal']
q5, q95 = v1.quantile(0.05), v1.quantile(0.95)
overlap = ((v0 >= q5) & (v0 <= q95)).mean()
print(f'5050 mean: {v0.mean():.4f} (std={v0.std():.4f})')
print(f'4654 mean: {v1.mean():.4f} (std={v1.std():.4f})')
print(f'Difference in means: {abs(v0.mean() - v1.mean()):.4f}')
print(f'90% range of 4654: [{q5:.4f}, {q95:.4f}]')
print(f'Fraction of 5050 atoms within 90% range of 4654: {overlap:.3f}')
if overlap > 0.5:
    print('>>> HIGH OVERLAP (>0.5): MRO alone NOT sufficient for 5050 vs 4654')
    print('>>> Second-level MRO (neighbors of neighbors) likely needed')
else:
    print('>>> LOW OVERLAP (<0.3): MRO provides separation')

# ====== Step 4: 3-class prediction on polyamorphous ======
print('\n' + '='*60)
print('3-CLASS PREDICTION ON POLYAMORPHOUS')
print('='*60)

X_poly = df_poly[feature_cols].values
probs = model.predict_proba(X_poly)
preds = model.predict(X_poly)
confidence = np.max(probs, axis=1)

with open('Atomic_Simulation_Post-Processing_Pipeline/files/snapshots_polyamorphous/combined_final_dump.lammpstrj') as f:
    lines = f.readlines()
cols = lines[8].strip().replace('ITEM: ATOMS', '').strip().split()
atoms = pd.DataFrame([l.strip().split() for l in lines[9:] if l.strip()], columns=cols)
atoms['id'] = atoms['id'].astype(int)
atoms['x'] = atoms['x'].astype(float)
atoms_sorted = atoms.sort_values('id').reset_index(drop=True)

print(f'\n{"Thresh":>7s} {"Class 0":>50s} {"Class 1":>50s} {"Class 2":>50s} {"Uncert":>10s}')
for thresh in [0.50, 0.60, 0.70, 0.80, 0.90]:
    phase = np.where(confidence >= thresh, preds, -1)
    total = len(phase)
    def _fmt(mask):
        if mask.sum() == 0:
            return f'{0:6d} {"0.0%":>5s} {"-":>7s} {"-":>7s}'
        x = atoms_sorted.loc[mask, 'x'].values
        return f'{mask.sum():6d} {mask.sum()/total*100:5.1f}% {np.mean(x):7.1f} {np.std(x):7.1f}'
    print(f'{thresh:.2f}   {_fmt(phase==0)}   {_fmt(phase==1)}   {_fmt(phase==2)}   {_fmt(phase==-1)}')
    if thresh in [0.60, 0.90]:
        print(f'         Class 1 false positives at {thresh}: {(phase==1).sum()}')

print('\nDone.')