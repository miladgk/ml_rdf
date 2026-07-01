"""3-class retrain with MRO features — the Cu46Zr54 separability test."""
import sys, os, pandas as pd, numpy as np, joblib, yaml

sys.path.insert(0, 'Machine_Learning_Pipeline_for_Materials_Science/src')
from feature_builder_data_clean import build_ml_table

with open('Machine_Learning_Pipeline_for_Materials_Science/config.yaml') as f:
    cfg = yaml.safe_load(f)
atomic_radii = cfg.get('atomic_radii')

# Rebuild ML tables from raw outputs
for ds in ['5050', '4654', '6436', 'polyamorphous']:
    in_path = f'Atomic_Simulation_Post-Processing_Pipeline/outputs/features_{ds}.csv'
    out_path = f'Machine_Learning_Pipeline_for_Materials_Science/data/features_{ds}.csv'
    df = build_ml_table(in_path, tolerance=0.2, atomic_radii=atomic_radii)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'{ds}: {len(df)} rows, {len(df.columns)} cols')

# Load data
df_5050 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_5050.csv')
df_4654 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_4654.csv')
df_6436 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_6436.csv')
df_poly = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_polyamorphous.csv')

df_5050['phase_label'] = 0
df_4654['phase_label'] = 1
df_6436['phase_label'] = 2
df_labelled = pd.concat([df_5050, df_4654, df_6436], ignore_index=True)

feature_cols = [c for c in cfg['features'] if c in df_labelled.columns]
print(f'\nUsing {len(feature_cols)} features')

# =============================================================================
# Step 1: 3-class model
# =============================================================================
print('\n' + '='*60)
print('3-CLASS MODEL: 5050(0) vs 4654(1) vs 6436(2) — WITH MRO')
print('='*60)

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

X = df_labelled[feature_cols].values
y = df_labelled['phase_label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

hgb = HistGradientBoostingClassifier(random_state=42, early_stopping=True,
                                     validation_fraction=0.1, n_iter_no_change=10)
param_dist = {
    'max_iter': [100, 200, 300, 500],
    'learning_rate': [0.1, 0.05, 0.01],
    'max_leaf_nodes': [15, 31, 63],
    'min_samples_leaf': [10, 20, 30],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rs = RandomizedSearchCV(hgb, param_dist, n_iter=48, cv=cv, scoring='f1_macro', 
                         random_state=42, n_jobs=-1)
rs.fit(X_train, y_train)
best = rs.best_estimator_

print(f'Best CV F1 (macro): {rs.best_score_:.4f}')

y_pred = best.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {test_acc:.4f}')

print('\nPer-class F1:')
report = classification_report(y_test, y_pred, target_names=['Cu50Zr50(0)', 'Cu46Zr54(1)', 'Cu64Zr36(2)'],
                                output_dict=True)
for c in ['Cu50Zr50(0)', 'Cu46Zr54(1)', 'Cu64Zr36(2)']:
    print(f'  {c}: {report[c]["f1-score"]:.4f}')

print(f'\nConfusion matrix (3x3):')
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=['True 0', 'True 1', 'True 2'],
                    columns=['Pred 0', 'Pred 1', 'Pred 2']).to_string())

# Permutation importance (top 15)
r = permutation_importance(best, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
sorted_idx = r.importances_mean.argsort()[::-1]
print(f'\n{"Rank":>5s} {"Feature":40s} {"Importance":>10s}')
print('-' * 57)
mro_features = ['mean_neighbor_volume_temporal','std_neighbor_volume_temporal',
                'mean_neighbor_pentagon_fraction_temporal','std_neighbor_pentagon_fraction_temporal',
                'mean_neighbor_CN_temporal','mean_neighbor_free_volume_temporal',
                'mean_neighbor_asphericity_temporal']
for rank, i in enumerate(sorted_idx[:15], 1):
    fn = feature_cols[i]
    tag = ' <<< MRO' if fn in mro_features else ''
    print(f'{rank:5d} {fn:40s} {r.importances_mean[i]:10.4f}{tag}')

# Save
model_path = 'Machine_Learning_Pipeline_for_Materials_Science/outputs/models/model_3class.pkl'
joblib.dump(best, model_path)
print(f'\nModel saved to {model_path}')

# =============================================================================
# Step 2: MRO volume by phase and atom type
# =============================================================================
print('\n' + '='*60)
print('mean_neighbor_volume_temporal by phase and atom type')
print('='*60)

for phase_name, df in [('5050', df_5050), ('4654', df_4654), ('6436', df_6436)]:
    col = 'mean_neighbor_volume_temporal'
    print(f'\n=== {phase_name} ===')
    print(f'All atoms: mean={df[col].mean():.4f}, std={df[col].std():.4f}')
    cu = df[df['type'] == 1]
    zr = df[df['type'] == 2]
    print(f'Cu atoms:  mean={cu[col].mean():.4f}, std={cu[col].std():.4f}')
    print(f'Zr atoms:  mean={zr[col].mean():.4f}, std={zr[col].std():.4f}')

# =============================================================================
# Step 3: 5050 vs 4654 overlap
# =============================================================================
print('\n' + '='*60)
print('Overlap: 5050 vs 4654 in mean_neighbor_volume_temporal')
print('='*60)

v5050 = df_5050['mean_neighbor_volume_temporal']
v4654 = df_4654['mean_neighbor_volume_temporal']
q5_4654 = v4654.quantile(0.05)
q95_4654 = v4654.quantile(0.95)
overlap = ((v5050 >= q5_4654) & (v5050 <= q95_4654)).mean()

print(f'4654 mean_neighbor_volume: {v4654.mean():.4f} (std={v4654.std():.4f})')
print(f'5050 mean_neighbor_volume: {v5050.mean():.4f} (std={v5050.std():.4f})')
print(f'Difference in means: {abs(v4654.mean() - v5050.mean()):.4f}')
print(f'90% range of 4654: [{q5_4654:.4f}, {q95_4654:.4f}]')
print(f'Fraction of 5050 atoms within 90% range of 4654: {overlap:.3f}')
if overlap > 0.5:
    print('>>> HIGH OVERLAP: MRO alone not sufficient for 5050 vs 4654')
else:
    print('>>> LOW OVERLAP: MRO provides separation')

# =============================================================================
# Step 4: Apply 3-class model to polyamorphous
# =============================================================================
print('\n' + '='*60)
print('3-CLASS PREDICTION ON POLYAMORPHOUS DATA')
print('='*60)

X_poly = df_poly[feature_cols].values
probs = best.predict_proba(X_poly)
preds = best.predict(X_poly)
confidence = np.max(probs, axis=1)

# Load x-positions from dump
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

print(f'\n{"Thresh":>7s} {"Class 0":>50s} {"Class 1":>50s} {"Class 2":>50s} {"Uncertain":>12s}')
print(f'{"":7s} {"count":>6s} {"pct":>5s} {"x_mean":>7s} {"x_std":>7s} ', end='')
print(f'{"count":>6s} {"pct":>5s} {"x_mean":>7s} {"x_std":>7s} ', end='')
print(f'{"count":>6s} {"pct":>5s} {"x_mean":>7s} {"x_std":>7s} {"count":>6s} {"pct":>5s}')
print('-' * 120)

for thresh in [0.50, 0.60, 0.70, 0.80, 0.90]:
    phase = np.where(confidence >= thresh, preds, -1)
    total = len(phase)
    
    def fmt_class(mask):
        if mask.sum() > 0:
            xvals = atoms_sorted.loc[mask, 'x'].values
            return f'{mask.sum():6d} {mask.sum()/total*100:5.1f}% {np.mean(xvals):7.1f} {np.std(xvals):7.1f}'
        return f'{0:6d} {"0.0%":>5s} {"-":>7s} {"-":>7s}'
    
    c0 = phase == 0
    c1 = phase == 1
    c2 = phase == 2
    unc = phase == -1
    
    print(f'{thresh:5.2f}   {fmt_class(c0)}   {fmt_class(c1)}   {fmt_class(c2)}   {unc.sum():6d} {unc.sum()/total*100:5.1f}%')
    
    if thresh in [0.60, 0.90]:
        print(f'         Class 1 false positives at {thresh:.2f}: {c1.sum():d}')

print('\nDone.')