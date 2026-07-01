"""Final retrain with FIXED MRO features."""
import sys, os, pandas as pd, numpy as np, joblib, yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

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
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    has_mro = [c for c in df.columns if 'mean_neighbor' in c]
    print(f'{ds}: {len(df)} rows, {len(df.columns)} cols, MRO: {len(has_mro)} features')

# Train binary model
print('\n' + '='*60)
print('Binary model: 5050(0) vs 6436(2) — WITH FIXED MRO')
print('='*60)

df_0 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_5050.csv')
df_2 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_6436.csv')
df_0['phase_label'] = 0
df_2['phase_label'] = 2
df = pd.concat([df_0, df_2], ignore_index=True)

feature_cols = [c for c in cfg['features'] if c in df.columns]
print(f'Using {len(feature_cols)} features')

X = df[feature_cols].values
y = df['phase_label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

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
mro_features = ['mean_neighbor_volume_temporal','std_neighbor_volume_temporal','mean_neighbor_pentagon_fraction_temporal','std_neighbor_pentagon_fraction_temporal','mean_neighbor_CN_temporal','mean_neighbor_free_volume_temporal','mean_neighbor_asphericity_temporal']
for rank, i in enumerate(sorted_idx[:25], 1):
    fn = feature_cols[i]
    imp = r.importances_mean[i]
    tag = ' <<< MRO' if fn in mro_features else ''
    print(f'{rank:5d} {fn:40s} {imp:10.4f} {r.importances_std[i]:10.4f}{tag}')

# Save
joblib.dump(best_model, 'Machine_Learning_Pipeline_for_Materials_Science/outputs/models/model_binary_0_2.pkl')
print(f'\nModel saved')