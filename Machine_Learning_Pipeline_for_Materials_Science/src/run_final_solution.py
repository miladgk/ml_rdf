import os
import sys
import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, 'Machine_Learning_Pipeline_for_Materials_Science/src')
from feature_builder_data_clean import build_ml_table

with open('Machine_Learning_Pipeline_for_Materials_Science/config.yaml') as f:
    cfg = yaml.safe_load(f)
atomic_radii = cfg.get('atomic_radii', {1: 1.28, 2: 1.6})

print("="*60)
print("Step 1: Rebuilding ML tables from fixed pipeline outputs")
print("="*60)

datasets = ['5050', '4654', '6436', 'polyamorphous']
for ds in datasets:
    in_path = f'Atomic_Simulation_Post-Processing_Pipeline/outputs/features_{ds}.csv'
    out_path = f'Machine_Learning_Pipeline_for_Materials_Science/data/features_{ds}.csv'
    if os.path.exists(in_path):
        df = build_ml_table(in_path, tolerance=0.2, atomic_radii=atomic_radii)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"  {ds}: {len(df)} rows, {len(df.columns)} cols")
    else:
        print(f"  WARNING: {in_path} missing")

print("\n" + "="*60)
print("Step 2: Training 3-Class Classifier (5050 vs 4654 vs 6436)")
print("="*60)

df_0 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_5050.csv') # Class 0
df_1 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_4654.csv') # Class 1
df_2 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_6436.csv') # Class 2

df_0['phase_label'] = 0
df_1['phase_label'] = 1
df_2['phase_label'] = 2

df_train = pd.concat([df_0, df_1, df_2], ignore_index=True)

feature_cols = [c for c in cfg['features'] if c in df_train.columns]
print(f"Training on {len(df_train)} atoms using {len(feature_cols)} features")

X_train = df_train[feature_cols].values
y_train = df_train['phase_label'].values

model = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
model.fit(X_train, y_train)
train_acc = accuracy_score(y_train, model.predict(X_train))
print(f"Training accuracy: {train_acc*100:.2f}%")

print("\n" + "="*60)
print("Step 3: Predicting on Polyamorphous Data")
print("="*60)

df_poly = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_polyamorphous.csv')
df_poly = df_poly.sort_values('id').reset_index(drop=True)

X_poly = df_poly[feature_cols].values
preds = model.predict(X_poly)
probs = model.predict_proba(X_poly)
df_poly['pred'] = preds
df_poly['conf'] = np.max(probs, axis=1)

# Ground truth assignment based on user's exact atom counts:
# 1..11664 -> Cu46Zr54 (Class 1)
# 11665..23328 -> Cu50Zr50 (Class 0)
# 23329..36828 -> Cu64Zr36 (Class 2)

df_poly['true_label'] = -1
df_poly.loc[0:11663, 'true_label'] = 1 # 4654
df_poly.loc[11664:23327, 'true_label'] = 0 # 5050
df_poly.loc[23328:36827, 'true_label'] = 2 # 6436

y_true = df_poly['true_label'].values
poly_acc = accuracy_score(y_true, preds)
print(f"Polyamorphous Overall Accuracy: {poly_acc*100:.2f}%\n")

target_names = ['Cu50Zr50 (Class 0)', 'Cu46Zr54 (Class 1)', 'Cu64Zr36 (Class 2)']
print(classification_report(y_true, preds, target_names=target_names, digits=4))

print("Confusion Matrix (Rows=True, Cols=Pred):")
cm = confusion_matrix(y_true, preds)
cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
print(cm_df)

df_poly.to_csv('Machine_Learning_Pipeline_for_Materials_Science/outputs/polyamorphous_fixed_predictions.csv', index=False)
print("\nSaved predictions to outputs/polyamorphous_fixed_predictions.csv")
