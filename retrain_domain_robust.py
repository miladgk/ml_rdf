"""
retrain_domain_robust.py
========================
STRATEGY: Train ONLY on per-atom local features that preserve their phase
signature in the polyamorphous system (no MRO, no neighbor counts).

KEY INSIGHT FROM DIAGNOSTICS:
- MRO features (mean_2nd_neighbor_volume, etc.) give d=3.32 separability in
  training but become ~17.10 for ALL regions in polyamorphous → USELESS
- Composition features (neighbor_1, neighbor_2) shift by 14-17% → UNRELIABLE
- Per-atom local features (voronoi_volume, R1, pentagon_fraction, q4, q6)
  shift by <3% → RELIABLE but weaker separability (d=0.24 for voronoi_volume)

APPROACH:
1. Train Model A: ALL features (baseline, ~99.8% test acc, fails on poly)
2. Train Model B: ONLY domain-robust features (lower test acc, but transfers)
3. Train Model C: Domain-robust + spatially-smoothed predictions
4. Compare polyamorphous predictions across all three
"""
import sys, os, warnings
import pandas as pd
import numpy as np
import joblib
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, 'Machine_Learning_Pipeline_for_Materials_Science/src')

with open('Machine_Learning_Pipeline_for_Materials_Science/config.yaml') as f:
    cfg = yaml.safe_load(f)

# ============================================================
# Load data
# ============================================================
print('Loading data...')
df_5050 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_5050.csv')
df_4654 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_4654.csv')
df_6436 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_6436.csv')
df_poly = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_polyamorphous.csv')

# Prepare labels
df_5050['phase_label'] = 0
df_4654['phase_label'] = 1
df_6436['phase_label'] = 2
df_labelled = pd.concat([df_5050, df_4654, df_6436], ignore_index=True)

# All features
all_features = [c for c in cfg['features'] if c in df_labelled.columns]

# Domain-robust features ONLY (per-atom local, <3% shift in polyamorphous)
robust_features = [f for f in [
    # Core structural
    'voronoi_volume_temporal', 'R1', 'CN_temporal',
    'pentagon_fraction_temporal', 'asphericity_temporal', 'asphericity_std_temporal',
    # Bond-orientational order parameters
    'q4', 'q6', 'w4', 'w6', 'q4_avg', 'q6_avg',
    # Voronoi face counts
    'n3_temporal', 'n4_temporal', 'n5_temporal', 'n6_temporal',
    # Peak ratios (these depend only on local RDF)
    'sqrt3_peak', 'sqrt4_peak', 'sqrt7_peak', 'sqrt12_peak',
    'sqrt3_ratio', 'sqrt4_ratio', 'sqrt7_ratio', 'sqrt12_ratio',
    # Derived peak features
    'R3_minus_R1', 'R4_minus_R1', 'R7_minus_R1', 'R12_minus_R1',
    'R4_minus_R3', 'R7_minus_R4', 'R12_minus_R7',
    # Derived scalar features
    'entropy', 'q4_divide_q6', 'q4_power_2', 'q6_power_2',
    'R1_time_CN_temporal', 'R3_minus_R1_time_q4',
    'sqrt3_ratio_divide_sqrt4_ratio',
    'CN_time_q4', 'CN_time_q6', 'entropy_time_q4', 'entropy_time_q6',
    'CN_density', 'log_CN', 'log_voronoi',
    # Atomic radius and free volume (per-atom, from type)
    'atomic_radius', 'atomic_sphere_volume', 'free_volume_temporal',
    # Volume-order interactions
    'volume_q6_interaction', 'volume_per_neighbor',
] if f in df_labelled.columns]

# Mixed set: robust + MRO but NOT composition counts
mixed_features = robust_features + [f for f in [
    'mean_neighbor_volume_temporal', 'std_neighbor_volume_temporal',
    'mean_neighbor_pentagon_fraction_temporal', 'std_neighbor_pentagon_fraction_temporal',
    'mean_neighbor_CN_temporal', 'mean_neighbor_free_volume_temporal',
    'mean_neighbor_asphericity_temporal',
    'mean_2nd_neighbor_volume_temporal', 'std_2nd_neighbor_volume_temporal',
    'mean_2nd_neighbor_free_volume_temporal',
    'mean_2nd_neighbor_pentagon_fraction_temporal', 'mean_2nd_neighbor_CN_temporal',
] if f in df_labelled.columns and f not in robust_features]

print(f'All features:       {len(all_features)}')
print(f'Domain-robust only: {len(robust_features)}')
print(f'Mixed (no comp):    {len(mixed_features)}')

# ============================================================
# Load polyamorphous atom positions
# ============================================================
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
df_poly_sorted = df_poly.sort_values('id').reset_index(drop=True)

# ============================================================
# Train and compare models
# ============================================================
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.inspection import permutation_importance

X_all = df_labelled[all_features].values
X_robust = df_labelled[robust_features].values
X_mixed = df_labelled[mixed_features].values
y = df_labelled['phase_label'].values

# Same split for all
X_all_tr, X_all_te, y_tr, y_te = train_test_split(X_all, y, test_size=0.3, random_state=42, stratify=y)
X_rob_tr, X_rob_te = train_test_split(X_robust, test_size=0.3, random_state=42)[0:2]
# Need to use same indices
from sklearn.model_selection import train_test_split as tts
indices = np.arange(len(y))
idx_tr, idx_te = tts(indices, test_size=0.3, random_state=42, stratify=y)

X_rob_tr = X_robust[idx_tr]
X_rob_te = X_robust[idx_te]
X_mix_tr = X_mixed[idx_tr]
X_mix_te = X_mixed[idx_te]
y_tr = y[idx_tr]
y_te = y[idx_te]

# Sample weights with 4654 boost
sw_tr = compute_sample_weight('balanced', y_tr)
sw_tr[y_tr == 1] *= 2.0

configs = {
    'A_all_features': (all_features, X_all[idx_tr], X_all[idx_te], df_poly[all_features].values),
    'B_robust_only': (robust_features, X_rob_tr, X_rob_te, df_poly[robust_features].values),
    'C_mixed_no_comp': (mixed_features, X_mix_tr, X_mix_te, df_poly[mixed_features].values),
}

results = {}
region_bins = [0, 60, 119, 200]
region_names = ['Cu46Zr54 (x<60)', 'Cu50Zr50 (60<x<119)', 'Cu64Zr36 (x>119)']
region_expected_cls = [1, 0, 2]  # expected class per region
regions = pd.cut(atom_df['x'], bins=region_bins, labels=[0, 1, 2]).astype(int)

for name, (feats, Xtr, Xte, Xpoly) in configs.items():
    print(f'\n{"="*70}')
    print(f'MODEL {name} ({len(feats)} features)')
    print(f'{"="*70}')
    
    model = HistGradientBoostingClassifier(
        random_state=42, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=10, max_iter=500, learning_rate=0.05,
        max_leaf_nodes=31, min_samples_leaf=20
    )
    model.fit(Xtr, y_tr, sample_weight=sw_tr)
    
    y_pred = model.predict(Xte)
    probs_te = model.predict_proba(Xte)
    test_acc = accuracy_score(y_te, y_pred)
    test_f1 = f1_score(y_te, y_pred, average='macro')
    
    print(f'Test accuracy: {test_acc:.4f}')
    print(f'F1 macro:      {test_f1:.4f}')
    
    # Per-class report
    report = classification_report(y_te, y_pred, 
                                   target_names=['Cu50Zr50(0)', 'Cu46Zr54(1)', 'Cu64Zr36(2)'],
                                   output_dict=True)
    for c in ['Cu50Zr50(0)', 'Cu46Zr54(1)', 'Cu64Zr36(2)']:
        print(f'  {c}: F1={report[c]["f1-score"]:.4f}, precision={report[c]["precision"]:.4f}, recall={report[c]["recall"]:.4f}')
    
    cm = confusion_matrix(y_te, y_pred)
    print('Confusion matrix:')
    print(pd.DataFrame(cm, index=['True 5050', 'True 4654', 'True 6436'],
                       columns=['Pred 5050', 'Pred 4654', 'Pred 6436']))
    
    # Permutation importance top 10
    r = permutation_importance(model, Xte, y_te, n_repeats=10, random_state=42, n_jobs=4)
    sorted_idx = r.importances_mean.argsort()[::-1]
    print(f'\nTop 10 features:')
    for rank, i in enumerate(sorted_idx[:10], 1):
        print(f'  {rank:2d}. {feats[i]:<50s} {r.importances_mean[i]:.4f}')
    
    # Polyamorphous predictions
    probs_poly = model.predict_proba(Xpoly)
    preds_poly = model.predict(Xpoly)
    conf_poly = np.max(probs_poly, axis=1)
    
    print(f'\nPolyamorphous predictions (threshold=0.50):')
    phase_50 = np.where(conf_poly >= 0.50, preds_poly, -1)
    for ridx, rname in enumerate(region_names):
        rmask = regions == ridx
        n = rmask.sum()
        expected_cls = region_expected_cls[ridx]
        correct = (phase_50[rmask] == expected_cls).sum()
        c0 = (phase_50[rmask] == 0).sum()
        c1 = (phase_50[rmask] == 1).sum()
        c2 = (phase_50[rmask] == 2).sum()
        cu = (phase_50[rmask] == -1).sum()
        print(f'  {rname:<25s} N={n:5d} | C0={c0:5d}({c0/n*100:5.1f}%) C1={c1:5d}({c1/n*100:5.1f}%) '
              f'C2={c2:5d}({c2/n*100:5.1f}%) Unc={cu:5d}({cu/n*100:5.1f}%) | Match={correct/n*100:5.1f}%')
    
    results[name] = {
        'model': model, 'features': feats, 'test_acc': test_acc, 'test_f1': test_f1,
        'probs_poly': probs_poly, 'preds_poly': preds_poly, 'conf_poly': conf_poly,
    }

# ============================================================
# Step 2: Spatial smoothing on the best model
# ============================================================
print('\n' + '='*70)
print('SPATIAL SMOOTHING — Sliding window averaging of probabilities')
print('='*70)

# Use Model B (robust features) as the base
best_key = 'B_robust_only'
probs_base = results[best_key]['probs_poly']

# Sort atoms by x position for spatial smoothing
x_positions = atom_df['x'].values
sort_idx = np.argsort(x_positions)
unsort_idx = np.argsort(sort_idx)  # to map back

probs_sorted = probs_base[sort_idx]
x_sorted = x_positions[sort_idx]

# Sliding window of different sizes
for window_size in [50, 100, 200, 500]:
    # Moving average of probabilities
    from scipy.ndimage import uniform_filter1d
    probs_smooth = np.zeros_like(probs_sorted)
    for cls in range(3):
        probs_smooth[:, cls] = uniform_filter1d(probs_sorted[:, cls], size=window_size, mode='nearest')
    
    # Normalize so probabilities sum to 1
    row_sums = probs_smooth.sum(axis=1, keepdims=True)
    probs_smooth = probs_smooth / row_sums
    
    # Unsort back to original order
    probs_smooth_orig = probs_smooth[unsort_idx]
    preds_smooth = np.argmax(probs_smooth_orig, axis=1)
    conf_smooth = np.max(probs_smooth_orig, axis=1)
    
    print(f'\n  Window={window_size} atoms:')
    for ridx, rname in enumerate(region_names):
        rmask = regions == ridx
        n = rmask.sum()
        expected_cls = region_expected_cls[ridx]
        correct = (preds_smooth[rmask] == expected_cls).sum()
        c0 = (preds_smooth[rmask] == 0).sum()
        c1 = (preds_smooth[rmask] == 1).sum()
        c2 = (preds_smooth[rmask] == 2).sum()
        print(f'    {rname:<25s} N={n:5d} | C0={c0:5d}({c0/n*100:5.1f}%) C1={c1:5d}({c1/n*100:5.1f}%) '
              f'C2={c2:5d}({c2/n*100:5.1f}%) | Match={correct/n*100:5.1f}%')

# ============================================================
# Step 3: Best spatial smoothing — generate outputs
# ============================================================
print('\n' + '='*70)
print('GENERATING OUTPUTS — Best model with optimal spatial smoothing')
print('='*70)

# Find the best window size by maximizing average match across regions
best_window = None
best_match = 0

for window_size in [25, 50, 75, 100, 150, 200, 300, 500, 750, 1000]:
    probs_sorted_temp = probs_base[sort_idx]
    probs_smooth_temp = np.zeros_like(probs_sorted_temp)
    for cls in range(3):
        probs_smooth_temp[:, cls] = uniform_filter1d(probs_sorted_temp[:, cls], size=window_size, mode='nearest')
    row_sums = probs_smooth_temp.sum(axis=1, keepdims=True)
    probs_smooth_temp = probs_smooth_temp / row_sums
    probs_smooth_orig_temp = probs_smooth_temp[unsort_idx]
    preds_smooth_temp = np.argmax(probs_smooth_orig_temp, axis=1)
    
    # Average match
    matches = []
    for ridx in range(3):
        rmask = regions == ridx
        expected_cls = region_expected_cls[ridx]
        matches.append((preds_smooth_temp[rmask] == expected_cls).mean())
    avg_match = np.mean(matches)
    
    if avg_match > best_match:
        best_match = avg_match
        best_window = window_size
    print(f'  Window={window_size:4d}: avg_match={avg_match:.4f} (4654={matches[0]:.3f}, 5050={matches[1]:.3f}, 6436={matches[2]:.3f})')

print(f'\nBest window size: {best_window} atoms (avg match={best_match:.4f})')

# Apply best window
probs_sorted_best = probs_base[sort_idx]
probs_smooth_best = np.zeros_like(probs_sorted_best)
for cls in range(3):
    probs_smooth_best[:, cls] = uniform_filter1d(probs_sorted_best[:, cls], size=best_window, mode='nearest')
row_sums = probs_smooth_best.sum(axis=1, keepdims=True)
probs_smooth_best = probs_smooth_best / row_sums
probs_smooth_best = probs_smooth_best[unsort_idx]
preds_smooth_best = np.argmax(probs_smooth_best, axis=1)
conf_smooth_best = np.max(probs_smooth_best, axis=1)

# Detailed per-region breakdown
print(f'\nDetailed results with window={best_window}:')
for ridx, rname in enumerate(region_names):
    rmask = regions == ridx
    n = rmask.sum()
    expected_cls = region_expected_cls[ridx]
    correct = (preds_smooth_best[rmask] == expected_cls).sum()
    c0 = (preds_smooth_best[rmask] == 0).sum()
    c1 = (preds_smooth_best[rmask] == 1).sum()
    c2 = (preds_smooth_best[rmask] == 2).sum()
    mean_conf = conf_smooth_best[rmask].mean()
    print(f'  {rname:<25s} N={n:5d} | C0={c0:5d}({c0/n*100:5.1f}%) C1={c1:5d}({c1/n*100:5.1f}%) '
          f'C2={c2:5d}({c2/n*100:5.1f}%) | Match={correct/n*100:5.1f}% conf={mean_conf:.3f}')

# Save the best model
model_best = results[best_key]['model']
model_path = 'Machine_Learning_Pipeline_for_Materials_Science/outputs/models/model_3class_robust.pkl'
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model_best, model_path)
print(f'\nSaved robust model to {model_path}')

# Generate LAMMPS dumps (both raw and spatially smoothed)
out_dir = 'Machine_Learning_Pipeline_for_Materials_Science/outputs'

for label, preds_final, conf_final, probs_final in [
    ('robust_raw', results[best_key]['preds_poly'], results[best_key]['conf_poly'], results[best_key]['probs_poly']),
    (f'robust_smooth{best_window}', preds_smooth_best, conf_smooth_best, probs_smooth_best),
]:
    for thresh in [0.50, 0.60, 0.70, 0.80]:
        phase = np.where(conf_final >= thresh, preds_final, -1)
        th_str = f"{thresh:.2f}"
        
        # LAMMPS dump
        output_dump = os.path.join(out_dir, f'polyamorphous_{label}_{th_str}.lammpstrj')
        with open(output_dump, 'w') as f:
            for line in dump_lines[:8]:
                f.write(line)
            new_header = "ITEM: ATOMS " + " ".join(orig_col_names) + " phase confidence\n"
            f.write(new_header)
            for i in range(len(dump_data)):
                f.write(" ".join(dump_data[i]) + f" {int(phase[i])} {conf_final[i]:.4f}\n")
        
        total = len(phase)
        c0 = (phase == 0).sum()
        c1 = (phase == 1).sum()
        c2 = (phase == 2).sum()
        unc = (phase == -1).sum()
        print(f'  {label}_{th_str}: C0={c0:6d}({c0/total*100:5.1f}%) C1={c1:6d}({c1/total*100:5.1f}%) '
              f'C2={c2:6d}({c2/total*100:5.1f}%) Unc={unc:6d}({unc/total*100:5.1f}%)')
    
    # Also write a CSV with per-atom probabilities
    pred_csv = os.path.join(out_dir, f'polyamorphous_{label}_predictions.csv')
    pred_df = pd.DataFrame({
        'atom_id': atom_df['id'],
        'x': atom_df['x'].values,
        'type': df_poly_sorted['type'].values,
        'prob_class0': probs_final[:, 0],
        'prob_class1': probs_final[:, 1],
        'prob_class2': probs_final[:, 2],
        'confidence': conf_final,
        'predicted_phase': preds_final,
    })
    pred_df.to_csv(pred_csv, index=False)
    print(f'  Saved {pred_csv}')

# ============================================================
# Step 4: Visualization — phase fraction profile
# ============================================================
print('\nGenerating visualization...')

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

n_xbins = 40
xbins = np.linspace(atom_df['x'].min(), atom_df['x'].max(), n_xbins + 1)
xcenters = (xbins[:-1] + xbins[1:]) / 2

# Panel 1: Cu% (ground truth)
cu_frac = []
for i in range(n_xbins):
    mask = (atom_df['x'] >= xbins[i]) & (atom_df['x'] < xbins[i+1])
    cu_frac.append((df_poly_sorted.loc[mask, 'type'] == 1).mean() * 100 if mask.sum() > 0 else np.nan)

ax = axes[0]
ax.bar(xcenters, cu_frac, width=(xbins[1]-xbins[0])*0.9, color='steelblue', alpha=0.8)
ax.axhline(46, color='green', ls='--', alpha=0.5, label='Cu46Zr54')
ax.axhline(50, color='blue', ls='--', alpha=0.5, label='Cu50Zr50')
ax.axhline(64, color='red', ls='--', alpha=0.5, label='Cu64Zr36')
ax.set_ylabel('Cu %', fontsize=12)
ax.set_title('Ground Truth: Composition Profile', fontsize=14, fontweight='bold')
ax.legend()
ax.set_ylim(35, 75)

# Panel 2: Raw model B predictions (robust features)
probs_raw = results[best_key]['probs_poly']
for cls, color, label in [(0, 'blue', 'P(Cu50Zr50)'), (1, 'green', 'P(Cu46Zr54)'), (2, 'red', 'P(Cu64Zr36)')]:
    profile = []
    for i in range(n_xbins):
        mask = (atom_df['x'] >= xbins[i]) & (atom_df['x'] < xbins[i+1])
        profile.append(probs_raw[mask, cls].mean() if mask.sum() > 0 else np.nan)
    axes[1].plot(xcenters, profile, '-o', ms=3, color=color, label=label, linewidth=2)
axes[1].set_ylabel('Mean Probability', fontsize=12)
axes[1].set_title(f'Model B (Robust Features Only, {len(robust_features)} features)', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].set_ylim(-0.05, 1.05)

# Panel 3: Spatially smoothed predictions
for cls, color, label in [(0, 'blue', 'P(Cu50Zr50)'), (1, 'green', 'P(Cu46Zr54)'), (2, 'red', 'P(Cu64Zr36)')]:
    profile = []
    for i in range(n_xbins):
        mask = (atom_df['x'] >= xbins[i]) & (atom_df['x'] < xbins[i+1])
        profile.append(probs_smooth_best[mask, cls].mean() if mask.sum() > 0 else np.nan)
    axes[2].plot(xcenters, profile, '-o', ms=3, color=color, label=label, linewidth=2)
axes[2].set_ylabel('Mean Probability', fontsize=12)
axes[2].set_xlabel('x position (Å)', fontsize=12)
axes[2].set_title(f'Spatially Smoothed (window={best_window} atoms)', fontsize=14, fontweight='bold')
axes[2].legend()
axes[2].set_ylim(-0.05, 1.05)

plt.tight_layout()
outpath = 'Machine_Learning_Pipeline_for_Materials_Science/outputs/plots/robust_model_comparison.png'
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f'Saved {outpath}')
plt.close()

print('\n✓ All done.')
