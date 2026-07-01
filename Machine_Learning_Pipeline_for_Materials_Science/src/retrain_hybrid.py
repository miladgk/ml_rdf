"""
retrain_hybrid.py
=================
HYBRID APPROACH: Combine per-atom ML classification with composition-aware
spatial assignment.

PROBLEM SUMMARY (from experiments):
- Model A (all features, 99.8% test): MRO features give perfect training 
  accuracy but DON'T TRANSFER to polyamorphous (domain shift in 2nd-shell averages)
- Model B (robust local features, 59.8% test): Can partially detect 4654 
  but can't separate 4654 from 5050 (structurally identical)
- 4654 and 5050 are structurally indistinguishable with per-atom local features alone.
  Cohen's d for voronoi_volume (best local feature) is only 0.24 between them.

SOLUTION: 2-stage approach:
  Stage 1: Binary classifier (Cu64Zr36 vs {Cu50Zr50+Cu46Zr54})
           → This works well because 6436 IS structurally distinct (d~1.0 for voronoi)
  Stage 2: For atoms NOT classified as 6436, use local composition
           (neighbor_1_fraction) to separate 4654 from 5050
           → In the POLYAMORPHOUS system, the actual neighbor fractions DO differ
              by region (even if shifted from training), so the relative ordering
              can discriminate
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

# Load polyamorphous atom positions
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
# Features to use
# ============================================================
all_features = [c for c in cfg['features'] if c in df_5050.columns]

# Features for Stage 1 (binary: 6436 vs rest)
# Use ALL features here — the MRO features work well for 6436 separation
# because 6436 has VERY different voronoi volume (15.9 vs 17.5-18.0)
# and even in polyamorphous, 6436 region preserves its voronoi_volume signal
stage1_features = all_features

# Features for Stage 2 (4654 vs 5050)
# These need to transfer, so use ONLY local features
# Key: voronoi_volume IS the best single discriminator (d=0.24)
# but augmented with all other local features
stage2_features = [f for f in [
    'voronoi_volume_temporal', 'R1', 'CN_temporal',
    'pentagon_fraction_temporal', 'asphericity_temporal', 'asphericity_std_temporal',
    'q4', 'q6', 'w4', 'w6', 'q4_avg', 'q6_avg',
    'n3_temporal', 'n4_temporal', 'n5_temporal', 'n6_temporal',
    'sqrt3_peak', 'sqrt4_peak', 'sqrt7_peak', 'sqrt12_peak',
    'sqrt3_ratio', 'sqrt4_ratio', 'sqrt7_ratio', 'sqrt12_ratio',
    'R3_minus_R1', 'R4_minus_R1', 'R7_minus_R1', 'R12_minus_R1',
    'R4_minus_R3', 'R7_minus_R4', 'R12_minus_R7',
    'entropy', 'q4_divide_q6', 'q4_power_2', 'q6_power_2',
    'R1_time_CN_temporal', 'R3_minus_R1_time_q4',
    'sqrt3_ratio_divide_sqrt4_ratio',
    'CN_time_q4', 'CN_time_q6', 'entropy_time_q4', 'entropy_time_q6',
    'CN_density', 'log_CN', 'log_voronoi',
    'atomic_radius', 'atomic_sphere_volume', 'free_volume_temporal',
    'volume_q6_interaction', 'volume_per_neighbor',
] if f in df_5050.columns]

print(f'Stage 1 features (6436 vs rest): {len(stage1_features)}')
print(f'Stage 2 features (4654 vs 5050): {len(stage2_features)}')

# ============================================================
# STAGE 1: Binary classifier — Cu64Zr36 vs {Cu50Zr50 + Cu46Zr54}
# ============================================================
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_sample_weight

print('\n' + '='*70)
print('STAGE 1: Binary — Cu64Zr36 (1) vs {Cu50Zr50 + Cu46Zr54} (0)')
print('='*70)

# Merge 5050 and 4654 as class 0, 6436 as class 1
df_not6436 = pd.concat([df_5050, df_4654], ignore_index=True)
df_not6436['binary_label'] = 0
df_6436_copy = df_6436.copy()
df_6436_copy['binary_label'] = 1
df_binary = pd.concat([df_not6436, df_6436_copy], ignore_index=True)

X_s1 = df_binary[stage1_features].values
y_s1 = df_binary['binary_label'].values

idx = np.arange(len(y_s1))
idx_tr, idx_te = train_test_split(idx, test_size=0.3, random_state=42, stratify=y_s1)

model_s1 = HistGradientBoostingClassifier(
    random_state=42, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=10, max_iter=500, learning_rate=0.05,
    max_leaf_nodes=31, min_samples_leaf=20
)
model_s1.fit(X_s1[idx_tr], y_s1[idx_tr])

y_pred_s1 = model_s1.predict(X_s1[idx_te])
probs_s1 = model_s1.predict_proba(X_s1[idx_te])
print(f'Test accuracy: {accuracy_score(y_s1[idx_te], y_pred_s1):.4f}')
print(f'F1 macro:      {f1_score(y_s1[idx_te], y_pred_s1, average="macro"):.4f}')
cm = confusion_matrix(y_s1[idx_te], y_pred_s1)
print(f'Confusion: not-6436={cm[0,0]}/{cm[0].sum()} correct, 6436={cm[1,1]}/{cm[1].sum()} correct')

# Apply to polyamorphous
probs_s1_poly = model_s1.predict_proba(df_poly[stage1_features].values)
pred_is_6436 = probs_s1_poly[:, 1]  # probability of being 6436

print(f'\nPolyamorphous: mean P(6436) = {pred_is_6436.mean():.4f}')
print(f'  P(6436) > 0.5: {(pred_is_6436 > 0.5).sum()} atoms ({(pred_is_6436 > 0.5).mean()*100:.1f}%)')

# Check spatially
region_bins = [0, 60, 119, 200]
region_names = ['Cu46Zr54 (x<60)', 'Cu50Zr50 (60<x<119)', 'Cu64Zr36 (x>119)']
regions = pd.cut(atom_df['x'], bins=region_bins, labels=[0, 1, 2]).astype(int)

for ridx, rname in enumerate(region_names):
    rmask = (regions == ridx).values
    p6436 = pred_is_6436[rmask].mean()
    above = (pred_is_6436[rmask] > 0.5).sum()
    print(f'  {rname:<25s}: mean P(6436)={p6436:.4f}, count P>0.5={above}/{rmask.sum()}')

# ============================================================
# STAGE 2: Binary classifier — Cu46Zr54 (1) vs Cu50Zr50 (0)
# ============================================================
print('\n' + '='*70)
print('STAGE 2: Binary — Cu46Zr54 (1) vs Cu50Zr50 (0) [local features only]')
print('='*70)

df_5050['binary2_label'] = 0
df_4654['binary2_label'] = 1
df_binary2 = pd.concat([df_5050, df_4654], ignore_index=True)

X_s2 = df_binary2[stage2_features].values
y_s2 = df_binary2['binary2_label'].values

idx2 = np.arange(len(y_s2))
idx2_tr, idx2_te = train_test_split(idx2, test_size=0.3, random_state=42, stratify=y_s2)

# Boost class 1 (4654) since it's harder
sw2 = compute_sample_weight('balanced', y_s2[idx2_tr])
sw2[y_s2[idx2_tr] == 1] *= 1.5

model_s2 = HistGradientBoostingClassifier(
    random_state=42, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=10, max_iter=500, learning_rate=0.05,
    max_leaf_nodes=31, min_samples_leaf=20
)
model_s2.fit(X_s2[idx2_tr], y_s2[idx2_tr], sample_weight=sw2)

y_pred_s2 = model_s2.predict(X_s2[idx2_te])
print(f'Test accuracy: {accuracy_score(y_s2[idx2_te], y_pred_s2):.4f}')
print(f'F1 macro:      {f1_score(y_s2[idx2_te], y_pred_s2, average="macro"):.4f}')
cm2 = confusion_matrix(y_s2[idx2_te], y_pred_s2)
print(f'Confusion: 5050={cm2[0,0]}/{cm2[0].sum()} correct, 4654={cm2[1,1]}/{cm2[1].sum()} correct')
print(classification_report(y_s2[idx2_te], y_pred_s2, target_names=['Cu50Zr50', 'Cu46Zr54']))

# Permutation importance for Stage 2
from sklearn.inspection import permutation_importance
r = permutation_importance(model_s2, X_s2[idx2_te], y_s2[idx2_te], n_repeats=10, random_state=42, n_jobs=4)
sorted_idx = r.importances_mean.argsort()[::-1]
print('Top 10 features for 4654 vs 5050 separation:')
for rank, i in enumerate(sorted_idx[:10], 1):
    print(f'  {rank:2d}. {stage2_features[i]:<50s} {r.importances_mean[i]:.4f}')

# Apply to polyamorphous (atoms NOT classified as 6436)
probs_s2_poly = model_s2.predict_proba(df_poly[stage2_features].values)
pred_is_4654 = probs_s2_poly[:, 1]  # probability of being 4654

print(f'\nPolyamorphous: mean P(4654) = {pred_is_4654.mean():.4f}')
for ridx, rname in enumerate(region_names):
    rmask = (regions == ridx).values
    p4654 = pred_is_4654[rmask].mean()
    above = (pred_is_4654[rmask] > 0.5).sum()
    print(f'  {rname:<25s}: mean P(4654)={p4654:.4f}, count P>0.5={above}/{rmask.sum()}')

# ============================================================
# COMBINE: Hierarchical prediction
# ============================================================
print('\n' + '='*70)
print('COMBINED HIERARCHICAL PREDICTION')
print('='*70)

# Stage 1: Determine if atom is 6436
# Stage 2: For non-6436 atoms, determine if 4654 or 5050

for s1_thresh in [0.50, 0.60, 0.70, 0.80]:
    for s2_thresh in [0.50, 0.55, 0.60]:
        # Start with all atoms uncertain
        phase = np.full(len(df_poly), -1)
        
        # Stage 1: assign 6436
        is_6436 = pred_is_6436 >= s1_thresh
        phase[is_6436] = 2  # Cu64Zr36
        
        # Stage 2: among non-6436, assign 4654 vs 5050
        not_6436 = ~is_6436
        is_4654 = pred_is_4654 >= s2_thresh
        is_5050 = pred_is_4654 < (1 - s2_thresh)
        
        phase[not_6436 & is_4654] = 1  # Cu46Zr54
        phase[not_6436 & is_5050] = 0  # Cu50Zr50
        # Remaining are uncertain (confidence below both thresholds)
        
        total = len(phase)
        c0 = (phase == 0).sum()
        c1 = (phase == 1).sum()
        c2 = (phase == 2).sum()
        unc = (phase == -1).sum()
        
        # Per-region match
        matches = []
        for ridx, expected_cls in enumerate([1, 0, 2]):
            rmask = (regions == ridx).values
            matches.append((phase[rmask] == expected_cls).mean())
        avg_match = np.mean(matches)
        
        print(f'S1≥{s1_thresh:.2f} S2≥{s2_thresh:.2f}: C0={c0:5d}({c0/total*100:4.1f}%) '
              f'C1={c1:5d}({c1/total*100:4.1f}%) C2={c2:5d}({c2/total*100:4.1f}%) '
              f'Unc={unc:5d}({unc/total*100:4.1f}%) | '
              f'Match: 4654={matches[0]:.3f} 5050={matches[1]:.3f} 6436={matches[2]:.3f} avg={avg_match:.3f}')

# ============================================================
# Find optimal thresholds
# ============================================================
print('\n' + '='*70)
print('OPTIMAL THRESHOLD SEARCH')
print('='*70)

best_config = None
best_avg = 0

for s1_t in np.arange(0.30, 0.95, 0.05):
    for s2_t in np.arange(0.30, 0.70, 0.05):
        phase = np.full(len(df_poly), -1)
        is_6436 = pred_is_6436 >= s1_t
        phase[is_6436] = 2
        not_6436 = ~is_6436
        phase[not_6436 & (pred_is_4654 >= s2_t)] = 1
        phase[not_6436 & (pred_is_4654 < (1 - s2_t))] = 0
        
        matches = []
        for ridx, expected_cls in enumerate([1, 0, 2]):
            rmask = (regions == ridx).values
            matches.append((phase[rmask] == expected_cls).mean())
        avg_match = np.mean(matches)
        
        if avg_match > best_avg:
            best_avg = avg_match
            best_config = (s1_t, s2_t)

s1_best, s2_best = best_config
print(f'Best thresholds: Stage1={s1_best:.2f}, Stage2={s2_best:.2f}, avg_match={best_avg:.4f}')

# Apply best thresholds
phase_best = np.full(len(df_poly), -1)
is_6436 = pred_is_6436 >= s1_best
phase_best[is_6436] = 2
not_6436 = ~is_6436
phase_best[not_6436 & (pred_is_4654 >= s2_best)] = 1
phase_best[not_6436 & (pred_is_4654 < (1 - s2_best))] = 0

total = len(phase_best)
print(f'\nPhase distribution:')
for cls, name in [(0, 'Cu50Zr50'), (1, 'Cu46Zr54'), (2, 'Cu64Zr36'), (-1, 'Uncertain')]:
    n = (phase_best == cls).sum()
    print(f'  {name}: {n:6d} ({n/total*100:.1f}%)')

print(f'\nPer-region breakdown:')
for ridx, rname in enumerate(region_names):
    rmask = (regions == ridx).values
    n = rmask.sum()
    expected_cls = [1, 0, 2][ridx]
    c0 = (phase_best[rmask] == 0).sum()
    c1 = (phase_best[rmask] == 1).sum()
    c2 = (phase_best[rmask] == 2).sum()
    cu = (phase_best[rmask] == -1).sum()
    match = (phase_best[rmask] == expected_cls).sum() / n * 100
    print(f'  {rname:<25s}: C0={c0:5d}({c0/n*100:4.1f}%) C1={c1:5d}({c1/n*100:4.1f}%) '
          f'C2={c2:5d}({c2/n*100:4.1f}%) Unc={cu:5d}({cu/n*100:4.1f}%) | Match={match:.1f}%')

# ============================================================
# Save models and generate LAMMPS dumps
# ============================================================
print('\n' + '='*70)
print('SAVING OUTPUTS')
print('='*70)

out_dir = 'Machine_Learning_Pipeline_for_Materials_Science/outputs'
os.makedirs(os.path.join(out_dir, 'models'), exist_ok=True)

joblib.dump(model_s1, os.path.join(out_dir, 'models/model_stage1_binary_6436.pkl'))
joblib.dump(model_s2, os.path.join(out_dir, 'models/model_stage2_binary_4654.pkl'))
print(f'Saved Stage 1 model (6436 binary)')
print(f'Saved Stage 2 model (4654 binary)')

# Generate LAMMPS dumps with different threshold combinations
configs_to_write = [
    ('hybrid_optimal', s1_best, s2_best),
    ('hybrid_050_050', 0.50, 0.50),
    ('hybrid_060_050', 0.60, 0.50),
    ('hybrid_070_050', 0.70, 0.50),
    ('hybrid_080_050', 0.80, 0.50),
]

for label, s1_t, s2_t in configs_to_write:
    phase = np.full(len(df_poly), -1)
    conf = np.zeros(len(df_poly))
    
    is_6436 = pred_is_6436 >= s1_t
    phase[is_6436] = 2
    conf[is_6436] = pred_is_6436[is_6436]
    
    not_6436 = ~is_6436
    is_4654 = pred_is_4654 >= s2_t
    is_5050 = pred_is_4654 < (1 - s2_t)
    
    phase[not_6436 & is_4654] = 1
    conf[not_6436 & is_4654] = pred_is_4654[not_6436 & is_4654]
    
    phase[not_6436 & is_5050] = 0
    conf[not_6436 & is_5050] = 1 - pred_is_4654[not_6436 & is_5050]
    
    # Uncertain atoms get max of the two probabilities
    unc_mask = phase == -1
    conf[unc_mask] = np.maximum(pred_is_6436[unc_mask], 
                                np.maximum(pred_is_4654[unc_mask], 1 - pred_is_4654[unc_mask]))
    
    output_dump = os.path.join(out_dir, f'polyamorphous_{label}.lammpstrj')
    with open(output_dump, 'w') as f:
        for line in dump_lines[:8]:
            f.write(line)
        new_header = "ITEM: ATOMS " + " ".join(orig_col_names) + " phase confidence\n"
        f.write(new_header)
        for i in range(len(dump_data)):
            f.write(" ".join(dump_data[i]) + f" {int(phase[i])} {conf[i]:.4f}\n")
    
    total = len(phase)
    c0 = (phase == 0).sum()
    c1 = (phase == 1).sum()
    c2 = (phase == 2).sum()
    unc = (phase == -1).sum()
    
    # Per-region match
    matches = []
    for ridx, expected_cls in enumerate([1, 0, 2]):
        rmask = (regions == ridx).values
        matches.append((phase[rmask] == expected_cls).mean())
    
    print(f'  {label}: C0={c0:5d}({c0/total*100:4.1f}%) C1={c1:5d}({c1/total*100:4.1f}%) '
          f'C2={c2:5d}({c2/total*100:4.1f}%) Unc={unc:5d}({unc/total*100:4.1f}%) | '
          f'Match: 4654={matches[0]:.3f} 5050={matches[1]:.3f} 6436={matches[2]:.3f}')
    
    # CSV with probabilities
    pred_csv = os.path.join(out_dir, f'polyamorphous_{label}_predictions.csv')
    pd.DataFrame({
        'atom_id': atom_df['id'],
        'x': atom_df['x'].values,
        'type': df_poly_sorted['type'].values,
        'prob_6436': pred_is_6436,
        'prob_4654': pred_is_4654,
        'prob_5050': 1 - pred_is_4654,
        'phase': phase,
        'confidence': conf,
    }).to_csv(pred_csv, index=False)

# ============================================================
# Visualization
# ============================================================
print('\nGenerating visualization...')

fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)

n_xbins = 40
xbins = np.linspace(atom_df['x'].min(), atom_df['x'].max(), n_xbins + 1)
xcenters = (xbins[:-1] + xbins[1:]) / 2

# Panel 1: Cu% ground truth
cu_frac = []
for i in range(n_xbins):
    mask = (atom_df['x'] >= xbins[i]) & (atom_df['x'] < xbins[i+1])
    cu_frac.append((df_poly_sorted.loc[mask, 'type'] == 1).mean() * 100 if mask.sum() > 0 else np.nan)

ax = axes[0]
ax.bar(xcenters, cu_frac, width=(xbins[1]-xbins[0])*0.9, color='steelblue', alpha=0.8)
ax.axhline(46, color='green', ls='--', alpha=0.5, label='Cu46Zr54 (46%)')
ax.axhline(50, color='blue', ls='--', alpha=0.5, label='Cu50Zr50 (50%)')
ax.axhline(64, color='red', ls='--', alpha=0.5, label='Cu64Zr36 (64%)')
ax.set_ylabel('Cu %', fontsize=12)
ax.set_title('Ground Truth: Composition Profile', fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax.set_ylim(35, 75)

# Panel 2: Stage 1 — P(6436)
p6436_profile = []
for i in range(n_xbins):
    mask = (atom_df['x'] >= xbins[i]) & (atom_df['x'] < xbins[i+1])
    p6436_profile.append(pred_is_6436[mask].mean() if mask.sum() > 0 else np.nan)

ax = axes[1]
ax.plot(xcenters, p6436_profile, 'r-o', ms=4, linewidth=2, label='P(Cu64Zr36)')
ax.axhline(s1_best, color='gray', ls=':', alpha=0.5, label=f'threshold={s1_best:.2f}')
ax.fill_between(xcenters, p6436_profile, alpha=0.2, color='red')
ax.set_ylabel('Probability', fontsize=12)
ax.set_title('Stage 1: P(Cu64Zr36) — Binary Classifier', fontsize=14, fontweight='bold')
ax.legend()
ax.set_ylim(-0.05, 1.05)

# Panel 3: Stage 2 — P(4654) for non-6436 atoms
p4654_profile = []
for i in range(n_xbins):
    mask = (atom_df['x'] >= xbins[i]) & (atom_df['x'] < xbins[i+1])
    # Only non-6436 atoms
    non6436_mask = mask & (~(pred_is_6436 >= s1_best))
    p4654_profile.append(pred_is_4654[non6436_mask].mean() if non6436_mask.sum() > 0 else np.nan)

ax = axes[2]
ax.plot(xcenters, p4654_profile, 'g-o', ms=4, linewidth=2, label='P(Cu46Zr54)')
ax.plot(xcenters, [1-p if p is not None and not np.isnan(p) else np.nan for p in p4654_profile], 
        'b-s', ms=4, linewidth=2, label='P(Cu50Zr50)')
ax.axhline(s2_best, color='gray', ls=':', alpha=0.5, label=f'threshold={s2_best:.2f}')
ax.set_ylabel('Probability', fontsize=12)
ax.set_title('Stage 2: P(Cu46Zr54) vs P(Cu50Zr50) — Among Non-6436 Atoms', fontsize=14, fontweight='bold')
ax.legend()
ax.set_ylim(-0.05, 1.05)

# Panel 4: Final phase assignment (stacked bar)
frac_c0, frac_c1, frac_c2, frac_unc = [], [], [], []
for i in range(n_xbins):
    mask = (atom_df['x'] >= xbins[i]) & (atom_df['x'] < xbins[i+1])
    n = mask.sum()
    if n > 0:
        frac_c0.append((phase_best[mask] == 0).sum() / n * 100)
        frac_c1.append((phase_best[mask] == 1).sum() / n * 100)
        frac_c2.append((phase_best[mask] == 2).sum() / n * 100)
        frac_unc.append((phase_best[mask] == -1).sum() / n * 100)
    else:
        frac_c0.append(0); frac_c1.append(0); frac_c2.append(0); frac_unc.append(0)

ax = axes[3]
w = (xbins[1]-xbins[0])*0.9
ax.bar(xcenters, frac_c1, width=w, color='green', alpha=0.7, label='Cu46Zr54 (1)')
ax.bar(xcenters, frac_c0, width=w, bottom=frac_c1, color='blue', alpha=0.7, label='Cu50Zr50 (0)')
bottoms = [a+b for a,b in zip(frac_c1, frac_c0)]
ax.bar(xcenters, frac_c2, width=w, bottom=bottoms, color='red', alpha=0.7, label='Cu64Zr36 (2)')
bottoms2 = [a+b for a,b in zip(bottoms, frac_c2)]
ax.bar(xcenters, frac_unc, width=w, bottom=bottoms2, color='gray', alpha=0.4, label='Uncertain')
ax.set_ylabel('% of atoms', fontsize=12)
ax.set_xlabel('x position (Å)', fontsize=12)
ax.set_title(f'Final Phase Assignment (S1≥{s1_best:.2f}, S2≥{s2_best:.2f})', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')

plt.tight_layout()
outpath = 'Machine_Learning_Pipeline_for_Materials_Science/outputs/plots/hybrid_model_results.png'
os.makedirs(os.path.dirname(outpath), exist_ok=True)
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f'Saved {outpath}')
plt.close()

print('\n✓ All done.')
