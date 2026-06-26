"""
diagnose_polyamorphous.py
=========================
Comprehensive diagnostic script for the polyamorphous phase classification.

Produces:
1. Cu% composition profile vs x
2. Model probability and prediction profile vs x
3. Feature domain shift analysis (training vs polyamorphous)
4. Feature importance vs domain shift scatter (identifies problematic features)
5. Per-region feature distributions
"""
import sys, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, 'Machine_Learning_Pipeline_for_Materials_Science/src')

# ============================================================
# Load data
# ============================================================
df_5050 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_5050.csv')
df_4654 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_4654.csv')
df_6436 = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_6436.csv')
df_poly = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/data/features_polyamorphous.csv')

# Load predictions
pred = pd.read_csv('Machine_Learning_Pipeline_for_Materials_Science/outputs/polyamorphous_phase_predictions_0.80.csv')

# Sort everything by id
df_poly_sorted = df_poly.sort_values('id').reset_index(drop=True)
pred_sorted = pred.sort_values('atom_id').reset_index(drop=True)

# ============================================================
# Figure 1: Composition + Classification Profile
# ============================================================
fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

n_bins = 40
bins = np.linspace(pred_sorted['x'].min(), pred_sorted['x'].max(), n_bins + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Panel 1: Cu% composition
cu_frac = []
for i in range(n_bins):
    mask = (pred_sorted['x'] >= bins[i]) & (pred_sorted['x'] < bins[i+1])
    if mask.sum() > 0:
        cu_frac.append((pred_sorted.loc[mask, 'type'] == 1).mean() * 100)
    else:
        cu_frac.append(np.nan)

ax = axes[0]
ax.bar(bin_centers, cu_frac, width=(bins[1]-bins[0])*0.9, color='steelblue', alpha=0.8)
ax.axhline(46, color='green', ls='--', alpha=0.5, label='Cu46Zr54 (46%)')
ax.axhline(50, color='blue', ls='--', alpha=0.5, label='Cu50Zr50 (50%)')
ax.axhline(64, color='red', ls='--', alpha=0.5, label='Cu64Zr36 (64%)')
ax.set_ylabel('Cu %', fontsize=12)
ax.set_title('Ground Truth: Cu Composition Profile', fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax.set_ylim(35, 75)

# Panel 2: Model probabilities
prob_c0, prob_c1, prob_c2 = [], [], []
for i in range(n_bins):
    mask = (pred_sorted['x'] >= bins[i]) & (pred_sorted['x'] < bins[i+1])
    if mask.sum() > 0:
        prob_c0.append(pred_sorted.loc[mask, 'prob_class0'].mean())
        prob_c1.append(pred_sorted.loc[mask, 'prob_class1'].mean())
        prob_c2.append(pred_sorted.loc[mask, 'prob_class2'].mean())
    else:
        prob_c0.append(np.nan)
        prob_c1.append(np.nan)
        prob_c2.append(np.nan)

ax = axes[1]
ax.plot(bin_centers, prob_c0, 'b-o', ms=4, label='P(Cu50Zr50)', linewidth=2)
ax.plot(bin_centers, prob_c1, 'g-s', ms=4, label='P(Cu46Zr54)', linewidth=2)
ax.plot(bin_centers, prob_c2, 'r-^', ms=4, label='P(Cu64Zr36)', linewidth=2)
ax.set_ylabel('Mean Probability', fontsize=12)
ax.set_title('Model Probabilities by Position', fontsize=14, fontweight='bold')
ax.legend()
ax.set_ylim(-0.05, 1.05)

# Panel 3: Phase assignment fractions per bin
frac_c0, frac_c1, frac_c2, frac_unc = [], [], [], []
for i in range(n_bins):
    mask = (pred_sorted['x'] >= bins[i]) & (pred_sorted['x'] < bins[i+1])
    n = mask.sum()
    if n > 0:
        frac_c0.append((pred_sorted.loc[mask, 'phase'] == 0).sum() / n * 100)
        frac_c1.append((pred_sorted.loc[mask, 'phase'] == 1).sum() / n * 100)
        frac_c2.append((pred_sorted.loc[mask, 'phase'] == 2).sum() / n * 100)
        frac_unc.append((pred_sorted.loc[mask, 'phase'] == -1).sum() / n * 100)
    else:
        frac_c0.append(np.nan); frac_c1.append(np.nan)
        frac_c2.append(np.nan); frac_unc.append(np.nan)

ax = axes[2]
ax.bar(bin_centers, frac_c0, width=(bins[1]-bins[0])*0.9, color='blue', alpha=0.6, label='Cu50Zr50 (0)')
ax.bar(bin_centers, frac_c1, width=(bins[1]-bins[0])*0.9, bottom=frac_c0, color='green', alpha=0.6, label='Cu46Zr54 (1)')
bottoms = [a+b for a,b in zip(frac_c0, frac_c1)]
ax.bar(bin_centers, frac_c2, width=(bins[1]-bins[0])*0.9, bottom=bottoms, color='red', alpha=0.6, label='Cu64Zr36 (2)')
bottoms2 = [a+b for a,b in zip(bottoms, frac_c2)]
ax.bar(bin_centers, frac_unc, width=(bins[1]-bins[0])*0.9, bottom=bottoms2, color='gray', alpha=0.4, label='Uncertain')
ax.set_ylabel('% of atoms', fontsize=12)
ax.set_title('Phase Classification per Bin (threshold=0.80)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')

# Panel 4: Expected vs actual
ax = axes[3]
expected = []
for i in range(n_bins):
    cu = cu_frac[i]
    if cu is not None and not np.isnan(cu):
        if cu < 48:
            expected.append('Cu46Zr54')
        elif cu < 55:
            expected.append('Cu50Zr50')
        else:
            expected.append('Cu64Zr36')
    else:
        expected.append('Unknown')

colors_expected = {'Cu46Zr54': 'green', 'Cu50Zr50': 'blue', 'Cu64Zr36': 'red', 'Unknown': 'gray'}
for i in range(n_bins):
    ax.bar(bin_centers[i], 100, width=(bins[1]-bins[0])*0.9, 
           color=colors_expected[expected[i]], alpha=0.3)
    # Dominant predicted class
    dominant = max([(frac_c0[i], 'blue'), (frac_c1[i], 'green'), (frac_c2[i], 'red')], key=lambda x: x[0])
    ax.bar(bin_centers[i], dominant[0], width=(bins[1]-bins[0])*0.5, 
           color=dominant[1], alpha=0.8)

ax.set_ylabel('% dominant class', fontsize=12)
ax.set_xlabel('x position (Å)', fontsize=12)
ax.set_title('Expected Phase (background) vs Dominant Predicted (foreground)', fontsize=14, fontweight='bold')

plt.tight_layout()
outpath = 'Machine_Learning_Pipeline_for_Materials_Science/outputs/plots/diagnostic_polyamorphous.png'
os.makedirs(os.path.dirname(outpath), exist_ok=True)
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f'Saved {outpath}')
plt.close()

# ============================================================
# Figure 2: Feature Domain Shift
# ============================================================
features_to_analyze = [
    'voronoi_volume_temporal', 'R1', 'pentagon_fraction_temporal',
    'free_volume_temporal', 'neighbor_1_fraction_temporal',
    'neighbor_1', 'neighbor_2', 'CN_temporal',
    'mean_neighbor_volume_temporal', 'mean_2nd_neighbor_volume_temporal',
    'mean_neighbor_free_volume_temporal', 'mean_2nd_neighbor_free_volume_temporal',
    'q4', 'q6', 'asphericity_temporal',
    'mean_neighbor_pentagon_fraction_temporal', 'mean_neighbor_CN_temporal'
]

# Assign regions to polyamorphous atoms
pred_sorted['region'] = pd.cut(pred_sorted['x'], bins=[0, 60, 119, 200], labels=['4654_region', '5050_region', '6436_region'])
df_poly_sorted['region'] = pred_sorted['region'].values

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.ravel()

# Select 9 most informative features
key_features = [
    'voronoi_volume_temporal', 'neighbor_1_fraction_temporal', 'mean_neighbor_volume_temporal',
    'mean_2nd_neighbor_volume_temporal', 'R1', 'pentagon_fraction_temporal',
    'neighbor_1', 'neighbor_2', 'free_volume_temporal'
]

for idx, feat in enumerate(key_features):
    ax = axes[idx]
    if feat not in df_5050.columns:
        ax.text(0.5, 0.5, f'{feat}\nNOT FOUND', transform=ax.transAxes, ha='center')
        continue
    
    # Training distributions
    data_train = {
        'Cu46Zr54 (train)': df_4654[feat].dropna(),
        'Cu50Zr50 (train)': df_5050[feat].dropna(),
        'Cu64Zr36 (train)': df_6436[feat].dropna(),
    }
    # Polyamorphous by region
    data_poly = {
        'Cu46Zr54 region': df_poly_sorted.loc[df_poly_sorted['region']=='4654_region', feat].dropna(),
        'Cu50Zr50 region': df_poly_sorted.loc[df_poly_sorted['region']=='5050_region', feat].dropna(),
        'Cu64Zr36 region': df_poly_sorted.loc[df_poly_sorted['region']=='6436_region', feat].dropna(),
    }
    
    colors_train = ['green', 'blue', 'red']
    colors_poly = ['lightgreen', 'lightblue', 'lightsalmon']
    
    all_vals = pd.concat([v for v in list(data_train.values()) + list(data_poly.values())])
    xmin, xmax = all_vals.quantile(0.01), all_vals.quantile(0.99)
    hist_bins = np.linspace(xmin, xmax, 50)
    
    for i, (label, vals) in enumerate(data_train.items()):
        ax.hist(vals, bins=hist_bins, density=True, alpha=0.5, color=colors_train[i], label=label, histtype='stepfilled')
    
    for i, (label, vals) in enumerate(data_poly.items()):
        ax.hist(vals, bins=hist_bins, density=True, alpha=0.8, color=colors_poly[i], label=label, histtype='step', linewidth=2)
    
    ax.set_title(feat.replace('_temporal', '').replace('_', ' '), fontsize=10, fontweight='bold')
    if idx == 0:
        ax.legend(fontsize=6, loc='upper right')
    ax.set_xlabel('')

fig.suptitle('Feature Distributions: Training Phases (filled) vs Polyamorphous Regions (lines)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
outpath2 = 'Machine_Learning_Pipeline_for_Materials_Science/outputs/plots/diagnostic_feature_shift.png'
plt.savefig(outpath2, dpi=150, bbox_inches='tight')
print(f'Saved {outpath2}')
plt.close()

# ============================================================
# Print summary table
# ============================================================
print('\n' + '='*80)
print('DOMAIN SHIFT SUMMARY')
print('='*80)
print(f"\n{'Feature':<50s} {'Avg Rel Err%':>12s} {'Status':>8s}")
print('-'*72)

for f in features_to_analyze:
    if f not in df_5050.columns or f not in df_poly_sorted.columns:
        continue
    t50, t46, t64 = df_5050[f].mean(), df_4654[f].mean(), df_6436[f].mean()
    p46 = df_poly_sorted.loc[df_poly_sorted['region']=='4654_region', f].mean()
    p50 = df_poly_sorted.loc[df_poly_sorted['region']=='5050_region', f].mean()
    p64 = df_poly_sorted.loc[df_poly_sorted['region']=='6436_region', f].mean()
    
    err_46 = abs(p46 - t46) / (abs(t46) + 1e-10) * 100
    err_50 = abs(p50 - t50) / (abs(t50) + 1e-10) * 100
    err_64 = abs(p64 - t64) / (abs(t64) + 1e-10) * 100
    avg_err = (err_46 + err_50 + err_64) / 3
    
    status = 'GOOD' if avg_err < 3 else ('OK' if avg_err < 10 else 'BAD')
    print(f'  {f:<48s} {avg_err:10.1f}%  {status:>6s}')

print('\nGOOD = feature values in polyamorphous match corresponding training phase (<3% error)')
print('BAD  = significant domain shift (>10% error), model predictions based on these may be unreliable')
