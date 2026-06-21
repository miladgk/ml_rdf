# Project Analysis & Improvement Roadmap

## 1. Project Overview

### Goal
Classify amorphous phases in metallic glasses from atomistic simulation data. Given two labeled datasets (phase A, phase B), train a model to identify which atoms belong to which phase, then apply it to a polyamorphous material (potentially as a time series of snapshots).

### Data
- **Phase A** (`features_5050.csv`): ~11,664 atoms, group ID 101
- **Phase B** (`features_6436.csv`): ~13,500 atoms, group ID 123
- **Polyamorphous** (`features_polyamorphous.csv`): unlabeled data for prediction

### Features (9 core structural descriptors)
| Feature | Description | SHAP Importance |
|---------|-------------|-----------------|
| `voronoi_volume_temporal` | Voronoi polyhedron volume | **1.90** |
| `neighbor_1` | First-shell neighbor count | **1.13** |
| `CN_temporal` | Coordination number | **0.54** |
| `q6` | Sixth-order bond orientational parameter | **0.41** |
| `gaussian_peak2_center` | Position of 2nd Gaussian peak in RDF | **0.40** |
| `q4` | Fourth-order bond orientational parameter | **0.23** |
| `gaussian_peak3_center` | Position of 3rd Gaussian peak in RDF | **0.15** |
| `R1_SG` | First interatomic distance peak | **0.12** |
| `neighbor_2` | Second-shell neighbor count | **0.04** |

### Current Performance
- **Best model**: HistGradientBoostingClassifier
- **CV macro F1**: 0.8543
- **Test accuracy**: 0.8528
- **Group accuracy (majority vote)**: 1.0000
- **Runtime**: ~5.5 minutes (down from 28 min)

### Architecture

```
data_utils.py          → Load/label CSVs, group-aware split
models.py              → Build pipelines, RandomizedSearchCV, save/load
pipeline.py            → Main orchestrator (train, evaluate, explain)
explainability.py      → Permutation importance + SHAP
analyze_feature_importance.py → Post-hoc SHAP analysis
apply_model_to_unlabeled.py  → Apply saved model to new data
feature_builder_data_clean.py → Build feature table from raw simulation
```

---

## 2. Current Limitations

### A. No temporal information
The model treats each atom snapshot independently. For time series data, this ignores:
- How local structure evolves between frames
- Whether changes are continuous or abrupt (phase transitions)
- Which atoms are at phase boundaries vs. stable interior

### B. No probability calibration
HistGB probabilities are not well-calibrated. The predicted probability of an atom being phase A vs. B cannot be interpreted as a true confidence score. This matters when:
- Setting thresholds for "uncertain" atoms
- Computing reliable phase fractions per frame

### C. No uncertainty quantification
The model outputs a hard label with no confidence interval. For phase detection:
- An atom with p=0.51 is treated the same as p=0.99
- No way to detect atoms at the phase boundary
- No way to assess whether a prediction is trustworthy

### D. Per-atom noise
Individual atoms fluctuate. A 0.85 per-atom accuracy means ~15% of atoms are labeled wrong. For phase detection:
- Frame-level majority voting improves this (group accuracy = 1.00)
- But we could do better with temporal smoothing

### E. Single model
Only the best model (HistGB) is used. Ensemble predictions from all 3 models would reduce variance.

### F. No frame-level phase tracking
The pipeline doesn't produce the most useful output for time series: a phase fraction vs. time curve showing when and how the material transitions.

---

## 3. Improvement Ideas (Ranked by Impact/Effort)

### Tier 1: High Impact, Low Effort (days)

#### 1.1 Probability Calibration
Wrap the pipeline in `CalibratedClassifierCV` to get well-calibrated probabilities.

**Why**: Reliable probabilities enable threshold-based phase assignment, uncertainty quantification, and confidence-based filtering.

**Implementation**:
```python
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(estimator=histgb, method='isotonic', cv=5)
```

**Expected impact**: Probabilities become interpretable as true confidence scores. Atoms with p near 0.5 are truly uncertain (phase boundary atoms).

#### 1.2 Frame-Level Phase Fraction Tracking
After predicting per-atom labels for each snapshot, compute:
- `n_phase_A / n_total` (fraction of atoms in phase A)
- `n_phase_B / n_total` (fraction in phase B)
- `n_uncertain / n_total` (fraction with |p-0.5| < threshold)

Plot these as a function of snapshot index to visualize phase evolution.

**Why**: This is the key scientific output — it shows when and how the phase transition occurs. A sharp jump indicates a first-order transition; a gradual change suggests continuous transformation.

#### 1.3 Temporal Majority Voting
For each atom, smooth its prediction across ±N snapshots:
```python
smoothed_label = mode(pred[t-N : t+N+1])
```

**Why**: Reduces per-atom noise. An atom that fluctuates between phase A and B is likely at a phase boundary.

#### 1.4 Ensemble Prediction
Average predictions from RandomForest + HistGB + LinearSVC (with calibrated probabilities):
```python
ensemble_prob = np.mean([prob_rf, prob_histgb, prob_svc], axis=0)
```

**Why**: Ensembles reduce variance. With 3 diverse models, the combined prediction is more robust.

### Tier 2: Medium Impact, Medium Effort (1-2 weeks)

#### 2.1 Temporal Delta Features
For each atom, compute change in features between consecutive snapshots:
```python
Δ_feature[t] = feature[t] - feature[t-1]
```

Features to delta:
- `Δ_voronoi_volume_temporal` — volume change rate
- `Δ_CN_temporal` — coordination change rate  
- `Δ_q4`, `Δ_q6` — order parameter change rates
- `Δ_neighbor_1`, `Δ_neighbor_2` — neighbor change rates

Also add rolling statistics:
- `std_CN_temporal[-3:]` — coordination variability over 3 snapshots
- `mean_q6[-5:]` — smoothed q6

**Why**: These capture dynamics. An atom with rapidly changing structure (high deltas) may be at the phase transition front. This could be the strongest signal for identifying the transformation mechanism.

#### 2.2 Phase Transition Detection Algorithm
After computing frame-level phase fractions, apply change-point detection to identify when the transition occurs:

```python
from ruptures import Pelt
algo = Pelt(model="l2").fit(phase_fraction_A)
breakpoints = algo.predict(penalty=10)
```

**Why**: Automatically identifies transition onset, duration, and completion — rather than manual thresholding.

#### 2.3 Spatial Correlation Analysis
For each snapshot, cluster atoms by phase and compute:
- Size of phase A and B domains
- Interface width between domains
- Percolation analysis (does phase B form a connected network?)

**Why**: In amorphous materials, phase transitions often proceed via nucleation and growth. Spatial analysis reveals the mechanism.

### Tier 3: Longer-term, Higher Effort

#### 3.1 Window-Based Classification
Instead of per-atom features, create per-window features that describe the local environment over a time window:
- Mean and std of each feature over ±3 snapshots
- Slope of each feature over ±5 snapshots
- FFT of feature time series (to detect oscillation frequencies)

#### 3.2 Multi-Instance Learning
Treat each snapshot as a bag of atoms. Train a model to predict the frame-level phase fraction directly, rather than per-atom labels.

#### 3.3 Dimensionality Reduction
Apply UMAP or t-SNE to the 9 features across all snapshots to visualize the phase separation trajectory in 2D.

#### 3.4 Active Learning
If labeling snapshots is expensive, use uncertainty sampling to select the most informative frames for manual labeling.

---

## 4. Recommended Implementation Order

### Phase 1 (immediate, ~2 days)
1. Add `CalibratedClassifierCV` wrapper to `models.py`
2. Modify `apply_model_to_unlabeled.py` to output calibrated probabilities
3. Add frame-level phase fraction computation
4. Add temporal majority voting

### Phase 2 (next, ~1 week)
5. Add temporal delta features to `feature_builder_data_clean.py`
6. Re-train model with delta features
7. Add ensemble prediction
8. Add phase transition detection (change-point analysis)

### Phase 3 (future, ~1-2 weeks)
9. Add spatial correlation analysis
10. Add window-based classification
11. Add UMAP/t-SNE visualization

---

## 5. Key Questions for Discussion

1. **Snapshot format**: Are time series snapshots individual CSV files or one file with a frame column?
2. **Snapshot count**: How many snapshots? Every simulation step or subsampled?
3. **Phase transition**: Is the transition sharp (first-order) or gradual (continuous)?
4. **Labeled data source**: Are phases A and B from separate simulations (quenched to different densities/temperatures)?
5. **Polyamorphous data**: Is the polyamorphous material known to phase-separate into A and B domains?
6. **Ground truth**: Do you have any validation (e.g., experimental XRD, known transition points)?
7. **Performance target**: What accuracy is needed for the application to be useful?
8. **Computational budget**: How much time per snapshot is acceptable for predictions?