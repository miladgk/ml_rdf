# Machine Learning Pipeline for Materials Science

This project provides a complete, production-ready machine learning pipeline for classifying per-atom materials data into structural phases. It streamlines the entire workflow from **data preparation and group-aware splitting** to **model training, hyperparameter tuning, evaluation, and explainability** — all configurable from a single YAML file.

The pipeline is optimized for speed: **full training runs in ~5 minutes with 0.854 CV score**, compared to ~28 minutes in the original version.

Beyond the core pipeline, this project includes a comprehensive set of **experiment scripts** that explore advanced classification strategies: calibrated models, domain-robust feature selection, hybrid 2-stage approaches, MRO-boosted training, and spatial kernel smoothing for polyamorphous phase detection.

---

## Table of Contents

1. [Key Features](#key-features-)
2. [Technologies and Concepts Used](#technologies-and-concepts-used-)
3. [Project File Descriptions](#project-file-descriptions-)
4. [Configuration File](#configuration-file-)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Output and Explainability](#output-and-explainability-)
8. [Performance Optimizations](#performance-optimizations-)
9. [License](#license)

---

## Key Features ✨

- **End-to-End Workflow**: Orchestrates the ML process from data loading to model training, evaluation, and explainability.
- **Group-Aware Splitting**: Prevents leakage by ensuring related atoms/samples (e.g., from the same simulation run) are not split across train/test sets.
- **Feature Engineering**: Builds feature tables from atomic-level data, extracting structural descriptors (bond length ratios, Voronoi volumes, coordination numbers, order parameters).
- **Feature Selection**: The pipeline identifies and ranks the most predictive features via SHAP importance — reducing from 25+ derived features to just 9 core structural descriptors with no loss in accuracy.
- **Multiple Classifiers**: Supports Random Forest, Histogram Gradient Boosting, and Linear SVC.
- **Hyperparameter Optimization**: Uses `RandomizedSearchCV` with StratifiedKFold for robust parameter tuning.
- **Comprehensive Evaluation**: Accuracy, F1, classification reports, confusion matrices, and per-group majority-vote metrics.
- **Explainability**: Model interpretation via permutation importance and SHAP summary plots.
- **Cached Preprocessing**: The preprocessor (imputation + scaling) is fit once; all CV folds reuse the cached transform — saving ~30s per model.
- **Early Stopping**: HistGB uses early stopping internally, cutting training time from 10+ minutes to <1 minute.
- **No Convergence Warnings**: LinearSVC uses the primal solver (`dual=False`) which converges instantly with thousands of samples and few features.
- **Configurable via YAML**: A single configuration file defines datasets, features, models, tuning strategies, and output paths for reproducibility.

---

## Technologies and Concepts Used 🛠️

### Python Libraries
- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn` (Pipelines, Transformers, ColumnTransformer, FunctionTransformer, Group-aware CV, Hyperparameter tuning, Metrics)
- **Model Explainability**: `shap`
- **Visualization**: `matplotlib`
- **Utilities**: `joblib`, `yaml`, `argparse`, `logging`, `pathlib`

### Algorithms and Concepts
- **Classification**: Random Forest, Histogram Gradient Boosting, Linear SVC
- **Preprocessing**: `SimpleImputer`, `StandardScaler`, `MissingIndicator`, `OneHotEncoder`, `FunctionTransformer`
- **Cross-Validation**: StratifiedKFold
- **Hyperparameter Search**: `RandomizedSearchCV` with cached pre-transformed features
- **Feature Importance**: Permutation Importance, SHAP values (TreeExplainer)
- **Group-Aware Splitting**: Custom 2-group independent split ensures balanced representation
- **Reproducibility**: Config-driven experiments, fixed random states, consistent output structure

---

## Project File Descriptions 📂

### Core Pipeline Modules (`src/`)

| File | Description |
|------|-------------|
| `pipeline.py` | Main entry point. Loads data, splits by group, trains all configured models via `RandomizedSearchCV`, evaluates the best model, runs explainability (permutation importance + SHAP), and saves outputs. |
| `models.py` | Constructs scikit-learn pipelines with preprocessing (imputation + scaling), `RandomizedSearchCV` with StratifiedKFold, parameter validation, and model save/load via joblib. Includes `build_cached_preprocessing_pipeline()` for pre-transformed data. |
| `explainability.py` | Utilities for model interpretability: permutation feature importance, SHAP analysis (tree-based and kernel-based), top-k feature bar plots, mapping transformed features back to original names, and stratified background sampling. |
| `data_utils.py` | Data preparation: load/validate CSVs, attach phase labels and group IDs, ensure required columns, perform group-aware stratified train/test splits for 2-group datasets, save predictions to CSV. |
| `feature_builder_data_clean.py` | Processes atomic-level CSV data into ML-ready feature tables. Extracts interatomic distance peaks (sqrt3, sqrt4, sqrt7, sqrt12), peak-to-R1 ratios, Voronoi volumes, coordination numbers, Q4/Q6 order parameters, neighbor statistics, and derived interaction features. Numba-accelerated peak matching. |
| `apply_model_to_unlabeled.py` | Applies a saved pipeline to unlabeled data, producing annotated CSVs with predicted phase labels and probabilities. |
| `analyze_feature_importance.py` | Standalone script that loads saved SHAP values and prints ranked feature importance — run after pipeline completes. |

### Experiment & Validation Scripts (`src/`)

| File | Description |
|------|-------------|
| `retrain_calibrated.py` | Retrains the 3-class model with probability calibration (`CalibratedClassifierCV`), class weighting, and feature importance analysis with domain-shift awareness. Generates labeled LAMMPS dumps at multiple thresholds. |
| `retrain_domain_robust.py` | Trains using only per-atom local features (voronoi_volume, R1, pentagon_fraction, q4, q6) that preserve their phase signature in polyamorphous systems. Addresses domain shift in MRO and composition features. |
| `retrain_hybrid.py` | 2-stage hybrid approach: combines per-atom ML classification with composition-aware spatial assignment. Stage 1: binary classifier to detect 6436 vs others. Stage 2: ternary classifier on non-6436 atoms. |
| `train_3class_mro.py` | 3-class retraining with MRO features, including grid search and comprehensive evaluation across all datasets. |
| `train_3class_fast.py` | Fast 3-class retraining with MRO features (no grid search). Rebuilds ML tables and trains a HistGB classifier. |
| `train_mro_retrain.py` | Retrains models with MRO features, comparing performance with and without MRO descriptors. |
| `run_full_pipeline.py` | End-to-end pipeline: rebuilds feature tables, trains binary model, runs diagnostics, and evaluates on polyamorphous data. |
| `run_final_solution.py` | Final solution script: builds ML tables, trains a HistGB classifier, applies to polyamorphous data, and generates labeled LAMMPS dumps. |
| `apply_spatial_kernel_smoothing.py` | Applies spatial kernel smoothing to model predictions using neighbor averaging, producing smoothed phase labels for visualization in OVITO. |
| `diagnose_polyamorphous.py` | Comprehensive diagnostic script for polyamorphous phase classification: composition profiles, model probability profiles, feature domain shift analysis, and per-region feature distributions. |
| `final_validation_mro2.py` | Complete validation of second-level MRO features: rebuilds ML tables, trains models, and evaluates on all datasets including polyamorphous. |
| `final_validation_mro2_fixed.py` | Corrected version of MRO2 validation with proper box bounds and output paths. |
| `generate_all_thresholds.py` | Generates labeled LAMMPS files at multiple confidence thresholds for visual inspection in OVITO. |
| `generate_correct_labeled_dump.py` | Generates correctly aligned labeled LAMMPS dump files from prediction CSVs. |

### Data Files (in `data/`)

| File | Description |
|------|-------------|
| `features_5050.csv` | Labeled data for phase A (group 101, ~11,664 rows) |
| `features_6436.csv` | Labeled data for phase B (group 123, ~13,500 rows) |
| `features_polyamorphous.csv` | Unlabeled data for prediction |

---

## Configuration File 📝

The `config.yaml` file governs the pipeline's behavior in a single place:

```yaml
data:
  phase_A_csv: "data/features_5050.csv"
  phase_B_csv: "data/features_6436.csv"
  unlabeled_csv: "data/features_polyamorphous.csv"

features:
  - R1_SG
  - voronoi_volume_temporal
  - CN_temporal
  - q4
  - q6
  - gaussian_peak2_center
  - gaussian_peak3_center
  - neighbor_1
  - neighbor_2

models:
  RandomForest: { n_iter_search: 48, ... }
  HistGB:       { n_iter_search: 48, early_stopping: true, ... }
  LinearSVC:    { n_iter_search: 10, dual: false, ... }
```

**Key configuration options:**
- **Features**: Only 9 core structural descriptors are used — SHAP analysis confirmed that the other 8+ derived features (peak ratios, differences, interaction terms) were redundant.
- **Models**: 3 models are trained. SVC was removed because LinearSVC achieves the same score in 1/12th the time.
- **LinearSVC**: `dual: false` uses the primal solver — converges instantly when samples >> features.
- **HistGB**: `early_stopping: true` stops training when the validation score plateaus, cutting runtime by 10×.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/MiladGK/ml_rdf.git
cd ml_rdf

# Create and activate a conda environment (recommended)
conda create -n materials-ml python=3.10
conda activate materials-ml

# Install dependencies
pip install pandas numpy scikit-learn matplotlib shap joblib pyyaml tqdm
```

Optional dependencies:
- `numba` — accelerates peak matching in `feature_builder_data_clean.py` (~2× faster)

---

## Usage

### 1. Run the full training pipeline

```bash
python src/pipeline.py --config config.yaml
```

This will:
1. Load and label the two phase datasets
2. Perform a group-aware 80/20 train-test split
3. Preprocess features (median imputation + standardization) once and cache
4. Train Random Forest, HistGB, and LinearSVC with randomized hyperparameter search
5. Select the best model by macro F1 score
6. Evaluate on the held-out test set
7. Compute permutation importance and SHAP values
8. Save the best pipeline, predictions, and plots

### 2. Analyze feature importance after training

```bash
python src/analyze_feature_importance.py
```

Prints a ranked table of features by mean absolute SHAP value and saves a bar chart to `outputs/plots/feature_importance_shap.png`.

### 3. Apply a saved model to new unlabeled data

```bash
python src/apply_model_to_unlabeled.py --config config.yaml
```

### 4. Build a feature table from raw simulation data

```bash
python src/feature_builder_data_clean.py
```

Processes atomic simulation CSV output and generates an ML-ready feature table with peak positions, ratios, and structural descriptors.

### 5. Run experiment scripts

```bash
# Calibrated 3-class model
python src/retrain_calibrated.py

# Domain-robust model (local features only)
python src/retrain_domain_robust.py

# Hybrid 2-stage approach
python src/retrain_hybrid.py

# Fast 3-class training with MRO
python src/train_3class_fast.py

# Full pipeline with diagnostics
python src/run_full_pipeline.py

# Final solution with spatial kernel smoothing
python src/run_final_solution.py
python src/apply_spatial_kernel_smoothing.py

# Diagnostics and validation
python src/diagnose_polyamorphous.py
python src/final_validation_mro2.py
python src/generate_all_thresholds.py
```

---

## Output and Explainability 📊

After a successful run, the pipeline produces:

| Output | Path | Description |
|--------|------|-------------|
| **Best model** | `outputs/models/best_pipeline.joblib` | Full sklearn pipeline (preprocessor + classifier) |
| **Test predictions** | `predictions.csv` | Annotated test set with predicted phases and probabilities |
| **Permutation importance** | `outputs/plots/permutation_importance.png` | Bar chart of top features by permutation importance |
| **SHAP values** | `outputs/shap_data/shap_values.csv` | SHAP values per sample per feature |
| **SHAP summary** | `outputs/plots/shap_summary.png` | SHAP summary beeswarm plot |
| **Feature ranking** | `outputs/plots/feature_importance_shap.png` | Bar chart from `analyze_feature_importance.py` |

### Typical Results

With the default 9 core features and 3 models, the pipeline achieves:

| Metric | Value |
|--------|-------|
| Best model | HistGradientBoostingClassifier |
| CV score (macro F1) | 0.8543 |
| Test accuracy | 0.8528 |
| Group majority-vote accuracy | 1.0000 |

Top features by SHAP importance:
1. `voronoi_volume_temporal` (1.90)
2. `neighbor_1` (1.13)
3. `CN_temporal` (0.54)
4. `q6` (0.41)
5. `gaussian_peak2_center` (0.40)

---

## Performance Optimizations 🚀

The pipeline incorporates several optimizations that reduced runtime from ~28 minutes to ~5.5 minutes with **no loss in accuracy**:

| Optimization | Impact |
|---|---|
| **Feature reduction** (25 → 9 features) | Fewer columns to process, zero accuracy loss |
| **MissingIndicator removed** | Halved ColumnTransformer feature count |
| **Pre-transform once, cache** | Preprocessor fit once; CV folds reuse cached array |
| **`early_stopping: true` on HistGB** | Cut HistGB from ~10 min to ~1 min |
| **`dual: false` on LinearSVC** | Eliminated convergence warnings, converges instantly |
| **SVC removed** | Saved ~12 min — LinearSVC achieves same score |
| **`n_jobs=-1`** on all models | Full CPU parallelism across all cores |
| **`build_cached_preprocessing_pipeline()`** | No-op passthrough for pre-transformed data |

---

## License

This project is provided for research and educational purposes. See the repository for details.

---

*Built with scikit-learn, SHAP, and Python. See [the repository](https://github.com/MiladGK/ml_rdf) for more information.*