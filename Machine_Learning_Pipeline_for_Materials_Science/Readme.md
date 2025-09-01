# Machine Learning Pipeline for Materials Science  

This project provides a complete, production-ready machine learning pipeline for classifying per-sample or per-atom materials data. It streamlines the entire workflow, from **data preparation and group-aware splitting** to **model training, hyperparameter tuning, evaluation, and explainability**. The pipeline is designed to be configurable, reproducible, and extensible, making it useful for both research and practical applications in **materials informatics**.  

---

## Table of Contents  

1. [Key Features](#key-features-)  
2. [Technologies and Concepts Used](#technologies-and-concepts-used-)  
   - [Python Libraries](#python-libraries)  
   - [Algorithms and Concepts](#algorithms-and-concepts)  
3. [Project File Descriptions](#project-file-descriptions-)  
4. [Configuration File](#configuration-file-)  
5. [Installation](#installation)  
6. [Usage](#usage)  
7. [License](#license)  

---

## Key Features ✨  

- **End-to-End Workflow**: Orchestrates the ML process from **data loading** to **model training**, **evaluation**, and **explainability outputs**.  
- **Group-Aware Splitting**: Prevents leakage by ensuring related atoms/samples are not split across train/test sets.  
- **Feature Engineering for Materials Science**: Builds feature tables from atomic-level data, extracting **structural descriptors** (bond length ratios, Voronoi volumes, coordination numbers, order parameters).  
- **Flexible Model Tuning**: Supports multiple scikit-learn classifiers with **randomized hyperparameter optimization**.  
- **Comprehensive Evaluation**: Accuracy, F1, classification reports, confusion matrices, and per-group majority-vote metrics.  
- **Explainability**: Model interpretation via **permutation importance** and **SHAP plots**.  
- **Production Deployment**: Apply trained models to new data, generating annotated CSVs with predictions and probabilities.  
- **Configurable via YAML**: A single configuration file defines datasets, features, models, tuning strategies, and output paths for **reproducibility**.  

---

## Technologies and Concepts Used 🛠️  

### Python Libraries  
- **Data Manipulation**: `pandas`, `numpy`  
- **Machine Learning**: `scikit-learn` (Pipelines, Transformers, Group-aware CV, Hyperparameter tuning, Metrics)  
- **Model Explainability**: `shap`  
- **Visualization**: `matplotlib`  
- **Utilities**: `joblib`, `yaml`, `argparse`, `logging`, `pathlib`, `scipy`  

### Algorithms and Concepts  
- **Classification**: Random Forest, Gradient Boosting, SVM  
- **Data Handling**: Group-aware train/test splitting, preprocessing, feature scaling, one-hot encoding, imputation  
- **Model Interpretation**: Permutation Importance, SHAP (tree-based + kernel-based)  
- **Cross-Validation**: Group-aware and stratified k-fold  
- **Reproducibility**: Config-driven experiments, fixed random states, consistent output structure  

---

## Project File Descriptions 📂  

| File | Description |  
|------|-------------|  
| `pipeline.py` | Implements the full end-to-end ML pipeline: data loading, group-aware splitting, training, hyperparameter tuning, evaluation, explainability, and model saving. |  
| `models.py` | Constructs scikit-learn pipelines, handles preprocessing for numeric/categorical features, performs randomized hyperparameter optimization, cross-validation, and model persistence. |  
| `explainability.py` | Utilities for model interpretability: permutation feature importance, SHAP analysis, top-k feature bar plots, mapping transformed features back to original names. |  
| `data_utils.py` | Data preparation utilities: load/validate CSVs, enforce feature presence, perform group-aware stratified train/test splits, and save prediction results. |  
| `feature_table_builder.py` | Processes atomic-level CSV data into ML-ready feature tables: interatomic distance peaks, peak-to-R1 ratios, Voronoi descriptors, coordination numbers, and saving results. |  
| `apply_model_to_unlabeled.py` | Applies a saved model to unlabeled per-atom data, producing annotated CSVs with predicted phase labels and probabilities. Supports CLI and YAML configuration. |  
| `config.yaml` | Central configuration file defining datasets, selected features, models, hyperparameter search spaces, splitting strategies, and output paths for reproducible experiments. |  

---

## Configuration File 📝  

The `config.yaml` file governs the pipeline’s behavior.  
It defines:  
- Input datasets (labeled + unlabeled)  
- Selected atomic structure features  
- Models and hyperparameter search spaces  
- Cross-validation and group-aware splitting strategies  
- Output directories (trained models, reports, plots, predictions)  


## Usage

- Run the full pipeline with a YAML configuration file:

python src/pipeline.py