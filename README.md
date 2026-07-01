# Materials Simulation & Machine Learning Project

This repository provides an integrated framework for **computational materials science**, combining:

1. **Atomic Simulation Post-Processing Pipeline** → A parallelized workflow for analyzing atomistic simulations (e.g., LAMMPS trajectories) and extracting structural descriptors including Voronoi tessellations, RDFs, bond order parameters, and MRO features.
2. **Machine Learning Pipeline for Materials Science** → A configurable ML framework for training, tuning, and interpreting models on per-atom and per-sample datasets, with experiment scripts for calibrated, domain-robust, and hybrid classification approaches.

Together, these two components enable **end-to-end workflows**:
- From **raw molecular dynamics trajectories** → to **processed structural features** → to **machine learning models** for phase classification and prediction.

---

## Repository Structure 📂

```
project/
│
├── README.md                                          # (this file) general overview
├── environment.yml                                    # Conda environment specification
│
├── Atomic_Simulation_Post-Processing_Pipeline/         # Post-processing pipeline
│   ├── README.md                                      # Detailed docs for post-processing pipeline
│   ├── config.yaml                                    # Pipeline configuration
│   ├── config/freud.ini                               # freud library configuration
│   ├── input.txt                                      # Input parameters
│   ├── requirements.txt                               # Python dependencies
│   ├── src/                                           # Source code
│   │   ├── pipeline_orchestrator.py                   # Main pipeline entry point
│   │   ├── pipeline_mpi.py                            # MPI-parallelized pipeline
│   │   ├── snapshot_processor.py                      # Per-snapshot processing
│   │   ├── io_module.py                               # LAMMPS dump I/O
│   │   ├── rdf.py                                     # RDF computation
│   │   ├── voronoi.py                                 # Voronoi tessellation
│   │   ├── peak_analysis.py                           # RDF peak detection
│   │   ├── temporal_averaging.py                      # Temporal averaging
│   │   ├── spatial_analysis_levels.py                 # Spatial analysis
│   │   ├── *.py                                       # Feature injection & workflow scripts
│   │   └── *.sh                                       # Shell workflow runners
│   ├── files/snapshots_{5050,4654,6436}/             # LAMMPS snapshot data
│   └── outputs/                                       # Generated outputs (CSVs, plots)
│
└── Machine_Learning_Pipeline_for_Materials_Science/   # ML pipeline
    ├── README.md                                      # Detailed docs for ML pipeline
    ├── config.yaml                                    # ML pipeline configuration
    ├── src/                                           # Source code
    │   ├── pipeline.py                                # Main training pipeline
    │   ├── models.py                                  # Model construction & tuning
    │   ├── data_utils.py                              # Data loading & splitting
    │   ├── feature_builder_data_clean.py              # Feature engineering
    │   ├── explainability.py                          # Model interpretability
    │   ├── analyze_feature_importance.py              # Feature importance analysis
    │   ├── apply_model_to_unlabeled.py                # Model deployment
    │   └── *.py                                       # Experiment & validation scripts
    ├── data/polyamorphous/                            # Polyamorphous dataset
    └── outputs/                                       # Models, plots, SHAP data
```

---

## Installation ⚙️

This project is designed to run in a dedicated conda environment.

### Step 1: Install System Packages (Linux/WSL)
Before creating the environment, ensure system compilers are available:
```bash
sudo apt update
sudo apt install build-essential
```

### Step 2: Create the Conda Environment
From the project root:

```bash
conda env create -f environment.yml
```

### Step 3: Activate the Environment

```bash
conda activate materials-sim-ml
```

---

## Sub-Projects 🚀

### 📦 Atomic Simulation Post-Processing Pipeline

- Parallelized analysis of LAMMPS trajectories with MPI support.
- Computes RDFs, Voronoi descriptors, bond order parameters (Q4, Q6), bond angle distributions, CSRO, and multi-level MRO features.
- Performs peak analysis, temporal/spatial averaging, and spatial kernel smoothing.
- Includes feature injection scripts and workflow automation (shell + Python runners).

*See `Atomic_Simulation_Post-Processing_Pipeline/README.md` for full documentation.*

### 🤖 Machine Learning Pipeline for Materials Science

- Configurable ML pipeline for 2-class and 3-class classification on per-atom data.
- Supports Random Forest, HistGB, LinearSVC with hyperparameter tuning.
- Group-aware splitting, explainability (SHAP, permutation importance), and model calibration.
- Experiment scripts for calibrated, domain-robust, hybrid, and MRO-boosted approaches.
- Production-ready deployment for applying trained models to new datasets.

*See `Machine_Learning_Pipeline_for_Materials_Science/README.md` for full documentation.*

---

## Quick Start

```bash
# Activate environment
conda activate materials-sim-ml

# Run the post-processing pipeline on a dataset
python Atomic_Simulation_Post-Processing_Pipeline/src/pipeline_orchestrator.py \
    --config Atomic_Simulation_Post-Processing_Pipeline/config.yaml

# Train ML models on processed features
python Machine_Learning_Pipeline_for_Materials_Science/src/pipeline.py \
    --config Machine_Learning_Pipeline_for_Materials_Science/config.yaml