# Materials Simulation & Machine Learning Project  

This repository provides an integrated framework for **computational materials science**, combining:  

1. **Atomic Simulation Post-Processing Pipeline** → A parallelized workflow for analyzing atomistic simulations (e.g., LAMMPS trajectories) and extracting structural descriptors.  
2. **Machine Learning Pipeline for Materials Science** → A configurable ML framework for training, tuning, and interpreting models on per-atom and per-sample datasets.  

Together, these two components enable **end-to-end workflows**:  
- From **raw molecular dynamics trajectories** → to **processed structural features** → to **machine learning models** for classification and prediction.  

---

## Repository Structure 📂  

project/
│
├── Atomic_Simulation_Post-Processing_Pipeline/
│ ├── README.md # Detailed docs for post-processing pipeline
│ └── ... # Code for trajectory parsing, RDFs, Voronoi, peak analysis
│
├── Machine_Learning_Pipeline_for_Materials_Science/
│ ├── README.md # Detailed docs for ML pipeline
│ └── ... # Code for training, evaluation, explainability, deployment
│
├── environment/
│ └── environment.yml # Conda environment specification
│
└── README.md # (this file) general overview



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

conda env create -f environment.yml

### Step 3: Activate the Environment
conda activate materials-sim-ml


## Sub-Projects 🚀

### 📦 Atomic Simulation Post-Processing Pipeline

- Parallelized analysis of LAMMPS trajectories.
- Computes RDFs, Voronoi descriptors, bond order parameters (Q4, Q6).
- Performs peak analysis and temporal/spatial averaging.

### 🤖 Machine Learning Pipeline for Materials Science

- Configurable ML pipeline for classification tasks on per-atom/per-sample data.
- Group-aware splitting, hyperparameter tuning, evaluation, and explainability (SHAP, permutation importance).
- Production-ready deployment for applying trained models to new datasets.
- Each sub-project contains its own README.md with detailed instructions, features, and usage examples.
