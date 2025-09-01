# Atomic Simulation Post-Processing Pipeline  

This project provides a comprehensive, parallelized pipeline for the structural analysis of atomistic simulation data, particularly from **LAMMPS** trajectories. It's designed to streamline the post-processing workflow, integrating snapshot parsing, temporal averaging, and multi-level spatial analysis to produce detailed, per-atom structural metrics. The pipeline enables high-throughput, reproducible, and parameterized characterization of complex systems, which is crucial for computational materials science and molecular dynamics.  

---

## Table of Contents  

1. [Key Features](#key-features-)  
2. [Technologies and Concepts Used](#technologies-and-concepts-used-)  
   - [Python Libraries](#python-libraries)  
   - [Core Concepts](#core-concepts)  
3. [Project File Descriptions](#project-file-descriptions-)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [License](#license)  

---

## Key Features ✨  

- **Parallelized Workflow**: Efficiently processes large LAMMPS trajectory files using multiprocessing to handle individual snapshots concurrently.  
- **Comprehensive Data Handling**: Reads and validates LAMMPS dump files, and associates atom data with radii from an external mapping file.  
- **Advanced Structural Analysis**: Computes structural descriptors like Radial Distribution Functions (RDFs), Voronoi tessellations, and Steinhardt bond order parameters (**Q4**, **Q6**).  
- **Multi-Level Analysis**: Supports both temporal-only and combined temporal-spatial averaging of RDFs, allowing detailed analysis of local atomic environments.  
- **Robust Peak Analysis**: Implements an RDF peak detection pipeline with Savitzky–Golay smoothing, targeted searches, Gaussian fitting, and validation against crystallographic ratios.  
- **Configurable and Reproducible**: Uses a flexible YAML configuration file to control all analysis parameters, from file paths to peak-finding thresholds.  
- **Data Export**: Outputs per-atom metrics and peak analysis results to CSV for easy integration with downstream tools.  
- **Computational Efficiency**: Optimized with `scipy.spatial.KDTree` for fast neighbor searches and adaptive block sizing for Voronoi computations.  

---

## Technologies and Concepts Used 🛠️  

### Python Libraries  

- **Data Manipulation**: `pandas`, `numpy`  
- **Parallel Processing**: `multiprocessing`, `concurrent.futures`, `joblib`  
- **Scientific Computing**: `scipy` (`spatial.KDTree`, `signal.find_peaks`, `optimize.curve_fit`, `interpolate.UnivariateSpline`)  
- **Domain-Specific Libraries**:  
  - `freud` (Voronoi tessellation, bond order parameters, neighbor queries)  
  - `PyVoro` (high-performance Voronoi tessellation)  
- **Configuration & I/O**: `yaml`, `matplotlib`, `logging`, `io.StringIO`  

### Core Concepts  

- **Molecular Dynamics Post-Processing**: Analysis of LAMMPS trajectory files.  
- **Computational Geometry**: Voronoi tessellation with periodic boundary conditions, KD-trees for neighbor searching.  
- **Statistical Analysis**: RDF computation and temporal averaging.  
- **Signal Processing**: Peak detection, Savitzky–Golay filter smoothing, spline interpolation.  
- **Modeling / Fitting**: Gaussian models for peak characterization.  
- **Parallelization**: Efficient multi-core analysis of large datasets.  

---

## Project File Descriptions 📂  

| File | Description |  
|------|-------------|  
| `pipeline_orchestrator.py` | Top-level entry point coordinating the full workflow: configuration loading, snapshot processing, temporal averaging, peak detection, optional spatial analysis, and final CSV export. |  
| `snapshot_processor.py` | Processes individual simulation snapshots: parses LAMMPS dumps, performs Voronoi tessellations, computes RDFs, and calculates Steinhardt Q4/Q6 order parameters. |  
| `temporal_averaging.py` | Aggregates per-snapshot atomic metrics into temporally averaged values (RDFs, Voronoi volumes, coordination numbers, q4/q6, neighbor counts). |  
| `spatial_analysis_levels.py` | Performs first-level spatial averaging of time-averaged RDFs using Voronoi neighbors, then applies peak detection for structural feature extraction. |  
| `rdf.py` | Provides per-atom RDF computation utilities using KDTree neighbor searches, adaptive binning, normalization, and validation checks. |  
| `peak_analysis.py` | Full RDF peak analysis module: smoothing, global + targeted peak detection, Gaussian fitting, spline interpolation, and validation against crystallographic ratios. |  
| `voronoi.py` | Utilities for weighted Voronoi tessellations in periodic 3D domains with adaptive block sizing, using PyVoro for efficiency. |  
| `io_module.py` | Parses LAMMPS dump files, extracts box boundaries, maps atom types to radii, and produces structured DataFrames for downstream analysis. |  

---

## Usage

- Run the full pipeline with a YAML configuration file:

python src/pipeline_orchestrator.py