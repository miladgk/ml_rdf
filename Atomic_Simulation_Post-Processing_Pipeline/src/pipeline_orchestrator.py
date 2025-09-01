"""
pipeline_orchestrator.py
=========================

This module orchestrates a complete structural analysis workflow for atomistic
simulation data, integrating snapshot processing, temporal averaging, peak
detection, and optional spatial analysis into a configurable, parallelized
pipeline.

It is designed for large-scale processing of LAMMPS trajectory datasets and
produces detailed per-atom structural metrics, exporting results to CSV.

Key functionalities:
--------------------
- Load configuration parameters from YAML.
- Efficiently process LAMMPS dump snapshots in parallel using `ProcessPoolExecutor`.
- Compute per-atom radial distribution functions (RDF) and Voronoi volumes.
- Perform temporal averaging of RDFs and structural descriptors across snapshots.
- Detect and characterize RDF peaks (global, targeted, validated) using Gaussian fitting.
- Execute multi-level spatial analyses (e.g., temporal-only, temporal-first-spatial).
- Export merged results to CSV with consistent column naming and serialization.

Core dependencies:
------------------
- **pandas**, **numpy**: numerical analysis and data handling.
- **multiprocessing**, **concurrent.futures**: parallel snapshot and atom-level processing.
- **yaml**: flexible pipeline configuration.
- **rdf**, **Voronoi tessellation**: structural analysis techniques.
- **Peak analysis**: Gaussian fitting, peak validation, and ratio checks.

This module serves as the top-level entry point for automated, large-scale
post-processing of molecular dynamics outputs, enabling reproducible and
parameterized structural characterization.
"""

import os
import gc
import time
import json
import yaml
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# Custom Modules
import spatial_analysis_levels
import peak_analysis
import snapshot_processor
import temporal_averaging
import rdf

# Configure logging (important to do early)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True
)


# --- Configuration Loading ---
def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load analysis pipeline configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Dictionary of loaded configuration parameters.

    Raises:
        SystemExit: If the file is not found or cannot be parsed.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_path}")
        exit(1)
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        exit(1)


def process_snapshots_in_chunks(
    snapshot_paths: list,
    chunk_size: int,
    max_workers: int,
    params: dict,
    bins_for_rdf_calc: np.ndarray,
    bin_volumes_for_rdf_calc: np.ndarray
) -> list:
    """
    Process LAMMPS snapshots in parallel chunks to extract per-atom RDFs and Voronoi volumes.

    Args:
        snapshot_paths (list): List of snapshot file paths.
        chunk_size (int): Number of snapshots processed per batch.
        max_workers (int): Number of worker processes to spawn.
        params (dict): Configuration parameters.
        bins_for_rdf_calc (np.ndarray): Precomputed RDF bins.
        bin_volumes_for_rdf_calc (np.ndarray): Precomputed RDF bin volumes.

    Returns:
        list: List of processed snapshot data dictionaries.
    """
    results = []
    total = len(snapshot_paths)

    for chunk_start in range(0, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        chunk = snapshot_paths[chunk_start:chunk_end]

        logging.info(f"Processing snapshots {chunk_start + 1}-{chunk_end}/{total}...")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    snapshot_processor.process_single_snapshot,
                    path,
                    params,
                    bins_for_rdf_calc,
                    bin_volumes_for_rdf_calc
                )
                for path in chunk
            ]

            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result(timeout=params.get('WORKER_TIMEOUT', 300))
                    if result:
                        results.append(result)
                    else:
                        logging.warning(f"Snapshot {chunk[i]} returned None.")
                except Exception as e:
                    logging.error(f"Error in snapshot {chunk[i]}: {e}", exc_info=True)

        # Cleanup memory between chunks
        gc.collect()
        logging.info(f"Memory cleaned up after processing snapshots {chunk_start + 1}-{chunk_end}.")

    return results


def _json_fallback_handler(obj):
    """
    Convert unsupported objects to JSON-serializable types.

    Args:
        obj: Input object to convert.

    Returns:
        JSON-serializable version of the input object.
    """
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): _json_fallback_handler(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_json_fallback_handler(v) for v in obj]
    return str(obj)


def convert_dict_keys_to_str(obj):
    """
    Recursively convert dictionary keys to strings.

    Args:
        obj (dict, list, or other): Input structure.

    Returns:
        dict, list, or object: Input with stringified keys where applicable.
    """
    if isinstance(obj, dict):
        return {str(k): convert_dict_keys_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_dict_keys_to_str(x) for x in obj]
    else:
        return obj


def csv_export(initial_atom_data_list: list, all_analysis_results: list,
               params: dict, analysis_level_name: str) -> None:
    """
    Export combined atom data and analysis results to a CSV file.

    Args:
        initial_atom_data_list (list): List of time-averaged atom data dictionaries.
        all_analysis_results (list): List of analysis results dictionaries (e.g., peak analysis).
        params (dict): Pipeline configuration parameters.
        analysis_level_name (str): Analysis level identifier for output naming.
    """
    rename_keys_for_csv = {
        'number_of_global_peaks_SG': 'number_of_global_peaks_SG',
        'global_peak_r_values_SG': 'global_peak_r_values_SG',
        'global_peak_heights_SG': 'global_peak_heights_SG',
        'R1_SG': 'R1_SG',
        'num_targeted_peaks_second_third': 'num_targeted_peaks_second_third',
        'targeted_peak_r_second_third': 'targeted_peak_r_second_third',
        'targeted_peak_heights_second_third': 'targeted_peak_heights_second_third',
        'validated_sqrt3_r_SG': 'validated_sqrt3_r_SG',
        'validated_sqrt4_r_SG': 'validated_sqrt4_r_SG',
        'validated_sqrt7_r_SG': 'validated_sqrt7_r_SG',
        'validated_sqrt12_r_SG': 'validated_sqrt12_r_SG',
        'Ri_over_R1_ratios_SG': 'Ri_over_R1_ratios_SG',
        'ratio_validated_sqrt3': 'ratio_validated_sqrt3',
        'ratio_validated_sqrt4': 'ratio_validated_sqrt4',
        'ratio_validated_sqrt7': 'ratio_validated_sqrt7',
        'ratio_validated_sqrt12': 'ratio_validated_sqrt12',
        'fit_success': 'gaussian_fit_successful',
        'fit_h1': 'gaussian_peak2_height',
        'fit_c1': 'gaussian_peak2_center',
        'fit_h2': 'gaussian_peak3_height',
        'fit_c2': 'gaussian_peak3_center',
        'fit_r2': 'gaussian_fit_r_squared'
    }

    final_atom_results_for_df = []

    # Map atom_id → temporal data for quick lookup
    temporal_avg_data_map = {d['id']: d for d in initial_atom_data_list}

    for result_atom_data in all_analysis_results:
        atom_id = result_atom_data['id']
        original_temporal_data = temporal_avg_data_map.get(atom_id, {})

        # Base atom data
        combined_data = {
            'id': original_temporal_data.get('id'),
            'type': original_temporal_data.get('type'),
            'x': original_temporal_data.get('x'),
            'y': original_temporal_data.get('y'),
            'z': original_temporal_data.get('z'),
            'radius': original_temporal_data.get('radius'),
            'voronoi_volume_temporal': original_temporal_data.get('voronoi_volume_temporal'),
            'CN_temporal': original_temporal_data.get('CN_temporal', []),
            'g_i_r_temporal_avg': json.dumps(
                np.around(original_temporal_data.get('g_i_r_temporal_avg', np.array([])), 4).tolist()
            ),
            'q4': original_temporal_data.get('q4_temporal', np.nan),
            'q6': original_temporal_data.get('q6_temporal', np.nan),
        }

        # Flatten neighbor composition into columns
        neighbors_dict = original_temporal_data.get('neighbors_by_type_temporal', {})
        for n_type, count in neighbors_dict.items():
            combined_data[f'neighbor_{n_type}'] = count

        # Rename and add analysis results
        renamed_result_atom_data = {
            rename_keys_for_csv.get(key, key): value
            for key, value in result_atom_data.items()
        }

        for key, value in renamed_result_atom_data.items():
            if key in ['id', 'type', 'x', 'y', 'z', 'radius',
                       'voronoi_volume_temporal', 'CN_temporal',
                       'g_i_r_temporal_avg', 'idx']:
                continue

            value_safe_keys = convert_dict_keys_to_str(value)

            try:
                if isinstance(value_safe_keys, np.ndarray):
                    combined_data[key] = np.around(value_safe_keys, 4).tolist()
                elif isinstance(value_safe_keys, list):
                    combined_data[key] = [_json_fallback_handler(v) for v in value_safe_keys]
                else:
                    combined_data[key] = _json_fallback_handler(value_safe_keys)
            except Exception as e:
                logging.warning(f"Could not serialize key '{key}' with value '{value}': {e}")
                combined_data[key] = str(value)

        final_atom_results_for_df.append(combined_data)

    # --- Create DataFrame ---
    logging.info("Creating final DataFrame and exporting results to CSV.")

    if final_atom_results_for_df:
        neighbor_type_cols = [
            col for col in final_atom_results_for_df[0] if col.startswith('neighbor_')
        ]
        df_final_results = pd.DataFrame(final_atom_results_for_df)
    else:
        logging.warning(
            f"No final results generated for '{analysis_level_name}' analysis. "
            "Output CSV will be empty."
        )
        neighbor_type_cols = []
        df_final_results = pd.DataFrame()

    # Define and enforce column order
    column_order = [
        'id', 'type', 'x', 'y', 'z', 'radius',
        'voronoi_volume_temporal', 'CN_temporal',
        'g_i_r_temporal_avg', 'first_avg_rdf',
        'number_of_global_peaks_SG', 'global_peak_r_values_SG',
        'global_peak_heights_SG', 'R1_SG',
        'num_targeted_peaks_second_third', 'targeted_peak_r_second_third',
        'targeted_peak_heights_second_third',
        'validated_sqrt3_r_SG', 'validated_sqrt4_r_SG',
        'validated_sqrt7_r_SG', 'validated_sqrt12_r_SG',
        'Ri_over_R1_ratios_SG',
        'ratio_validated_sqrt3', 'ratio_validated_sqrt4',
        'ratio_validated_sqrt7', 'ratio_validated_sqrt12',
        'gaussian_fit_successful', 'gaussian_peak2_center',
        'gaussian_peak2_height', 'gaussian_peak3_center',
        'gaussian_peak3_height', 'gaussian_fit_r_squared',
        'q4', 'q6',
    ]
    column_order.extend(neighbor_type_cols)
    existing_columns = df_final_results.columns.tolist()
    filtered_column_order = [col for col in column_order if col in existing_columns]

    df_final_results = df_final_results.reindex(
        columns=filtered_column_order,
        fill_value=np.nan
    )

    # Save to CSV
    try:
        output_csv_path = params.get(
            'output_csv',
            f'analysis_results_{analysis_level_name}.csv'
        )
        df_final_results.to_csv(output_csv_path, index=False, na_rep='NaN')
        logging.info(f"Final results saved to {output_csv_path}")
    except Exception as e:
        logging.error(f"Error saving final results CSV: {e}")


# --- Main Pipeline Orchestrator ---
def run_pipeline() -> None:
    """
    Execute the full analysis pipeline:
    - Load configuration
    - Process snapshots
    - Perform temporal averaging
    - Run peak and spatial analyses
    - Export results to CSV
    """
    start_time = time.time()
    logging.info("Starting structural analysis pipeline.")

    # Load configuration
    params = load_config()

    # Create the outputs/ directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)

    # Precompute RDF bins and volumes
    bins_for_rdf_calc = rdf.create_adaptive_bins(
        params["R_MIN"],
        params["R_MAX"],
        params["NUM_BINS"]
    )
    bin_volumes_for_rdf_calc = 4 / 3 * np.pi * (
        bins_for_rdf_calc[1:]**3 - bins_for_rdf_calc[:-1]**3
    )

    analysis_level = params.get('ANALYSIS_LEVEL', 'temporal_first_spatial')
    params['analysis_level'] = analysis_level
    logging.info(f"Analysis level set to: {analysis_level}")

    snapshot_dir = params.get('SNAPSHOT_DIRECTORY', 'files/')
    list_of_snapshot_files = sorted([
        os.path.join(snapshot_dir, f)
        for f in os.listdir(snapshot_dir)
        if f.endswith('.lammpstrj')
    ])

    if not list_of_snapshot_files:
        logging.error(
            f"No LAMMPS dump files found in {snapshot_dir}. "
            "Please check SNAPSHOT_DIRECTORY in config.yaml."
        )
        return

    logging.info(f"Found {len(list_of_snapshot_files)} snapshot files.")

    # --- Step 1: Process snapshots ---
    logging.info(
        f"Processing {len(list_of_snapshot_files)} snapshots in parallel "
        "to extract per-atom g_i(r) and Voronoi data."
    )

    all_snapshots_processed_data = process_snapshots_in_chunks(
        snapshot_paths=list_of_snapshot_files,
        chunk_size=params.get("CHUNK_SIZE", 2),
        max_workers=multiprocessing.cpu_count(),
        params=params,
        bins_for_rdf_calc=bins_for_rdf_calc,
        bin_volumes_for_rdf_calc=bin_volumes_for_rdf_calc
    )

    if not all_snapshots_processed_data:
        logging.error("No snapshot data processed. Exiting pipeline.")
        return

    # --- Step 2: Temporal averaging ---
    logging.info("Performing temporal averaging of g_i(r) and Voronoi volume.")
    initial_atom_data_list, consolidated_metadata = (
        temporal_averaging.perform_temporal_averaging(all_snapshots_processed_data)
    )

    if not initial_atom_data_list:
        logging.error("No atoms remain after temporal averaging. Exiting pipeline.")
        return

    # Precompute RDF bin centers
    r_values_bin_centers = peak_analysis.calculate_rdf_bin_centers(
        np.array(consolidated_metadata.bins)
    )

    # --- Step 3: Conditional analysis ---
    if analysis_level == 'temporal_only':
        logging.info("Performing peak analysis on time-averaged RDFs (Temporal Only).")
        final_processed_atom_results = []

        for atom_data in initial_atom_data_list:
            peak_results = peak_analysis.analyze_rdf_peaks(
                r_values_bins=r_values_bin_centers,
                rdf_data_array=atom_data['g_i_r_temporal_avg'],
                analysis_parameters=params,
                atom_id=atom_data['id']
            )

            result_entry = {**atom_data, **peak_results}
            if isinstance(result_entry['g_i_r_temporal_avg'], np.ndarray):
                result_entry['g_i_r_temporal_avg'] = (
                    result_entry['g_i_r_temporal_avg'].tolist()
                )
            final_processed_atom_results.append(result_entry)

        logging.info(f"Completed peak analysis for {len(final_processed_atom_results)} atoms.")
        csv_export(initial_atom_data_list, final_processed_atom_results, params, analysis_level)

    elif analysis_level == 'temporal_first_spatial':
        logging.info("Performing first-level spatial averaging and peak analysis.")
        final_processed_atom_results = spatial_analysis_levels.perform_temporal_first_spatial_analysis(
            initial_atom_data_list,
            r_values_bin_centers,
            params
        )
        csv_export(initial_atom_data_list, final_processed_atom_results, params, analysis_level)

    end_time = time.time()
    logging.info(f"Total pipeline execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn', force=True)
    run_pipeline()
