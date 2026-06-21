"""
pipeline_orchestrator.py
=========================

This module orchestrates a complete structural analysis workflow for atomistic
simulation data, integrating snapshot processing, temporal averaging, peak
detection, and optional spatial analysis into a configurable, parallelized
pipeline.
"""

import os
import time
import json
import yaml
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from functools import partial
import numpy as np
import pandas as pd

import spatial_analysis_levels
import peak_analysis
import snapshot_processor
import temporal_averaging
import rdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)


def load_config(config_path: str = "config.yaml") -> dict:
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


def resolve_snapshot_paths(snapshot_dir: str, config_path: str | None = None) -> list:
    snapshot_dir_resolved = snapshot_dir
    if config_path and snapshot_dir and not os.path.isabs(snapshot_dir):
        config_dir = os.path.dirname(os.path.abspath(config_path))
        snapshot_dir_resolved = os.path.normpath(os.path.join(config_dir, snapshot_dir))
    snapshot_files = []
    if snapshot_dir_resolved and os.path.isdir(snapshot_dir_resolved):
        for file_name in sorted(os.listdir(snapshot_dir_resolved)):
            if file_name.endswith('.lammpstrj'):
                full_path = os.path.join(snapshot_dir_resolved, file_name)
                snapshot_files.append(full_path)
    else:
        logging.warning(f"Snapshot directory '{snapshot_dir_resolved}' does not exist or is not a directory.")
    return snapshot_files


def _process_snapshot_worker(snapshot_file, params, bins_for_rdf_calc, bin_volumes_for_rdf_calc):
    return snapshot_processor.process_single_snapshot(snapshot_file, params, bins_for_rdf_calc, bin_volumes_for_rdf_calc)


def process_snapshots_in_chunks(snapshot_paths, chunk_size, max_workers, params, bins_for_rdf_calc, bin_volumes_for_rdf_calc):
    results = []
    total = len(snapshot_paths)
    chunk_size = min(chunk_size, max(1, len(snapshot_paths) // max_workers))
    if total > 10:
        logging.info(f"Processing {total} snapshots in {max_workers} parallel workers...")
    if len(snapshot_paths) <= 1 and total > 0:
        for snapshot in snapshot_paths:
            try:
                result = _process_snapshot_worker(snapshot, params, bins_for_rdf_calc, bin_volumes_for_rdf_calc)
                if result:
                    results.append(result)
            except Exception as exc:
                logging.error(f"Error processing snapshot {snapshot}: {exc}", exc_info=True)
        return results
    worker = partial(_process_snapshot_worker, params=params, bins_for_rdf_calc=bins_for_rdf_calc, bin_volumes_for_rdf_calc=bin_volumes_for_rdf_calc)
    with ProcessPoolExecutor(max_workers=min(max_workers, total)) as executor:
        futures = [executor.submit(worker, snapshot) for snapshot in snapshot_paths]
        worker_timeout = params.get('WORKER_TIMEOUT', 300)
        pending = set(futures)
        try:
            for future in as_completed(pending, timeout=worker_timeout):
                pending.discard(future)
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as exc:
                    logging.error(f"Error processing snapshot: {exc}", exc_info=True)
        except FuturesTimeoutError:
            logging.warning(f"Worker timeout reached ({worker_timeout}s). Continuing with {len(pending)} remaining.")
            for future in pending:
                future.cancel()
        except Exception as exc:
            logging.error(f"Unexpected error in parallel processing: {exc}", exc_info=True)
    return results


def _json_fallback_handler(obj):
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
    if isinstance(obj, dict):
        return {str(k): convert_dict_keys_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_dict_keys_to_str(x) for x in obj]
    else:
        return obj


def csv_export(initial_atom_data_list, all_analysis_results, params, analysis_level_name):
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
        'fit_w1': 'gaussian_peak2_sigma',
        'fit_fwhm1': 'gaussian_peak2_fwhm',
        'fit_h2': 'gaussian_peak3_height',
        'fit_c2': 'gaussian_peak3_center',
        'fit_w2': 'gaussian_peak3_sigma',
        'fit_fwhm2': 'gaussian_peak3_fwhm',
        'fit_r2': 'gaussian_fit_r_squared'
    }

    final_atom_results_for_df = []
    temporal_avg_data_map = {d['id']: d for d in initial_atom_data_list}

    for result_atom_data in all_analysis_results:
        atom_id = result_atom_data['id']
        original_temporal_data = temporal_avg_data_map.get(atom_id, {})

        combined_data = {
            'id': original_temporal_data.get('id'),
            'type': original_temporal_data.get('type'),
            'x': original_temporal_data.get('x'),
            'y': original_temporal_data.get('y'),
            'z': original_temporal_data.get('z'),
            'radius': original_temporal_data.get('radius'),
            'voronoi_volume_temporal': original_temporal_data.get('voronoi_volume_temporal'),
            'CN_temporal': original_temporal_data.get('CN_temporal', []),
            'g_i_r_temporal_avg': json.dumps(np.around(original_temporal_data.get('g_i_r_temporal_avg', np.array([])), 4).tolist()),
            'q4': original_temporal_data.get('q4_temporal', np.nan),
            'q6': original_temporal_data.get('q6_temporal', np.nan),
            'w4': original_temporal_data.get('w4_temporal', np.nan),
            'w6': original_temporal_data.get('w6_temporal', np.nan),
            'q4_avg': original_temporal_data.get('q4_avg_temporal', np.nan),
            'q6_avg': original_temporal_data.get('q6_avg_temporal', np.nan),
        }

        neighbors_dict = original_temporal_data.get('neighbors_by_type_temporal', {})
        for n_type, count in neighbors_dict.items():
            combined_data[f'neighbor_{n_type}'] = count

        renamed_result_atom_data = {rename_keys_for_csv.get(key, key): value for key, value in result_atom_data.items()}

        for key, value in renamed_result_atom_data.items():
            if key in ['id', 'type', 'x', 'y', 'z', 'radius', 'voronoi_volume_temporal', 'CN_temporal', 'g_i_r_temporal_avg', 'idx']:
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
                logging.warning(f"Could not serialize key '{key}': {e}")
                combined_data[key] = str(value)

        final_atom_results_for_df.append(combined_data)

    logging.info("Creating final DataFrame and exporting results to CSV.")

    if final_atom_results_for_df:
        neighbor_type_cols = [col for col in final_atom_results_for_df[0] if col.startswith('neighbor_')]
        df_final_results = pd.DataFrame(final_atom_results_for_df)
    else:
        logging.warning(f"No final results generated for '{analysis_level_name}'. Output CSV will be empty.")
        neighbor_type_cols = []
        df_final_results = pd.DataFrame()

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
        'gaussian_peak2_height', 'gaussian_peak2_sigma',
        'gaussian_peak2_fwhm',
        'gaussian_peak3_center', 'gaussian_peak3_height',
        'gaussian_peak3_sigma', 'gaussian_peak3_fwhm',
        'gaussian_fit_r_squared',
        'q4', 'q6', 'w4', 'w6',
        'q4_avg', 'q6_avg',
    ]
    column_order.extend(neighbor_type_cols)
    existing_columns = df_final_results.columns.tolist()
    filtered_column_order = [col for col in column_order if col in existing_columns]

    df_final_results = df_final_results.reindex(columns=filtered_column_order, fill_value=np.nan)

    try:
        output_csv_path = params.get('output_csv', f'analysis_results_{analysis_level_name}.csv')
        df_final_results.to_csv(output_csv_path, index=False, na_rep='NaN')
        logging.info(f"Final results saved to {output_csv_path}")
    except Exception as e:
        logging.error(f"Error saving results CSV: {e}")


def run_pipeline(config_path: str | None = None) -> None:
    start_time = time.time()
    logging.info("Starting structural analysis pipeline.")
    cfg_path = config_path if config_path else "config.yaml"
    params = load_config(cfg_path)

    params_resolve_dir = os.getcwd()
    if config_path:
        config_dir = os.path.dirname(os.path.abspath(config_path))
        params_resolve_dir = config_dir

    if 'radius_file' in params and isinstance(params['radius_file'], str):
        rf = params['radius_file']
        if not os.path.isabs(rf):
            params['radius_file'] = os.path.normpath(os.path.join(params_resolve_dir, rf))

    if 'SNAPSHOT_DIRECTORY' in params and isinstance(params['SNAPSHOT_DIRECTORY'], str):
        sd = params['SNAPSHOT_DIRECTORY']
        if not os.path.isabs(sd):
            params['SNAPSHOT_DIRECTORY'] = os.path.normpath(os.path.join(params_resolve_dir, sd))

    os.makedirs("outputs", exist_ok=True)

    bins_for_rdf_calc = rdf.create_adaptive_bins(params["R_MIN"], params["R_MAX"], params["NUM_BINS"])
    bin_volumes_for_rdf_calc = 4 / 3 * np.pi * (bins_for_rdf_calc[1:]**3 - bins_for_rdf_calc[:-1]**3)

    analysis_level = params.get('ANALYSIS_LEVEL', 'temporal_first_spatial')
    params['analysis_level'] = analysis_level
    logging.info(f"Analysis level set to: {analysis_level}")

    snapshot_dir = params.get('SNAPSHOT_DIRECTORY', 'files/')
    list_of_snapshot_files = resolve_snapshot_paths(snapshot_dir, config_path=config_path)

    if not list_of_snapshot_files:
        logging.error(f"No LAMMPS dump files found in {snapshot_dir}.")
        return

    logging.info(f"Found {len(list_of_snapshot_files)} snapshot files.")

    logging.info(f"Processing {len(list_of_snapshot_files)} snapshots in parallel.")
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

    logging.info("Performing temporal averaging.")
    initial_atom_data_list, consolidated_metadata = temporal_averaging.perform_temporal_averaging(all_snapshots_processed_data)

    if not initial_atom_data_list:
        logging.error("No atoms remain after temporal averaging. Exiting pipeline.")
        return

    r_values_bin_centers = peak_analysis.calculate_rdf_bin_centers(np.array(consolidated_metadata.bins))

    if analysis_level == 'temporal_only':
        logging.info("Performing peak analysis (Temporal Only).")
        final_processed_atom_results = []
        for atom_data in initial_atom_data_list:
            peak_results = peak_analysis.analyze_rdf_peaks(r_values_bins=r_values_bin_centers, rdf_data_array=atom_data['g_i_r_temporal_avg'], analysis_parameters=params, atom_id=atom_data['id'])
            result_entry = {**atom_data, **peak_results}
            if isinstance(result_entry['g_i_r_temporal_avg'], np.ndarray):
                result_entry['g_i_r_temporal_avg'] = result_entry['g_i_r_temporal_avg'].tolist()
            final_processed_atom_results.append(result_entry)
        logging.info(f"Completed peak analysis for {len(final_processed_atom_results)} atoms.")
        csv_export(initial_atom_data_list, final_processed_atom_results, params, analysis_level)

    elif analysis_level == 'temporal_first_spatial':
        logging.info("Performing first-level spatial averaging and peak analysis.")
        final_processed_atom_results = spatial_analysis_levels.perform_temporal_first_spatial_analysis(initial_atom_data_list, r_values_bin_centers, params)
        csv_export(initial_atom_data_list, final_processed_atom_results, params, analysis_level)

    end_time = time.time()
    logging.info(f"Total pipeline execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Atomic_Simulation_Post-Processing_Pipeline")
    parser.add_argument("--config", default=None, help="Path to config YAML.")
    args = parser.parse_args()
    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn', force=True)
    run_pipeline(config_path=args.config)