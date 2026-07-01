"""
spatial_analysis_levels.py
--------------------------

This module performs a **temporal + first-level spatial analysis** of radial distribution
functions (RDFs) for atomic systems, followed by peak detection to extract structural
metrics.
"""

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed


import numpy as np

import peak_analysis


def perform_temporal_first_spatial_analysis(initial_atom_data_list, r_values_bin_centers, params):
    """
    Perform first-level spatial averaging on time-averaged RDFs, followed by peak analysis.
    """
    logging.info("Performing first-level spatial averaging on time-averaged RDFs.")
    num_bins = params.get("NUM_BINS", 100)
    r_values = np.asarray(r_values_bin_centers, dtype=float)

    # Step 1: Pre-convert all RDFs to a 2D numpy array
    num_atoms = len(initial_atom_data_list)
    rdf_matrix = np.empty((num_atoms, num_bins), dtype=np.float64)
    atom_ids = np.empty(num_atoms, dtype=np.int64)
    base_metadata = []

    for i, atom_data in enumerate(initial_atom_data_list):
        raw_rdf = atom_data['g_i_r_temporal_avg']
        if isinstance(raw_rdf, str):
            rdf_arr = np.array(json.loads(raw_rdf), dtype=np.float64)
        elif isinstance(raw_rdf, (list, np.ndarray)):
            rdf_arr = np.asarray(raw_rdf, dtype=np.float64)
        else:
            rdf_arr = np.zeros(num_bins, dtype=np.float64)

        rdf_matrix[i] = rdf_arr
        atom_ids[i] = int(atom_data['id'])
        base_metadata.append({
            'id': int(atom_data['id']),
            'type': atom_data.get('type'),
            'x': atom_data.get('x'),
            'y': atom_data.get('y'),
            'z': atom_data.get('z'),
            'radius': atom_data.get('radius'),
            'voronoi_volume_temporal': atom_data.get('voronoi_volume_temporal'),
            'CN_temporal': atom_data.get('CN_temporal'),
            'neighbors_by_type_temporal': atom_data.get('neighbors_by_type_temporal'),
            'q4_temporal': atom_data.get('q4_temporal'),
            'q6_temporal': atom_data.get('q6_temporal'),
            'q4_avg_temporal': atom_data.get('q4_avg_temporal'),
            'q6_avg_temporal': atom_data.get('q6_avg_temporal'),
            'n3_temporal': atom_data.get('n3_temporal'),
            'n4_temporal': atom_data.get('n4_temporal'),
            'n5_temporal': atom_data.get('n5_temporal'),
            'n6_temporal': atom_data.get('n6_temporal'),
            'pentagon_fraction_temporal': atom_data.get('pentagon_fraction_temporal'),
            'asphericity_temporal': atom_data.get('asphericity_temporal'),
            'asphericity_std_temporal': atom_data.get('asphericity_std_temporal'),
            's2_entropy_temporal': atom_data.get('s2_entropy_temporal'),
            's2_entropy_avg_temporal': atom_data.get('s2_entropy_avg_temporal'),
            'isb_temporal': atom_data.get('isb_temporal'),
            'mean_neighbor_volume_temporal': atom_data.get('mean_neighbor_volume_temporal'),
            'std_neighbor_volume_temporal': atom_data.get('std_neighbor_volume_temporal'),
            'mean_neighbor_pentagon_fraction_temporal': atom_data.get('mean_neighbor_pentagon_fraction_temporal'),
            'std_neighbor_pentagon_fraction_temporal': atom_data.get('std_neighbor_pentagon_fraction_temporal'),
            'mean_neighbor_CN_temporal': atom_data.get('mean_neighbor_CN_temporal'),
            'mean_neighbor_free_volume_temporal': atom_data.get('mean_neighbor_free_volume_temporal'),
            'mean_neighbor_asphericity_temporal': atom_data.get('mean_neighbor_asphericity_temporal'),
            'mean_2nd_neighbor_volume_temporal': atom_data.get('mean_2nd_neighbor_volume_temporal'),
            'std_2nd_neighbor_volume_temporal': atom_data.get('std_2nd_neighbor_volume_temporal'),
            'mean_2nd_neighbor_free_volume_temporal': atom_data.get('mean_2nd_neighbor_free_volume_temporal'),
            'mean_2nd_neighbor_pentagon_fraction_temporal': atom_data.get('mean_2nd_neighbor_pentagon_fraction_temporal'),
            'mean_2nd_neighbor_CN_temporal': atom_data.get('mean_2nd_neighbor_CN_temporal'),
            'cu_fraction_2nd_shell_temporal': atom_data.get('cu_fraction_2nd_shell_temporal'),
            'zr_fraction_2nd_shell_temporal': atom_data.get('zr_fraction_2nd_shell_temporal'),
            'mean_3rd_neighbor_volume_temporal': atom_data.get('mean_3rd_neighbor_volume_temporal'),
            'std_3rd_neighbor_volume_temporal': atom_data.get('std_3rd_neighbor_volume_temporal'),
            'mean_3rd_neighbor_free_volume_temporal': atom_data.get('mean_3rd_neighbor_free_volume_temporal'),
            'mean_3rd_neighbor_pentagon_fraction_temporal': atom_data.get('mean_3rd_neighbor_pentagon_fraction_temporal'),
            'mean_3rd_neighbor_CN_temporal': atom_data.get('mean_3rd_neighbor_CN_temporal'),
            'cu_fraction_3rd_shell_temporal': atom_data.get('cu_fraction_3rd_shell_temporal'),
            'zr_fraction_3rd_shell_temporal': atom_data.get('zr_fraction_3rd_shell_temporal'),
            'csro_unlike_temporal': atom_data.get('csro_unlike_temporal'),
            'csro_like_temporal': atom_data.get('csro_like_temporal'),
            'csro_unlike_std_temporal': atom_data.get('csro_unlike_std_temporal'),
            'mean_angle_all_temporal': atom_data.get('mean_angle_all_temporal'),
            'std_angle_all_temporal': atom_data.get('std_angle_all_temporal'),
            'skewness_angle_all_temporal': atom_data.get('skewness_angle_all_temporal'),
            'mean_angle_liketype_temporal': atom_data.get('mean_angle_liketype_temporal'),
            'mean_angle_unliketype_temporal': atom_data.get('mean_angle_unliketype_temporal'),
            'mean_angle_mixedtype_temporal': atom_data.get('mean_angle_mixedtype_temporal'),
        })

    # Build id -> index lookup
    id_to_idx = {int(atom_ids[i]): i for i in range(num_atoms)}

    # Step 2: Spatial averaging
    spatially_avg_rdf_matrix = np.empty_like(rdf_matrix)
    for i, atom_data in enumerate(initial_atom_data_list):
        central_rdf = rdf_matrix[i]
        neighbor_ids_raw = atom_data.get('vor_neighbors', [])
        valid_neighbor_indices = [
            id_to_idx[int(nid)] for nid in neighbor_ids_raw if int(nid) in id_to_idx
        ]
        if valid_neighbor_indices:
            neighbor_sum = rdf_matrix[valid_neighbor_indices].sum(axis=0)
            spatially_avg_rdf_matrix[i] = (central_rdf + neighbor_sum) / (1 + len(valid_neighbor_indices))
        else:
            spatially_avg_rdf_matrix[i] = central_rdf

    del rdf_matrix

    # Step 3: Parallel peak analysis
    max_workers = params.get('MAX_WORKERS_SPATIAL', None)
    if max_workers is None:
        import multiprocessing
        max_workers = max(1, multiprocessing.cpu_count() - 1)

    chunk_size = max(1, min(256, num_atoms // (max_workers * 4)))
    r_values_list = r_values.tolist()
    sample_atom_id = params.get('PLOT_SAMPLE_ATOM_ID', None)

    main_process_ids = set()
    if sample_atom_id is not None:
        main_process_ids.add(int(sample_atom_id))

    worker_chunks = []
    main_process_batch = []

    for i, atom_id in enumerate(atom_ids):
        aid = int(atom_id)
        worker_payload = (
            aid,
            spatially_avg_rdf_matrix[i].tolist(),
            r_values_list,
            params,
        )
        if aid in main_process_ids:
            main_process_batch.append(worker_payload)
        else:
            worker_chunks.append(worker_payload)

    final_processed_atom_results = []

    if main_process_batch:
        logging.info(f"Processing sample atom {sample_atom_id} in main process.")
        for payload in main_process_batch:
            result = _analyze_one_atom(payload)
            final_processed_atom_results.append(result)

    if worker_chunks:
        chunked = [worker_chunks[i:i + chunk_size] for i in range(0, len(worker_chunks), chunk_size)]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_chunk, chunk) for chunk in chunked]
            for future in as_completed(futures):
                try:
                    chunk_results = future.result(timeout=params.get('SPATIAL_TIMEOUT', 600))
                    final_processed_atom_results.extend(chunk_results)
                except Exception as exc:
                    logging.error(f"Spatial analysis chunk failed: {exc}", exc_info=True)

    logging.info(f"Completed temporal-first-spatial analysis for {len(final_processed_atom_results)} atoms.")

    # Step 4: Restore metadata
    base_metadata_map = {m['id']: m for m in base_metadata}
    full_results = []
    for result in final_processed_atom_results:
        atom_id = result['id']
        meta = dict(base_metadata_map.get(atom_id, {}))
        meta.update(result)
        meta['g_i_r_temporal_avg'] = json.dumps(
            np.around(meta.get('g_i_r_temporal_avg', np.zeros(num_bins)), 4).tolist()
        )
        full_results.append(meta)

    return full_results


def _analyze_one_atom(worker_payload):
    """Run peak analysis for a single atom."""
    atom_id, rdf_list, r_values_list, params = worker_payload
    rdf_array = np.array(rdf_list, dtype=float)
    r_values = np.array(r_values_list, dtype=float)
    g_i_r = {"id": atom_id, "g_i_r_temporal_avg": rdf_array.tolist()}
    result = peak_analysis.process_atom_rdf(r_values, g_i_r, params)
    result['id'] = atom_id
    return result


def _process_chunk(chunk_args):
    """Process a chunk of atoms in one worker process."""
    return [_analyze_one_atom(args) for args in chunk_args]