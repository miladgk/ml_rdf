"""
spatial_analysis_levels.py
--------------------------

This module performs a **temporal + first-level spatial analysis** of radial distribution functions (RDFs) 
for atomic systems, followed by peak detection to extract structural metrics. It is designed for use in 
materials science and computational chemistry workflows, where structural information about disordered 
systems (e.g., metallic glasses, liquids) is needed.

Workflow:
1. Each atom’s time-averaged RDF is retrieved.
2. A first-level spatial average is computed by combining the central atom’s RDF with those of its Voronoi neighbors.
3. Automated peak detection is performed on the spatially averaged RDF to extract structural features.
4. Results are packaged into dictionaries for downstream processing or export.

Key Features:
- Efficient atom lookup via an ID-indexed map for fast neighbor retrieval.
- Handles cases where atoms have no Voronoi neighbors.
- Converts NumPy arrays to Python lists for safe JSON/CSV serialization.
- Uses external `peak_analysis` module for automated RDF peak detection.

Dependencies:
- **pandas** and **NumPy** for numerical operations.
- **logging** for diagnostics and tracking.
- **peak_analysis** (custom module) for structural feature extraction.

Intended Use:
This module integrates into larger simulation post-processing pipelines, 
particularly for workflows analyzing atomic structure using RDFs and Voronoi tessellations.
"""

import numpy as np
import logging

# Custom Modules
import peak_analysis


def perform_temporal_first_spatial_analysis(initial_atom_data_list, r_values_bin_centers, params):
    """
    Perform first-level spatial averaging on time-averaged RDFs for each atom, 
    followed by peak analysis.

    Parameters
    ----------
    initial_atom_data_list : list of dict
        List of per-atom dictionaries containing at least:
        - 'id' (int): Atom identifier.
        - 'type' (int or str): Atom type/classification.
        - 'g_i_r_temporal_avg' (np.ndarray): Time-averaged RDF values for the atom.
        - 'vor_neighbors' (list[int], optional): IDs of Voronoi neighbors.
    r_values_bin_centers : np.ndarray
        Array of RDF bin center values corresponding to RDF values.
    params : dict
        Dictionary of configuration/analysis parameters for peak detection.

    Returns
    -------
    list of dict
        A list of dictionaries containing:
        - Original atom data.
        - First-level spatially averaged RDF (used for peak analysis).
        - Peak analysis results.
        Arrays are converted to lists for JSON/CSV serialization compatibility.
    """
    logging.info("Performing first-level spatial averaging on time-averaged RDFs.")
    final_processed_atom_results = []

    # Map atom data by ID for efficient O(1) neighbor lookups
    atom_data_map = {atom['id']: atom for atom in initial_atom_data_list}

    for atom_data in initial_atom_data_list:
        atom_id = atom_data['id']
        atom_type = atom_data['type']

        # Retrieve the central atom's temporal RDF
        central_rdf = atom_data['g_i_r_temporal_avg']

        # Initialize neighbor RDF sum
        neighbor_rdf_sum = np.zeros_like(central_rdf, dtype=float)
        neighbor_ids = atom_data.get('vor_neighbors', [])
        num_neighbors = len(neighbor_ids)

        if num_neighbors > 0:
            for neighbor_id in neighbor_ids:
                neighbor_data = atom_data_map.get(neighbor_id)
                if neighbor_data:
                    neighbor_rdf_sum += neighbor_data['g_i_r_temporal_avg']
                else:
                    logging.warning(
                        f"Neighbor ID {neighbor_id} for atom {atom_id} not found in the data map."
                    )

            # Compute first-level spatial average RDF
            # Formula: g_i^first(r) = (g_i^temporal(r) + Σ g_j^temporal(r)) / (1 + N_i)
            first_avg_rdf = (central_rdf + neighbor_rdf_sum) / (1 + num_neighbors)
        else:
            # If atom has no neighbors, use its temporal RDF as the spatial average
            first_avg_rdf = central_rdf
            logging.debug(
                f"Atom {atom_id} has no neighbors. Using its temporal average as its spatial average."
            )

        # Perform peak analysis on the spatially averaged RDF
        logging.debug(
            f"Starting peak analysis for atom {atom_id} (type {atom_type}) on spatially-averaged RDF."
        )
        peak_results = peak_analysis.analyze_rdf_peaks(
            r_values_bins=r_values_bin_centers,
            rdf_data_array=first_avg_rdf,
            analysis_parameters=params,
            atom_id=atom_id
        )

        # Combine results into a single dictionary
        result_entry = {
            **atom_data,
            **peak_results,
        }

        # Convert NumPy arrays to lists for serialization compatibility
        if isinstance(result_entry['g_i_r_temporal_avg'], np.ndarray):
            result_entry['g_i_r_temporal_avg'] = result_entry['g_i_r_temporal_avg'].tolist()

        final_processed_atom_results.append(result_entry)

    logging.info(
        f"Completed temporal-first-spatial analysis for {len(final_processed_atom_results)} atoms."
    )
    return final_processed_atom_results
