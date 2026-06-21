"""
temporal_averaging.py
---------------------

Module for performing temporal averaging of atom-level structural and volumetric properties
across multiple molecular dynamics or Monte Carlo simulation snapshots.

This module processes per-snapshot atomic data—including radial distribution functions (g_i(r)),
Voronoi volumes, coordination numbers (CN), Steinhardt bond order parameters (q4, q6), and
neighbor-type counts—and produces temporally averaged quantities for downstream statistical
analysis or structural characterization. The module also consolidates relevant simulation
metadata (box size, density, binning) for consistency across processing stages.

Key functionalities:
- Aggregates and temporally averages g_i(r) per atom.
- Computes average Voronoi volumes and coordination numbers across snapshots.
- Calculates mean Steinhardt bond order parameters (q4, q6) for each atom.
- Averages neighbor-type counts across snapshots.
- Maintains consistent simulation metadata.
- Logs warnings for missing or incomplete data.

This module is designed to be part of a larger simulation analysis pipeline and is
not intended for standalone execution.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd

# Configure logging consistently across modules
# NOTE: Do NOT use force=True here; pipeline_orchestrator.py handles initial logging setup.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def perform_temporal_averaging(all_snapshots_processed_data):
    """
    Perform temporal averaging of atomic structural and volumetric properties
    across multiple simulation snapshots.

    Parameters
    ----------
    all_snapshots_processed_data : list of dict
        Each element corresponds to a processed snapshot and must contain:
        - 'atom_data_snapshot': list of atom dictionaries with keys:
            - 'id', 'type', 'x', 'y', 'z', 'radius'
            - 'g_i_r_snapshot': per-atom radial distribution function (array-like)
            - 'voronoi_volume': float or None
            - 'num_neighbors': int or None
            - 'neighbors_by_type': dict of neighbor counts per type
            - 'q4', 'q6': float or array-like
        - 'snapshot_metadata': dict with simulation metadata:
            - 'box_size', 'total_volume', 'density', 'lower_bounds', 'bins', 'bin_volumes'

    Returns
    -------
    final_atom_data_list : list of dict
        List of atom dictionaries with temporally averaged properties:
        - 'g_i_r_temporal_avg', 'voronoi_volume_temporal', 'CN_temporal'
        - 'neighbors_by_type_temporal', 'q4_temporal', 'q6_temporal'
        - original metadata: 'id', 'type', 'x', 'y', 'z', 'radius'

    consolidated_metadata : types.SimpleNamespace
        Metadata consolidated from the first valid snapshot (box size, volume, density, bins).
    """

    logging.info(
        "Starting temporal averaging of g_i(r), Voronoi volume, CN, "
        "neighbor-type counts, and Steinhardt parameters across snapshots."
    )

    consolidated_metadata = None
    flat_atom_records = []

    for snapshot_data in all_snapshots_processed_data:
        if snapshot_data is None:
            logging.warning("Skipping a snapshot as its processing returned None.")
            continue

        atom_data_snapshot = snapshot_data.get('atom_data_snapshot', [])
        snapshot_metadata = snapshot_data.get('snapshot_metadata', {})

        if consolidated_metadata is None:
            consolidated_metadata = SimpleNamespace(
                box_size=snapshot_metadata.get('box_size', []),
                total_volume=snapshot_metadata.get('total_volume', 0.0),
                density=snapshot_metadata.get('density', 0.0),
                lower_bounds=snapshot_metadata.get('lower_bounds', []),
                bins=snapshot_metadata.get('bins', []),
                bin_volumes=snapshot_metadata.get('bin_volumes', [])
            )

        flat_atom_records.extend(atom_data_snapshot)

    if not flat_atom_records:
        logging.error("No valid snapshot data processed. Temporal averaging cannot proceed.")
        return [], SimpleNamespace()

    atom_frame = pd.DataFrame(flat_atom_records)
    atom_frame = atom_frame.copy()

    rdf_arrays = np.asarray(
        [np.asarray(x, dtype=float) for x in atom_frame['g_i_r_snapshot'].tolist()],
        dtype=float,
    )
    atom_frame['g_i_r_snapshot'] = list(rdf_arrays)

    def _mean_scalar(value):
        arr = np.asarray(value, dtype=float)
        return float(arr.mean()) if arr.ndim > 0 else float(arr)

    q4_values = np.asarray(atom_frame['q4'].tolist(), dtype=object)
    q6_values = np.asarray(atom_frame['q6'].tolist(), dtype=object)
    w4_values = np.asarray(atom_frame['w4'].tolist(), dtype=object)
    w6_values = np.asarray(atom_frame['w6'].tolist(), dtype=object)
    q4_avg_values = np.asarray(atom_frame['q4_avg'].tolist(), dtype=object)
    q6_avg_values = np.asarray(atom_frame['q6_avg'].tolist(), dtype=object)

    q4_mean = np.vectorize(lambda value: float(np.asarray(value, dtype=float).mean()), otypes=[float])
    q6_mean = np.vectorize(lambda value: float(np.asarray(value, dtype=float).mean()), otypes=[float])
    w4_mean = np.vectorize(lambda value: float(np.asarray(value, dtype=float).mean()), otypes=[float])
    w6_mean = np.vectorize(lambda value: float(np.asarray(value, dtype=float).mean()), otypes=[float])
    q4_avg_mean = np.vectorize(lambda value: float(np.asarray(value, dtype=float).mean()), otypes=[float])
    q6_avg_mean = np.vectorize(lambda value: float(np.asarray(value, dtype=float).mean()), otypes=[float])

    atom_frame['q4'] = q4_mean(q4_values)
    atom_frame['q6'] = q6_mean(q6_values)
    atom_frame['w4'] = w4_mean(w4_values)
    atom_frame['w6'] = w6_mean(w6_values)
    atom_frame['q4_avg'] = q4_avg_mean(q4_avg_values)
    atom_frame['q6_avg'] = q6_avg_mean(q6_avg_values)

    grouped_atoms = atom_frame.groupby('id', sort=False, observed=True)
    q4_avg_over_time = grouped_atoms['q4'].mean().to_dict()
    q6_avg_over_time = grouped_atoms['q6'].mean().to_dict()
    w4_avg_over_time = grouped_atoms['w4'].mean().to_dict()
    w6_avg_over_time = grouped_atoms['w6'].mean().to_dict()
    q4_avg_avg_over_time = grouped_atoms['q4_avg'].mean().to_dict()
    q6_avg_avg_over_time = grouped_atoms['q6_avg'].mean().to_dict()

    atom_ids = atom_frame['id'].to_numpy(dtype=int)
    unique_ids, inverse = np.unique(atom_ids, return_inverse=True)
    rdf_sum = np.zeros((len(unique_ids), rdf_arrays.shape[1]), dtype=float)
    np.add.at(rdf_sum, inverse, rdf_arrays)
    rdf_counts = np.bincount(inverse).astype(float)
    g_i_r_avg_over_time = {
        int(atom_id): rdf_sum[idx] / rdf_counts[idx]
        for idx, atom_id in enumerate(unique_ids)
    }

    volume_avg_over_time = grouped_atoms['voronoi_volume'].mean().to_dict()
    CN_avg_over_time = grouped_atoms['num_neighbors'].mean().to_dict()
    n3_avg_over_time = grouped_atoms['n3_voronoi'].mean().to_dict()
    n4_avg_over_time = grouped_atoms['n4_voronoi'].mean().to_dict()
    n5_avg_over_time = grouped_atoms['n5_voronoi'].mean().to_dict()
    n6_avg_over_time = grouped_atoms['n6_voronoi'].mean().to_dict()

    neighbor_records = pd.DataFrame.from_records(
        atom_frame['neighbors_by_type'].apply(lambda x: dict(x) if isinstance(x, dict) else {}).tolist(),
        index=atom_frame['id'],
    ).fillna(0.0)

    neighbor_avg_over_time = neighbor_records.groupby(level=0, sort=False, observed=True).mean()
    neighbor_type_union = set(neighbor_avg_over_time.columns.tolist())
    neighbors_by_type_avg_over_time = {
        int(atom_id): neighbor_avg_over_time.loc[atom_id].to_dict()
        for atom_id in neighbor_avg_over_time.index
    }

    all_neighbor_types = sorted(neighbor_type_union)

    if consolidated_metadata is None:
        logging.error("No valid snapshot data processed. Temporal averaging cannot proceed.")
        return [], SimpleNamespace()

    missing_g_i_r_atoms = [atom_id for atom_id in g_i_r_avg_over_time if not np.isfinite(g_i_r_avg_over_time[atom_id]).all()]
    missing_volume_atoms = [atom_id for atom_id in volume_avg_over_time if pd.isna(volume_avg_over_time[atom_id])]
    missing_CN_atoms = [atom_id for atom_id in CN_avg_over_time if pd.isna(CN_avg_over_time[atom_id])]
    missing_neighbors_by_type_atoms = [atom_id for atom_id in neighbors_by_type_avg_over_time if not neighbors_by_type_avg_over_time[atom_id]]

    # Log summary of missing data
    if missing_g_i_r_atoms:
        logging.warning(f"Skipped {len(missing_g_i_r_atoms)} atoms due to missing g_i(r).")
    if missing_volume_atoms:
        logging.warning(f"Skipped {len(missing_volume_atoms)} atoms due to missing volume.")
    if missing_neighbors_by_type_atoms:
        logging.warning(f"Skipped {len(missing_neighbors_by_type_atoms)} atoms due to missing neighbors_by_type.")
    if missing_CN_atoms:
        logging.warning(f"Skipped {len(missing_CN_atoms)} atoms due to missing CN.")

    logging.info("Temporal averaging complete. Constructing final atom data list.")

    metadata_reference = {
        int(atom['id']): {
            'id': int(atom['id']),
            'type': atom['type'],
            'x': atom['x'],
            'y': atom['y'],
            'z': atom['z'],
            'radius': atom['radius'],
        }
        for atom in flat_atom_records
    }

    final_atom_data_list = []
    for atom_id, metadata in metadata_reference.items():
        if atom_id in g_i_r_avg_over_time and atom_id in volume_avg_over_time and atom_id in CN_avg_over_time:
            averaged_entry = metadata.copy()
            averaged_entry['g_i_r_temporal_avg'] = g_i_r_avg_over_time[atom_id]
            averaged_entry['voronoi_volume_temporal'] = volume_avg_over_time[atom_id]
            averaged_entry['neighbors_by_type_temporal'] = {
                neighbor_type: neighbors_by_type_avg_over_time.get(atom_id, {}).get(neighbor_type, 0.0)
                for neighbor_type in all_neighbor_types
            }
            averaged_entry['CN_temporal'] = CN_avg_over_time[atom_id]
            averaged_entry['q4_temporal'] = q4_avg_over_time.get(atom_id, np.nan)
            averaged_entry['q6_temporal'] = q6_avg_over_time.get(atom_id, np.nan)
            averaged_entry['w4_temporal'] = w4_avg_over_time.get(atom_id, np.nan)
            averaged_entry['w6_temporal'] = w6_avg_over_time.get(atom_id, np.nan)
            averaged_entry['q4_avg_temporal'] = q4_avg_avg_over_time.get(atom_id, np.nan)
            averaged_entry['q6_avg_temporal'] = q6_avg_avg_over_time.get(atom_id, np.nan)
            averaged_entry['n3_temporal'] = n3_avg_over_time.get(atom_id, np.nan)
            averaged_entry['n4_temporal'] = n4_avg_over_time.get(atom_id, np.nan)
            averaged_entry['n5_temporal'] = n5_avg_over_time.get(atom_id, np.nan)
            averaged_entry['n6_temporal'] = n6_avg_over_time.get(atom_id, np.nan)
            # Pentagon fraction: n5 / (n3 + n4 + n5 + n6) with 1e-8 guard
            n_sum = (n3_avg_over_time.get(atom_id, 0) + n4_avg_over_time.get(atom_id, 0) +
                     n5_avg_over_time.get(atom_id, 0) + n6_avg_over_time.get(atom_id, 0))
            averaged_entry['pentagon_fraction_temporal'] = (
                n5_avg_over_time.get(atom_id, 0) / (n_sum + 1e-8)
            )
            final_atom_data_list.append(averaged_entry)
        else:
            logging.warning(
                f"Atom ID {atom_id} skipped from final analysis due to incomplete temporal data."
            )

    logging.info(f"Temporal averaging produced data for {len(final_atom_data_list)} atoms.")

    return final_atom_data_list, consolidated_metadata


if __name__ == '__main__':
    print("This module is intended to be imported as part of a larger pipeline.")
    print("Run 'pipeline_orchestrator.py' to execute the full workflow.")
