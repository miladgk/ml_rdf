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
from collections import defaultdict, Counter
from types import SimpleNamespace
import numpy as np

# Configure logging consistently across modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True
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

    # Initialize accumulators for averaging
    g_i_r_accumulator = defaultdict(list)
    volume_accumulator = defaultdict(list)
    CN_accumulator = defaultdict(list)
    neighbors_by_type_accumulator = defaultdict(list)
    q4_accumulator = defaultdict(list)
    q6_accumulator = defaultdict(list)
    metadata_reference = {}  # Stores per-atom metadata from first encountered snapshot

    consolidated_metadata = None
    all_neighbor_types = set()

    for snapshot_data in all_snapshots_processed_data:
        if snapshot_data is None:
            logging.warning("Skipping a snapshot as its processing returned None.")
            continue

        atom_data_snapshot = snapshot_data['atom_data_snapshot']
        snapshot_metadata = snapshot_data['snapshot_metadata']

        # Consolidate simulation metadata from the first valid snapshot
        if consolidated_metadata is None:
            consolidated_metadata = SimpleNamespace(
                box_size=snapshot_metadata['box_size'],
                total_volume=snapshot_metadata['total_volume'],
                density=snapshot_metadata['density'],
                lower_bounds=snapshot_metadata['lower_bounds'],
                bins=snapshot_metadata['bins'],
                bin_volumes=snapshot_metadata['bin_volumes']
            )

        for atom in atom_data_snapshot:
            atom_id = atom['id']
            all_neighbor_types.update(atom.get('neighbors_by_type', {}).keys())

            # Accumulate radial distribution function
            g_i_r_accumulator[atom_id].append(np.array(atom['g_i_r_snapshot']))

            # Accumulate Voronoi volume if available
            if atom['voronoi_volume'] is not None:
                volume_accumulator[atom_id].append(atom['voronoi_volume'])
            else:
                logging.warning(
                    f"Atom ID {atom_id} has no valid Voronoi volume in this snapshot."
                )

            # Accumulate coordination number if available
            if atom['num_neighbors'] is not None:
                CN_accumulator[atom_id].append(atom['num_neighbors'])
            else:
                logging.warning(f"Atom ID {atom_id} has no valid CN in this snapshot.")

            # Accumulate neighbor-type counts
            if atom.get('neighbors_by_type') is not None:
                neighbors_by_type_accumulator[atom_id].append(
                    Counter(atom['neighbors_by_type'])
                )

            # Accumulate Steinhardt parameters
            q4_accumulator[atom_id].append(atom['q4'])
            q6_accumulator[atom_id].append(atom['q6'])

            # Store atom metadata from first snapshot
            if atom_id not in metadata_reference:
                metadata_reference[atom_id] = {
                    'id': atom_id,
                    'type': atom['type'],
                    'x': atom['x'],
                    'y': atom['y'],
                    'z': atom['z'],
                    'radius': atom['radius'],
                }

    # Compute temporal averages for q4 and q6
    q4_avg_over_time = {atom_id: np.mean(values, axis=0) for atom_id, values in q4_accumulator.items()}
    q6_avg_over_time = {atom_id: np.mean(values, axis=0) for atom_id, values in q6_accumulator.items()}

    if consolidated_metadata is None:
        logging.error("No valid snapshot data processed. Temporal averaging cannot proceed.")
        return [], SimpleNamespace()

    # Initialize final average containers
    g_i_r_avg_over_time = {}
    volume_avg_over_time = {}
    CN_avg_over_time = {}
    neighbors_by_type_avg_over_time = {}

    missing_g_i_r_atoms = []
    missing_volume_atoms = []
    missing_CN_atoms = []
    missing_neighbors_by_type_atoms = []

    # Average g_i(r) per atom
    for atom_id, g_list in g_i_r_accumulator.items():
        if g_list:
            g_i_r_avg_over_time[atom_id] = np.mean(np.stack(g_list), axis=0)
        else:
            missing_g_i_r_atoms.append(atom_id)
            logging.warning(f"Atom ID {atom_id} missing g_i(r) across all snapshots.")

    # Average Voronoi volume per atom
    for atom_id, vol_list in volume_accumulator.items():
        if vol_list:
            volume_avg_over_time[atom_id] = np.mean(vol_list)
        else:
            missing_volume_atoms.append(atom_id)
            logging.warning(f"Atom ID {atom_id} missing volume data across all snapshots.")

    # Average neighbor-type counts per atom
    for atom_id, counters_list in neighbors_by_type_accumulator.items():
        if counters_list:
            summed_counts = Counter()
            for counter in counters_list:
                summed_counts.update(counter)
            neighbors_by_type_avg_over_time[atom_id] = {
                neighbor_type: summed_counts[neighbor_type] / len(counters_list)
                for neighbor_type in summed_counts
            }
        else:
            missing_neighbors_by_type_atoms.append(atom_id)
            logging.warning(f"Atom ID {atom_id} missing neighbors_by_type data across snapshots.")

    # Average coordination number per atom
    for atom_id, CN_list in CN_accumulator.items():
        if CN_list:
            CN_avg_over_time[atom_id] = np.mean(CN_list)
        else:
            missing_CN_atoms.append(atom_id)
            logging.warning(f"Atom ID {atom_id} missing CN data across all snapshots.")

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

    # Construct final list of atoms with averaged data
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
