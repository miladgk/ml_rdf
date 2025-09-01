"""
snapshot_processor.py
---------------------

Processes individual molecular dynamics snapshots to extract structural,
topological, and statistical descriptors of particle systems. Designed for
integration into a larger analysis pipeline, this module reads simulation
data, performs spatial analyses, and returns per-snapshot atom-level metrics.

Main responsibilities:
- Parse LAMMPS dump files to extract atom positions, types, and radii.
- Perform Voronoi tessellations to compute neighbor relationships, neighbor
  counts, and cell volumes.
- Compute local radial distribution functions gᵢ(r) for each atom using
  parallelized calculations.
- Calculate bond orientational order parameters (Steinhardt Q4 and Q6)
  using the `freud` library.
- Build periodic KD-trees for efficient neighbor queries.

Key libraries and concepts:
- **freud**: Voronoi tessellations, bond orientational order, neighbor queries.
- **SciPy KDTree**: Periodic boundary-aware neighbor searches.
- **NumPy**: Numerical operations.
- **Multiprocessing**: Parallel RDF calculations.
- **Voronoi decomposition, RDFs, Steinhardt order, periodic boundaries.**

This module is not intended to be run directly. It should be imported into
higher-level orchestration scripts such as `pipeline_orchestrator.py`.
"""

import logging
import multiprocessing
from collections import Counter

import freud
import numpy as np
from freud.box import Box
from scipy.spatial import KDTree

# Custom modules
import io_module
import voronoi
from rdf import compute_local_rdf


# ------------------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True
)

# Suppress verbose freud logs
logging.getLogger("freud").setLevel(logging.WARNING)


# ------------------------------------------------------------------------------
# Core function
# ------------------------------------------------------------------------------
def process_single_snapshot(snapshot_file, params, bins_for_rdf_calc, bin_volumes_for_rdf_calc):
    """
    Process a single LAMMPS snapshot to compute per-atom descriptors.

    Steps:
        1. Read atomic positions, types, and radii from LAMMPS dump.
        2. Compute Voronoi tessellation and extract neighbor information.
        3. Construct periodic KD-tree for neighbor searching.
        4. Calculate local radial distribution functions gᵢ(r).
        5. Compute bond orientational order parameters q₄ and q₆ using freud.
        6. Aggregate per-atom data and return along with snapshot metadata.

    Parameters
    ----------
    snapshot_file : str
        Path to the LAMMPS dump file.
    params : dict
        Dictionary of parameters (must include "radius_file" and "R_MAX").
    bins_for_rdf_calc : numpy.ndarray
        RDF bin edges used for gᵢ(r) calculation.
    bin_volumes_for_rdf_calc : numpy.ndarray
        Volumes corresponding to RDF bins.

    Returns
    -------
    dict or None
        A dictionary containing:
        - 'atom_data_snapshot': list of per-atom dictionaries with metrics.
        - 'snapshot_metadata': dictionary with snapshot-level metadata.
        Returns None if the snapshot fails to process.
    """
    try:
        logging.info(f"Reading LAMMPS dump file: {snapshot_file}")
        try:
            df_atoms, limits = io_module.read_lammps_dump(
                snapshot_file, params['radius_file']
            )
        except Exception as e:
            logging.error(f"Failed to read LAMMPS dump for {snapshot_file}: {e}")
            return None

        # Extract atomic data
        pos = df_atoms[['x', 'y', 'z']].values
        ids = df_atoms['id'].values
        types = df_atoms['type'].values
        radii = df_atoms['radius'].values
        num_atoms = len(ids)

        # Compute simulation box dimensions and density
        box_size = np.array([
            limits[0][1] - limits[0][0],
            limits[1][1] - limits[1][0],
            limits[2][1] - limits[2][0]
        ])
        total_volume = np.prod(box_size)
        density = num_atoms / total_volume

        # Wrap positions into periodic box
        lower_bounds = np.array([limits[0][0], limits[1][0], limits[2][0]])
        pos_relative = pos - lower_bounds
        try:
            pos_wrapped = np.mod(pos_relative, box_size)
            kd_tree = KDTree(pos_wrapped, boxsize=box_size)
        except Exception as e:
            logging.exception(f"Failed to build KDTree for {snapshot_file}: {e}")
            return None

        # Compute Voronoi tessellation
        block_size = voronoi.calculate_optimal_block_size(pos, box_size)
        try:
            voronoi_cells = voronoi.compute_weighted_voronoi_cells(
                pos, limits, block_size, radii
            )
        except Exception as e:
            logging.error(f"Failed to compute Voronoi cells for {snapshot_file}: {e}")
            return None

        # Map original atom IDs to 0-based indices
        id_to_idx = {atom_id: idx for idx, atom_id in enumerate(ids)}

        # Convert Voronoi neighbor IDs to 0-based indices
        processed_vor_neighbors = []
        for cell in voronoi_cells:
            neighbors_for_atom = [
                id_to_idx[face['adjacent_cell']]
                for face in cell['faces']
                if face['adjacent_cell'] > 0 and face['adjacent_cell'] in id_to_idx
            ]
            processed_vor_neighbors.append(neighbors_for_atom)

        vor_volumes = [cell.get('volume', None) for cell in voronoi_cells]
        num_neighbors = [len(neigh) for neigh in processed_vor_neighbors]

        # Count neighbors by atom type
        neighbors_by_type = {
            atom_id: Counter(types[n_idx] for n_idx in processed_vor_neighbors[a_idx])
            for atom_id, a_idx in id_to_idx.items()
        }

        # ----------------------------------------------------------------------
        # Compute local RDF gᵢ(r) in parallel
        # ----------------------------------------------------------------------
        logging.info(f"Computing individual g_i(r) for {snapshot_file}...")

        starmap_args = [
            (
                i,
                pos_wrapped,
                ids,
                kd_tree,
                params["R_MAX"],
                bins_for_rdf_calc,
                bin_volumes_for_rdf_calc,
                density,
                lower_bounds
            )
            for i in range(num_atoms)
        ]

        try:
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                rdf_results = pool.starmap(compute_local_rdf, starmap_args)
        except Exception as e:
            logging.exception(f"RDF starmap failed for {snapshot_file}: {e}")
            return None

        local_rdf_results = {}
        for atom_id, g_i_r_array in rdf_results:
            if g_i_r_array is not None:
                local_rdf_results[atom_id] = g_i_r_array
            else:
                logging.warning(
                    f"RDF calculation for atom ID {atom_id} "
                    f"returned None in {snapshot_file}."
                )

        logging.info(f"Individual g_i(r) for {snapshot_file} computed.")

        # ----------------------------------------------------------------------
        # Compute bond orientational order parameters q₄ and q₆
        # ----------------------------------------------------------------------
        box = Box.from_box(box_size)
        positions = pos_wrapped

        neighbor_query = freud.locality.AABBQuery(box, positions)
        neighbor_list = neighbor_query.query(
            positions, {'r_max': 3.5, 'exclude_ii': True}
        ).toNeighborList()

        steinhardt_4 = freud.order.Steinhardt(l=4, average=False)
        steinhardt_6 = freud.order.Steinhardt(l=6, average=False)

        steinhardt_4.compute((box, positions), neighbors=neighbor_list)
        steinhardt_6.compute((box, positions), neighbors=neighbor_list)

        q4 = steinhardt_4.particle_order
        q6 = steinhardt_6.particle_order

        # ----------------------------------------------------------------------
        # Assemble atom-level data
        # ----------------------------------------------------------------------
        atom_data = []
        for atom_index in range(num_atoms):
            atom_id = ids[atom_index]
            if atom_id in local_rdf_results:
                atom_data.append({
                    'id': atom_id,
                    'type': types[atom_index],
                    'x': pos[atom_index][0],
                    'y': pos[atom_index][1],
                    'z': pos[atom_index][2],
                    'radius': radii[atom_index],
                    # Convert back to original IDs for output
                    'vor_neighbors': [ids[i] for i in processed_vor_neighbors[atom_index]],
                    'num_neighbors': num_neighbors[atom_index],
                    'neighbors_by_type': dict(neighbors_by_type[atom_id]),
                    'voronoi_volume': vor_volumes[atom_index],
                    'g_i_r_snapshot': local_rdf_results[atom_id].tolist(),
                    'q4': q4[atom_index],
                    'q6': q6[atom_index],
                })
            else:
                logging.warning(
                    f"Atom ID {atom_id} in snapshot {snapshot_file} "
                    f"has no g_i(r) data. Excluded from results."
                )

        # ----------------------------------------------------------------------
        # Snapshot metadata
        # ----------------------------------------------------------------------
        snapshot_metadata = {
            'box_size': box_size.tolist(),
            'total_volume': total_volume,
            'density': density,
            'lower_bounds': lower_bounds.tolist(),
            'bins': bins_for_rdf_calc.tolist(),
            'bin_volumes': bin_volumes_for_rdf_calc.tolist()
        }

        return {
            'atom_data_snapshot': atom_data,
            'snapshot_metadata': snapshot_metadata,
        }

    except Exception as e:
        logging.exception(
            f"[CRITICAL] Exception inside process_single_snapshot "
            f"for {snapshot_file}: {e}"
        )
        return None


# ------------------------------------------------------------------------------
# Entry point (for clarity only, not intended for execution)
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    print("This module is intended to be imported.")
    print("Run pipeline_orchestrator.py to execute the full workflow.")
