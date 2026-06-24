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
from collections import Counter
from scipy.stats import skew as scipy_skew

import freud
import numpy as np
from freud.box import Box


# Custom modules
import io_module
import voronoi
from rdf import compute_all_local_rdfs, compute_pair_entropy_s2, compute_isb_vectorized


# ------------------------------------------------------------------------------
# Logging configuration
# NOTE: Do NOT use force=True here; pipeline_orchestrator.py handles initial logging setup.
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Suppress verbose freud logs
logging.getLogger("freud").setLevel(logging.WARNING)


# ------------------------------------------------------------------------------
# Core function
# ------------------------------------------------------------------------------
def compute_bond_angle_features(atom_pos, neighbor_positions, neighbor_types,
                                  atom_type, box_lengths):
    """
    atom_pos: position of central atom i (3,)
    neighbor_positions: positions of all Voronoi neighbors (N, 3)
    neighbor_types: atom types of neighbors (N,)
    atom_type: type of central atom (1 or 2)
    box_lengths: [Lx, Ly, Lz] for minimum image convention
    
    Returns dict of bond angle features.
    """
    n_neighbors = len(neighbor_positions)
    nan_dict = {k: np.nan for k in [
        'mean_angle_all', 'std_angle_all', 'skewness_angle_all',
        'mean_angle_liketype', 'mean_angle_unliketype', 'mean_angle_mixedtype',
        'count_liketype_pairs', 'count_unliketype_pairs'
    ]}
    if n_neighbors < 2:
        return nan_dict
    
    bond_vectors = neighbor_positions - atom_pos
    for d in range(3):
        bond_vectors[:, d] -= np.round(bond_vectors[:, d] / box_lengths[d]) * box_lengths[d]
    
    norms = np.linalg.norm(bond_vectors, axis=1)
    valid = norms > 1e-10
    if np.sum(valid) < 2:
        return nan_dict
    
    unit_vectors = bond_vectors[valid] / norms[valid, np.newaxis]
    valid_types = neighbor_types[valid]
    n_valid = len(unit_vectors)
    
    all_angles = []
    liketype_angles = []
    unliketype_angles = []
    mixedtype_angles = []
    
    for j in range(n_valid):
        for k in range(j + 1, n_valid):
            cos_angle = np.clip(np.dot(unit_vectors[j], unit_vectors[k]), -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_angle))
            all_angles.append(angle_deg)
            
            tj, tk = valid_types[j], valid_types[k]
            if tj == tk:
                if tj == atom_type:
                    liketype_angles.append(angle_deg)
                else:
                    unliketype_angles.append(angle_deg)
            else:
                mixedtype_angles.append(angle_deg)
    
    all_angles = np.array(all_angles)
    result = {}
    if len(all_angles) >= 2:
        result['mean_angle_all'] = np.mean(all_angles)
        result['std_angle_all'] = np.std(all_angles)
        result['skewness_angle_all'] = float(scipy_skew(all_angles)) if len(all_angles) >= 3 else np.nan
    else:
        result['mean_angle_all'] = np.nan
        result['std_angle_all'] = np.nan
        result['skewness_angle_all'] = np.nan
    
    result['mean_angle_liketype'] = np.mean(liketype_angles) if liketype_angles else np.nan
    result['mean_angle_unliketype'] = np.mean(unliketype_angles) if unliketype_angles else np.nan
    result['mean_angle_mixedtype'] = np.mean(mixedtype_angles) if mixedtype_angles else np.nan
    result['count_liketype_pairs'] = len(liketype_angles)
    result['count_unliketype_pairs'] = len(unliketype_angles)
    return result


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
                snapshot_file, params['radius_file'],
                atomic_radii_override=params.get('atomic_radii')
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
        except Exception as e:
            logging.exception(f"Failed to wrap positions for {snapshot_file}: {e}")
            return None


        # Compute Voronoi tessellation with optimized parameters
        try:
            # Use larger block size for better performance with many atoms
            block_size = max(5, voronoi.calculate_optimal_block_size(pos, box_size))
            
            # Only compute Voronoi if needed for analysis
            if params.get('COMPUTE_VORONOI', True):
                voronoi_cells = voronoi.compute_weighted_voronoi_cells(
                    pos, limits, block_size, radii
                )
            else:
                voronoi_cells = [{'volume': 0, 'faces': []} for _ in range(len(pos))]
                logging.debug("Skipping Voronoi computation as configured")
                
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
        # Compute Warren-Cowley Chemical Short-Range Order (CSRO)
        # ----------------------------------------------------------------------
        # Get global composition from params
        comp = params.get('composition', {})
        if comp.get('compute_from_snapshot', False):
            # Compute from actual atom counts in this snapshot
            n_type1 = int((types == 1).sum())
            n_type2 = int((types == 2).sum())
            total = n_type1 + n_type2
            x_type1 = n_type1 / total if total > 0 else 0.5
            x_type2 = n_type2 / total if total > 0 else 0.5
        else:
            x_type1 = float(comp.get('type_1_fraction', 0.5))
            x_type2 = float(comp.get('type_2_fraction', 0.5))
        
        csro_unlike_list = np.full(num_atoms, np.nan)
        csro_like_list = np.full(num_atoms, np.nan)
        
        for atom_index in range(num_atoms):
            atom_type = types[atom_index]
            n1 = neighbors_by_type[ids[atom_index]].get(1, 0)
            n2 = neighbors_by_type[ids[atom_index]].get(2, 0)
            cn = n1 + n2
            if cn == 0:
                continue
            
            p1 = n1 / cn
            p2 = n2 / cn
            
            if atom_type == 1:  # Cu atom: unlike = tendency to have Zr neighbors
                csro_unlike = 1.0 - (p2 / x_type2) if x_type2 > 1e-10 else np.nan
                csro_like = 1.0 - (p1 / x_type1) if x_type1 > 1e-10 else np.nan
            else:  # Zr atom: unlike = tendency to have Cu neighbors
                csro_unlike = 1.0 - (p1 / x_type1) if x_type1 > 1e-10 else np.nan
                csro_like = 1.0 - (p2 / x_type2) if x_type2 > 1e-10 else np.nan
            
            csro_unlike_list[atom_index] = csro_unlike
            csro_like_list[atom_index] = csro_like

        # ----------------------------------------------------------------------
        # Compute bond angle distribution features
        # ----------------------------------------------------------------------
        logging.info(f"Computing bond angle features for {snapshot_file}...")
        
        # Pre-allocate angle feature arrays
        angle_keys = ['mean_angle_all', 'std_angle_all', 'skewness_angle_all',
                      'mean_angle_liketype', 'mean_angle_unliketype', 'mean_angle_mixedtype',
                      'count_liketype_pairs', 'count_unliketype_pairs']
        angle_features = {k: np.full(num_atoms, np.nan) for k in angle_keys}
        
        for atom_index in range(num_atoms):
            neigh = processed_vor_neighbors[atom_index]
            if len(neigh) < 2:
                continue
            
            npos = pos_wrapped[neigh]
            ntypes = types[neigh]
            aresult = compute_bond_angle_features(
                pos_wrapped[atom_index], npos, ntypes, types[atom_index], box_size
            )
            for k in angle_keys:
                angle_features[k][atom_index] = aresult[k]
        
        logging.info(f"Bond angle features computed for {snapshot_file}.")

        # ----------------------------------------------------------------------
        # Compute local RDF gᵢ(r) in a vectorized pass.
        # ----------------------------------------------------------------------
        logging.info(f"Computing individual g_i(r) for {snapshot_file}...")

        try:
            local_rdf_results = compute_all_local_rdfs(
                pos_wrapped,
                ids,
                box_size,
                params["R_MAX"],
                bins_for_rdf_calc,
                bin_volumes_for_rdf_calc,
                density,
            )
        except Exception as e:
            logging.exception(f"Vectorized RDF calculation failed for {snapshot_file}: {e}")
            return None

        logging.info(f"Individual g_i(r) for {snapshot_file} computed.")

        # ----------------------------------------------------------------------
        # Compute pair entropy S2 from per-atom g(r)
        # ----------------------------------------------------------------------
        # Bin centers (midpoints of the RDF bins)
        r_centers = (bins_for_rdf_calc[:-1] + bins_for_rdf_calc[1:]) / 2.0
        s2_entropy_values = {}
        for atom_id, gr in local_rdf_results.items():
            if np.any(np.isfinite(gr)):
                s2 = compute_pair_entropy_s2(r_centers, gr, density)
            else:
                s2 = np.nan
            s2_entropy_values[atom_id] = s2

        # Extract S2 per atom in order and compute neighbor-averaged version
        # Lechner-Dellago convention: include central atom + its Voronoi neighbors
        s2_list = np.array([s2_entropy_values[atom_id] for atom_id in ids], dtype=float)
        s2_avg_list = np.copy(s2_list)
        for atom_index in range(num_atoms):
            neigh = processed_vor_neighbors[atom_index]
            if neigh:
                neigh_s2 = np.append(s2_list[atom_index], s2_list[neigh])
                s2_avg_list[atom_index] = np.mean(neigh_s2)

        # ----------------------------------------------------------------------
        # Compute inversion symmetry breaking (ISB)
        # ----------------------------------------------------------------------
        from rdf import compute_isb_vectorized as compute_isb
        isb_values = compute_isb(pos_wrapped, processed_vor_neighbors, box_size)

        # ----------------------------------------------------------------------
        # Compute bond orientational order parameters q₄ and q₆
        # ----------------------------------------------------------------------
        box = Box.from_box(box_size)
        positions = pos_wrapped

        neighbor_query = freud.locality.AABBQuery(box, positions)
        neighbor_list = neighbor_query.query(
            positions, {'r_max': 3.5, 'exclude_ii': True}
        ).toNeighborList()

        # Q4 and W4 (Steinhardt 2nd and 3rd order invariants, l=4)
        steinhardt_4 = freud.order.Steinhardt(l=4, average=False)
        steinhardt_4.compute((box, positions), neighbors=neighbor_list)
        q4 = steinhardt_4.particle_order

        wl4 = freud.order.Steinhardt(l=4, wl=True, average=False)
        wl4.compute((box, positions), neighbors=neighbor_list)
        w4 = wl4.particle_order

        # Q6 and W6 (Steinhardt 2nd and 3rd order invariants, l=6)
        steinhardt_6 = freud.order.Steinhardt(l=6, average=False)
        steinhardt_6.compute((box, positions), neighbors=neighbor_list)
        q6 = steinhardt_6.particle_order

        wl6 = freud.order.Steinhardt(l=6, wl=True, average=False)
        wl6.compute((box, positions), neighbors=neighbor_list)
        w6 = wl6.particle_order

        # Q4_avg and Q6_avg (Lechner-Dellago averaged, l=4 and l=6)
        steinhardt_4_avg = freud.order.Steinhardt(l=4, average=True)
        steinhardt_4_avg.compute((box, positions), neighbors=neighbor_list)
        q4_avg = steinhardt_4_avg.particle_order

        steinhardt_6_avg = freud.order.Steinhardt(l=6, average=True)
        steinhardt_6_avg.compute((box, positions), neighbors=neighbor_list)
        q6_avg = steinhardt_6_avg.particle_order

        # ----------------------------------------------------------------------
        # Assemble atom-level data
        # Voronoi index ⟨n3, n4, n5, n6⟩ is now embedded in voronoi_cells
        # from the freud polytopes (computed inside compute_weighted_voronoi_cells)
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
                    'w4': w4[atom_index],
                    'w6': w6[atom_index],
                    'q4_avg': q4_avg[atom_index],
                    'q6_avg': q6_avg[atom_index],
                    'n3_voronoi': voronoi_cells[atom_index].get('voronoi_index', {}).get('n3', 0),
                    'n4_voronoi': voronoi_cells[atom_index].get('voronoi_index', {}).get('n4', 0),
                    'n5_voronoi': voronoi_cells[atom_index].get('voronoi_index', {}).get('n5', 0),
                    'n6_voronoi': voronoi_cells[atom_index].get('voronoi_index', {}).get('n6', 0),
                    'asphericity_voronoi': voronoi_cells[atom_index].get('voronoi_index', {}).get('asphericity', np.nan),
                    's2_entropy': s2_list[atom_index],
                    's2_entropy_avg': s2_avg_list[atom_index],
                    'isb': isb_values[atom_index],
                    'csro_unlike': csro_unlike_list[atom_index],
                    'csro_like': csro_like_list[atom_index],
                    'mean_angle_all': angle_features['mean_angle_all'][atom_index],
                    'std_angle_all': angle_features['std_angle_all'][atom_index],
                    'skewness_angle_all': angle_features['skewness_angle_all'][atom_index],
                    'mean_angle_liketype': angle_features['mean_angle_liketype'][atom_index],
                    'mean_angle_unliketype': angle_features['mean_angle_unliketype'][atom_index],
                    'mean_angle_mixedtype': angle_features['mean_angle_mixedtype'][atom_index],
                    'count_liketype_pairs': angle_features['count_liketype_pairs'][atom_index],
                    'count_unliketype_pairs': angle_features['count_unliketype_pairs'][atom_index],
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
