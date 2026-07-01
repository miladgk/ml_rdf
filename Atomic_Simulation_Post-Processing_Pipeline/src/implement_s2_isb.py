"""
Comprehensive script to add S2 entropy and ISB features,
rebuild all data, train binary model, and run diagnostics.
"""
import numpy as np
from scipy.integrate import trapezoid

# ============================================================
# PART A: Functions to be added to rdf.py and snapshot_processor.py
# ============================================================

def compute_pair_entropy_s2(r_array, gr_array, rho):
    """Compute pair entropy S2 from per-atom g(r).
    r_array: radial distances (bin centers), shape (num_bins,)
    gr_array: per-atom g(r), shape (num_bins,)
    rho: number density (N/V)
    Returns: S2 (always negative)
    """
    gr_safe = np.where(gr_array <= 0, 1e-10, gr_array)
    integrand = (gr_safe * np.log(gr_safe) - gr_safe + 1) * r_array**2
    s2 = -2 * np.pi * rho * trapezoid(integrand, r_array)
    return s2


def compute_isb(atom_position, neighbor_positions, box_size):
    """Compute inversion symmetry breaking parameter.
    atom_position: (3,) array
    neighbor_positions: (N, 3) array of neighbor positions
    box_size: (3,) array of box dimensions for minimum image convention
    
    Returns: ISB scalar (0 = centrosymmetric, ~1 = asymmetric)
    """
    if len(neighbor_positions) == 0:
        return np.nan
    
    # Minimum image convention
    delta = neighbor_positions - atom_position
    delta = delta - np.round(delta / box_size) * box_size
    
    norms = np.linalg.norm(delta, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1e-10, norms)
    unit_vectors = delta / norms
    
    vector_sum = np.sum(unit_vectors, axis=0)
    isb = np.linalg.norm(vector_sum) / len(neighbor_positions)
    return isb


def compute_isb_vectorized(atom_positions, neighbor_idx_list, box_size):
    """Vectorized ISB computation for all atoms.
    atom_positions: (N, 3)
    neighbor_idx_list: list of lists, neighbor_idx_list[i] = [j1, j2, ...]
    box_size: (3,)
    Returns: (N,) array of ISB values
    """
    n_atoms = len(atom_positions)
    isb_values = np.full(n_atoms, np.nan)
    
    for i in range(n_atoms):
        neighbors = neighbor_idx_list[i]
        if len(neighbors) == 0:
            continue
        
        neigh_pos = atom_positions[neighbors]
        delta = neigh_pos - atom_positions[i]
        delta = delta - np.round(delta / box_size) * box_size
        
        norms = np.linalg.norm(delta, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1e-10, norms)
        unit_vectors = delta / norms
        
        isb_values[i] = np.linalg.norm(np.sum(unit_vectors, axis=0)) / len(neighbors)
    
    return isb_values