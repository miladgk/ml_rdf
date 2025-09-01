"""
rdf.py
------

This module provides utilities for computing the Radial Distribution Function (RDF)
at a per-atom level, which is widely used in molecular dynamics and materials science
to characterize local atomic structures.

The functions here are designed for integration into larger analysis pipelines
where structural descriptors (like RDFs) are combined with other features
(e.g., Voronoi volume, potential energy, bond order parameters) for studying
metallic glasses and related systems.

Key functionalities:
- **create_adaptive_bins**: Generate evenly spaced RDF bin edges.
- **validate_rdf**: Placeholder for RDF validation logic.
- **compute_local_rdf**: Compute the local RDF of an atom using a KDTree for neighbor search.

Technical details:
- Uses **NumPy** for vectorized numerical operations.
- Uses **SciPy's KDTree** for efficient neighbor search in 3D.
- Normalizes RDF by bin volumes and system density.
- Includes logging for error handling and debugging.

This module is intended to be imported and used by snapshot-level
processing modules (e.g., `snapshot_processor.py`) in simulation analysis pipelines.
"""

import numpy as np
import logging
from scipy.spatial import KDTree  # Required for neighbor searches

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)


def create_adaptive_bins(r_min, r_max, num_bins=100):
    """
    Create evenly spaced bin edges for RDF calculation.

    Args:
        r_min (float): Minimum radial distance.
        r_max (float): Maximum radial distance (cutoff).
        num_bins (int, optional): Number of bins. Defaults to 100.

    Returns:
        np.ndarray: Array of bin edges of shape (num_bins + 1,).
    """
    return np.linspace(r_min, r_max, num_bins + 1)


def validate_rdf(r_values, g_r, collision_diameter=1.0, long_range_threshold=0.1):
    """
    Placeholder for RDF validation.

    Intended to check short-range and long-range behavior of g(r).
    Currently not implemented.

    Args:
        r_values (np.ndarray): Array of radial distances.
        g_r (np.ndarray): RDF values corresponding to r_values.
        collision_diameter (float, optional): Expected hard-sphere diameter. Defaults to 1.0.
        long_range_threshold (float, optional): Threshold for long-range behavior. Defaults to 0.1.
    """
    pass


def compute_local_rdf(
    atom_index,
    all_positions,
    all_atom_ids,
    kd_tree,
    cutoff,
    bins,
    bin_volumes,
    density,
    lower_bounds,
):
    """
    Compute the local RDF for a single atom.

    Args:
        atom_index (int): Index of the atom for which RDF is computed.
        all_positions (np.ndarray): Shape (N, 3). All atom positions in the snapshot.
        all_atom_ids (np.ndarray): Shape (N,). IDs of atoms in the snapshot.
        kd_tree (scipy.spatial.KDTree): KDTree built from shifted positions.
        cutoff (float): Maximum distance for RDF calculation.
        bins (np.ndarray): Bin edges of shape (num_bins + 1,).
        bin_volumes (np.ndarray): Volumes of spherical shells for each bin (shape: num_bins).
        density (float): System number density.
        lower_bounds (np.ndarray): Simulation box lower bounds (shape: (3,)).

    Returns:
        tuple:
            - atom_id (int): ID of the atom.
            - g_r (np.ndarray): Local RDF values (shape: num_bins).
              Returns a zero array if an error occurs.
    """
    atom_id = all_atom_ids[atom_index]
    num_bins = len(bins) - 1  # Number of intervals

    try:
        # Query point relative to KDTree reference frame
        query_point = all_positions[atom_index] - lower_bounds
        neighbor_indices = kd_tree.query_ball_point(query_point, cutoff)

        # Compute distances to neighbors
        center = all_positions[atom_index]
        distances = np.linalg.norm(all_positions[neighbor_indices] - center, axis=1)

        # Exclude self-distance (possible floating point artifact)
        distances = distances[distances > 1e-8]

        # Histogram neighbor distances
        hist, _ = np.histogram(distances, bins=bins)

        # Sanity checks
        if len(hist) != num_bins:
            logging.error(
                f"Atom {atom_id}: Histogram length {len(hist)} != expected {num_bins}"
            )
            return atom_id, np.zeros(num_bins)

        if len(bin_volumes) != num_bins:
            logging.error(
                f"Atom {atom_id}: Bin volumes length {len(bin_volumes)} != expected {num_bins}"
            )
            return atom_id, np.zeros(num_bins)

        # Normalize histogram to compute g(r)
        denominator = density * bin_volumes
        g_r = np.zeros_like(denominator, dtype=float)

        # Avoid division by zero
        valid_bins = denominator > 1e-12
        g_r[valid_bins] = hist[valid_bins].astype(float) / denominator[valid_bins]

        # Ensure correct shape even if no neighbors were found
        if g_r.shape[0] == 0 and num_bins > 0:
            g_r = np.zeros(num_bins)

        return atom_id, g_r

    except Exception as error:
        logging.error(
            f"Error computing RDF for atom {atom_id}: {error}", exc_info=True
        )
        # Fall back to zero array in case of unexpected error
        return atom_id, np.zeros(num_bins if "num_bins" in locals() else 100)
