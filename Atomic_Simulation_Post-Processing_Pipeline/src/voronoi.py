"""
voronoi.py
----------

Voronoi-based Spatial Partitioning Utilities for 3D Particle Simulations

This module provides utility functions for computing weighted Voronoi tessellations
in 3D periodic domains. It focuses on computational efficiency through adaptive
block sizing based on particle density. This is useful for scientific computing
workflows involving spatial decomposition, particle simulations, and geometric analysis.

Key functionalities:
- Optimal Block Size Calculation: Dynamically determines a suitable block size 
  for Voronoi computation based on particle number, simulation box dimensions, 
  and desired particle density per block.
- Weighted Voronoi Tessellation: Generates Voronoi cells in 3D space with periodic 
  boundary conditions, supporting radius-based weighting for accurate partitioning.

Dependencies:
- NumPy: Efficient numerical computations for volume, density, and scaling calculations.
- PyVoro: Python interface to Voro++ for high-performance Voronoi tessellation.

Use case:
Integrate this module into simulation pipelines to improve computational performance
and accuracy in geometry-driven data analysis, particularly for systems requiring
periodic boundary conditions.

"""

import numpy as np

try:
    import pyvoro
except Exception:  # pragma: no cover - depends on optional geometry backend
    pyvoro = None

try:
    from scipy.spatial import KDTree
except Exception:  # pragma: no cover - SciPy is required for the fallback path
    KDTree = None


def calculate_optimal_block_size(positions, box_dimensions, target_particles_per_block=50, 
                                 min_block_factor=1/20, max_block_factor=1/5):
    """
    Determine an optimal block size for Voronoi tessellation based on particle density.

    Parameters
    ----------
    positions : array_like
        Array of particle positions with shape (N, 3), where N is the number of particles.
    box_dimensions : array_like
        Lengths of the simulation box along each axis [Lx, Ly, Lz].
    target_particles_per_block : int, optional
        Desired number of particles per Voronoi block (default is 50).
    min_block_factor : float, optional
        Minimum fraction of the smallest box dimension allowed for a block size (default is 1/20).
    max_block_factor : float, optional
        Maximum fraction of the smallest box dimension allowed for a block size (default is 1/5).

    Returns
    -------
    float
        Optimal block size clipped within the specified min and max limits.
    """
    # Compute the total volume of the simulation box
    total_volume = np.prod(box_dimensions)

    # Estimate particle density
    particle_density = len(positions) / total_volume

    # Compute target block volume based on desired particles per block
    target_block_volume = target_particles_per_block / particle_density

    # Convert volume to cubic block size
    optimal_block_size = np.cbrt(target_block_volume)

    # Enforce minimum and maximum block size limits
    min_block_size = min(box_dimensions) * min_block_factor
    max_block_size = min(box_dimensions) * max_block_factor

    # Return the block size constrained within limits
    return np.clip(optimal_block_size, min_block_size, max_block_size)


def compute_weighted_voronoi_cells(points, box_limits, block_size, particle_radii):
    """
    Compute a weighted Voronoi tessellation for 3D points.

    Uses freud's Voronoi implementation when available (fastest), falls back to
    PyVoro, and finally to a SciPy KDTree approximation if neither is available.

    freud's Voronoi is typically 2-3x faster than PyVoro for large systems since
    it uses optimized C++ with periodic boundary support natively.

    Parameters
    ----------
    points : array_like
        Array of particle positions with shape (N, 3).
    box_limits : array_like
        Bounds of the simulation box [[x_min, x_max], [y_min, y_max], [z_min, z_max]].
    block_size : float
        Size of each computational block (used by PyVoro fallback, ignored by freud).
    particle_radii : array_like
        Array of particle radii used for weighted tessellation.

    Returns
    -------
    list of dict
        Voronoi cells, each represented as a dictionary with geometric information.
    """
    points = np.asarray(points, dtype=float)
    box_limits = np.asarray(box_limits, dtype=float)
    box_size = box_limits[:, 1] - box_limits[:, 0]
    lower_bounds = box_limits[:, 0]

    # ----------------------------------------------------------------
    # Try freud first (fastest for periodic systems).
    # freud's Voronoi.compute() returns volumes and a neighbor list
    # that directly gives geometric Voronoi neighbors.
    # ----------------------------------------------------------------
    try:
        import freud
        from freud.box import Box
        from collections import defaultdict

        freud_box = Box(Lx=box_size[0], Ly=box_size[1], Lz=box_size[2])
        wrapped_points = np.mod(points - lower_bounds, box_size)

        voro = freud.locality.Voronoi()
        voro.compute((freud_box, wrapped_points))

        if not hasattr(voro, 'volumes') or len(voro.volumes) != len(points):
            raise RuntimeError("freud Voronoi returned unexpected output")

        # voro.nlist is a NeighborList, iterable as (point_idx, neighbor_idx) pairs
        nlist = voro.nlist
        neighbor_groups = defaultdict(list)
        for pair in nlist:
            i = int(pair[0])
            j = int(pair[1])
            neighbor_groups[i].append(j)

        voronoi_cells = []
        for i in range(len(points)):
            neighbors = neighbor_groups.get(i, [])
            cell_volume = float(voro.volumes[i])
            faces = [{'adjacent_cell': int(n)} for n in neighbors]
            voronoi_cells.append({
                'volume': cell_volume,
                'faces': faces,
            })

        return voronoi_cells

    except Exception:
        pass

    # ----------------------------------------------------------------
    # Fallback to PyVoro
    # ----------------------------------------------------------------
    if pyvoro is not None:
        try:
            return pyvoro.compute_voronoi(
                points,
                box_limits,
                block_size,
                periodic=[True, True, True],
                radii=particle_radii
            )
        except Exception as exc:
            raise RuntimeError(f"Error during PyVoro tessellation: {exc}") from exc

    # ----------------------------------------------------------------
    # Final fallback: SciPy KDTree approximation
    # ----------------------------------------------------------------
    if KDTree is None:
        raise RuntimeError(
            "No Voronoi backend available. Install freud or pyvoro, "
            "or disable the Voronoi analysis path."
        )

    wrapped_points = np.mod(points - lower_bounds, box_size)
    tree = KDTree(wrapped_points, boxsize=box_size)
    search_radius = max(block_size, 1.0)
    cell_volume = float(np.prod(box_size) / max(len(points), 1))

    voronoi_cells = []
    for i in range(len(points)):
        neighbors = tree.query_ball_point(wrapped_points[i], search_radius)
        neighbor_ids = [int(j) for j in neighbors if j != i]
        voronoi_cells.append({
            'volume': cell_volume,
            'faces': [{'adjacent_cell': neighbor_id} for neighbor_id in neighbor_ids]
        })

    return voronoi_cells
