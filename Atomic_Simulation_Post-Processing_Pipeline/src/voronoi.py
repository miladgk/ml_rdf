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
- Voronoi Index: Computes per-atom ⟨n3, n4, n5, n6⟩ face counts via pyvoro.

Dependencies:
- NumPy: Efficient numerical computations for volume, density, and scaling calculations.
- PyVoro: Python interface to Voro++ for high-performance Voronoi tessellation.

"""

import logging
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
    """
    total_volume = np.prod(box_dimensions)
    particle_density = len(positions) / total_volume
    target_block_volume = target_particles_per_block / particle_density
    optimal_block_size = np.cbrt(target_block_volume)
    min_block_size = min(box_dimensions) * min_block_factor
    max_block_size = min(box_dimensions) * max_block_factor
    return np.clip(optimal_block_size, min_block_size, max_block_size)


def compute_weighted_voronoi_cells(points, box_limits, block_size, particle_radii):
    """
    Compute a weighted Voronoi tessellation for 3D points.

    Uses freud's Voronoi implementation when available (fastest), falls back to
    PyVoro, and finally to a SciPy KDTree approximation if neither is available.

    freud's Voronoi is typically 2-3x faster than PyVoro for large systems since
    it uses optimized C++ with periodic boundary support natively.

    NOTE: freud does NOT return face vertex/area data, only volumes
    and neighbor lists. For Voronoi index (n3,n4,n5,n6), a separate
    pyvoro call is needed (see compute_voronoi_index).
    """
    points = np.asarray(points, dtype=float)
    box_limits = np.asarray(box_limits, dtype=float)
    box_size = box_limits[:, 1] - box_limits[:, 0]
    lower_bounds = box_limits[:, 0]

    # ----------------------------------------------------------------
    # Try freud first (fastest for periodic systems).
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

        logging.info("Voronoi backend: freud")
        return voronoi_cells

    except Exception:
        pass

    # ----------------------------------------------------------------
    # Fallback to PyVoro
    # ----------------------------------------------------------------
    if pyvoro is not None:
        try:
            result = pyvoro.compute_voronoi(
                points,
                box_limits,
                block_size,
                periodic=[True, True, True],
                radii=particle_radii
            )
            logging.info("Voronoi backend: pyvoro")
            return result
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


def compute_voronoi_index(points, box_limits, block_size, particle_radii, area_cutoff_fraction=0.01):
    """
    Compute Voronoi index ⟨n3, n4, n5, n6⟩ per atom using pyvoro (Voro++).
    
    This is a separate, dedicated call for face geometry only, independent of
    which backend is used for volume/CN computation.

    Parameters
    ----------
    points : array_like
        Array of particle positions with shape (N, 3).
    box_limits : array_like
        Bounds of the simulation box [[x_min, x_max], [y_min, y_max], [z_min, z_max]].
    block_size : float
        Size of each computational block.
    particle_radii : array_like
        Array of particle radii used for weighted tessellation.
    area_cutoff_fraction : float, optional
        Fraction of total cell surface area below which a face is discarded
        as a sliver (default 0.01 = 1%).

    Returns
    -------
    list of dict
        Each dict: {'n3': int, 'n4': int, 'n5': int, 'n6': int}
        for each atom.
    
    Raises
    ------
    RuntimeError
        If pyvoro is not installed.
    """
    if pyvoro is None:
        raise RuntimeError(
            "pyvoro is required for Voronoi index computation. "
            "Install with: pip install pyvoro"
        )

    points = np.asarray(points, dtype=float)
    # Convert box_limits to list format expected by pyvoro
    if hasattr(box_limits, 'tolist'):
        box_limits_list = box_limits.tolist()
    else:
        box_limits_list = [[float(bl[0]), float(bl[1])] for bl in box_limits]
    radii_list = particle_radii.tolist() if hasattr(particle_radii, 'tolist') else list(particle_radii)

    logging.info("Computing Voronoi index (face geometry) via pyvoro...")

    # Run pyvoro — this is a separate, parallel computation
    cells = pyvoro.compute_voronoi(
        points,
        box_limits_list,
        float(block_size),
        periodic=[True, True, True],
        radii=radii_list
    )

    results = []
    for cell in cells:
        # Each cell has: 'original', 'volume', 'vertices', 'adjacency', 'faces'
        verts = np.array(cell['vertices'])  # shape (N_verts, 3)
        faces = cell['faces']  # list of dict with 'adjacent_cell' and 'vertices'

        # Compute area for each face from vertex positions
        face_areas = []
        valid_face_vertex_counts = []
        for face in faces:
            face_vert_indices = face['vertices']
            if len(face_vert_indices) < 3:
                continue  # degenerate face
            
            # Get vertex positions for this face
            fverts = verts[face_vert_indices]  # shape (n_verts, 3)
            
            # Compute polygon area using triangle fan from first vertex
            v0 = fverts[0]
            area = 0.0
            for i in range(1, len(fverts) - 1):
                v1 = fverts[i] - v0
                v2 = fverts[i + 1] - v0
                cross = np.cross(v1, v2)
                area += 0.5 * np.linalg.norm(cross)
            
            face_areas.append(area)
            valid_face_vertex_counts.append(len(face_vert_indices))

        # Apply area cutoff: discard faces below cutoff fraction of total area
        total_area = sum(face_areas) if face_areas else 0.0
        if total_area > 0:
            cutoff = area_cutoff_fraction * total_area
            counts = {}
            for n_verts, area in zip(valid_face_vertex_counts, face_areas):
                if area >= cutoff and 3 <= n_verts <= 6:
                    counts[n_verts] = counts.get(n_verts, 0) + 1
        else:
            counts = {}

        results.append({
            'n3': counts.get(3, 0),
            'n4': counts.get(4, 0),
            'n5': counts.get(5, 0),
            'n6': counts.get(6, 0),
        })

    logging.info(f"Voronoi index computed for {len(results)} atoms.")
    return results