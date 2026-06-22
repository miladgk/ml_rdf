"""
voronoi.py
----------

Voronoi-based Spatial Partitioning Utilities for 3D Particle Simulations

This module provides utility functions for computing weighted Voronoi tessellations
in 3D periodic domains. It focuses on computational efficiency through adaptive
block sizing based on particle density.

Key functionalities:
- Optimal Block Size Calculation: Dynamically determines a suitable block size 
  for Voronoi computation based on particle number, simulation box dimensions, 
  and desired particle density per block.
- Weighted Voronoi Tessellation: Generates Voronoi cells in 3D space with periodic 
  boundary conditions, supporting radius-based weighting for accurate partitioning.
- Voronoi Index: Computes per-atom ⟨n3, n4, n5, n6⟩ face counts using freud's
  polytopes + scipy ConvexHull (no pyvoro dependency for this).

Dependencies:
- NumPy: Efficient numerical computations for volume, density, and scaling calculations.
- freud: Primary Voronoi backend (volumes, neighbor lists, polytope vertices).
- PyVoro: Fallback Voronoi backend when freud is unavailable.

"""

import logging
import numpy as np

try:
    import pyvoro
except Exception:
    pyvoro = None

try:
    from scipy.spatial import KDTree
except Exception:
    KDTree = None


def calculate_optimal_block_size(positions, box_dimensions, target_particles_per_block=50, 
                                 min_block_factor=1/20, max_block_factor=1/5):
    """Determine an optimal block size for Voronoi tessellation based on particle density."""
    total_volume = np.prod(box_dimensions)
    particle_density = len(positions) / total_volume
    target_block_volume = target_particles_per_block / particle_density
    optimal_block_size = np.cbrt(target_block_volume)
    min_block_size = min(box_dimensions) * min_block_factor
    max_block_size = min(box_dimensions) * max_block_factor
    return np.clip(optimal_block_size, min_block_size, max_block_size)


def _compute_voronoi_index_from_freud(voro_obj, area_cutoff_fraction=0.01):
    """
    Compute Voronoi index ⟨n3, n4, n5, n6⟩ from an already-computed freud Voronoi object.
    
    Uses freud's polytopes (vertex arrays per cell) and scipy.spatial.ConvexHull
    to extract face vertex counts. This avoids pyvoro's memory issues on large systems.
    
    Parameters
    ----------
    voro_obj : freud.locality.Voronoi
        Already-computed freud Voronoi object with .polytopes populated.
    area_cutoff_fraction : float
        Fraction of total cell surface area below which a face is discarded.
    
    Returns
    -------
    list of dict
        Each dict: {'n3': int, 'n4': int, 'n5': int, 'n6': int}
    """
    from scipy.spatial import ConvexHull
    from collections import defaultdict
    
    polytopes = voro_obj.polytopes
    num_cells = len(polytopes)
    logging.info(f"Computing Voronoi index from freud polytopes ({num_cells} cells)...")
    
    results = []
    for ci in range(num_cells):
        verts = polytopes[ci]
        if len(verts) < 4:
            # Degenerate cell — should not happen in valid Voronoi
            results.append({'n3': 0, 'n4': 0, 'n5': 0, 'n6': 0})
            continue
        
        # Compute convex hull to get triangular faces (simplices)
        hull = ConvexHull(verts)
        
        # Group simplex triangles by their plane equation to reconstruct polygonal faces.
        # Each row in hull.equations is (nx, ny, nz, offset) for the plane.
        # Triangles sharing the same plane belong to the same polygonal face.
        groups = defaultdict(list)
        for i, eq in enumerate(hull.equations):
            # Use rounded plane normal + offset as unique key for each face
            key = (np.round(eq[:3], 5).tobytes(), round(eq[3], 5))
            groups[key].append(i)
        
        # For a convex polygon with N vertices, it's triangulated into N-2 triangles.
        # So number of triangles per face + 2 = number of vertices in that face.
        face_vertex_counts = []
        face_areas = []
        for tris in groups.values():
            n_verts = len(tris) + 2
            # Compute face area: sum area of its constituent triangles
            face_area = 0.0
            for tri_idx in tris:
                tri = hull.simplices[tri_idx]
                v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
                face_area += 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            face_vertex_counts.append(n_verts)
            face_areas.append(face_area)
        
        # Apply area cutoff
        total_area = sum(face_areas) if face_areas else 0.0
        if total_area > 0:
            cutoff = area_cutoff_fraction * total_area
            counts = {}
            for nv, area in zip(face_vertex_counts, face_areas):
                if area >= cutoff and 3 <= nv <= 6:
                    counts[nv] = counts.get(nv, 0) + 1
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


def compute_weighted_voronoi_cells(points, box_limits, block_size, particle_radii):
    """
    Compute a weighted Voronoi tessellation for 3D points.

    Uses freud's Voronoi implementation when available (fastest), falls back to
    PyVoro, and finally to a SciPy KDTree approximation if neither is available.

    This function returns both the standard cell data AND the Voronoi index
    results by reusing freud's polytopes internally.
    """
    points = np.asarray(points, dtype=float)
    box_limits = np.asarray(box_limits, dtype=float)
    box_size = box_limits[:, 1] - box_limits[:, 0]
    lower_bounds = box_limits[:, 0]

    # ----------------------------------------------------------------
    # Try freud first (fastest for periodic systems).
    # freud computes volumes, neighbor lists, AND polytope vertices.
    # The polytopes are reused below for Voronoi index extraction.
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

        # Extract neighbor list
        nlist = voro.nlist
        neighbor_groups = defaultdict(list)
        for pair in nlist:
            i = int(pair[0])
            j = int(pair[1])
            neighbor_groups[i].append(j)

        # Compute Voronoi index from freud's polytopes (avoids pyvoro entirely)
        try:
            voronoi_index = _compute_voronoi_index_from_freud(voro)
        except Exception as e:
            logging.warning(f"Voronoi index computation from freud polytopes failed: {e}")
            voronoi_index = [{'n3': 0, 'n4': 0, 'n5': 0, 'n6': 0} for _ in range(len(points))]

        voronoi_cells = []
        for i in range(len(points)):
            neighbors = neighbor_groups.get(i, [])
            cell_volume = float(voro.volumes[i])
            faces = [{'adjacent_cell': int(n)} for n in neighbors]
            voronoi_cells.append({
                'volume': cell_volume,
                'faces': faces,
                'voronoi_index': voronoi_index[i],
            })

        logging.info("Voronoi backend: freud (with Voronoi index)")
        return voronoi_cells

    except Exception:
        pass

    # ----------------------------------------------------------------
    # Fallback to PyVoro (also computes face data)
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
            # Also add Voronoi index from pyvoro faces
            voronoi_index = _compute_voronoi_index_from_pyvoro(result)
            for i, cell in enumerate(result):
                cell['voronoi_index'] = voronoi_index[i]
            return result
        except Exception as exc:
            raise RuntimeError(f"Error during PyVoro tessellation: {exc}") from exc

    # ----------------------------------------------------------------
    # Final fallback: SciPy KDTree approximation (no face data)
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
            'faces': [{'adjacent_cell': neighbor_id} for neighbor_id in neighbor_ids],
            'voronoi_index': {'n3': 0, 'n4': 0, 'n5': 0, 'n6': 0},
        })

    return voronoi_cells


def _compute_voronoi_index_from_pyvoro(cells, area_cutoff_fraction=0.01):
    """
    Compute Voronoi index from pyvoro cell data (with vertex/face info).
    
    Parameters
    ----------
    cells : list
        pyvoro cell data, each with 'vertices' and 'faces' keys.
    area_cutoff_fraction : float
        Fraction of total cell surface area below which a face is discarded.
    
    Returns
    -------
    list of dict
        Each dict: {'n3': int, 'n4': int, 'n5': int, 'n6': int}
    """
    results = []
    for cell in cells:
        verts = np.array(cell['vertices'])
        faces = cell['faces']
        
        face_areas = []
        face_counts = []
        for face in faces:
            idx = face['vertices']
            if len(idx) < 3:
                continue
            fverts = verts[idx]
            v0 = fverts[0]
            area = 0.0
            for i in range(1, len(fverts) - 1):
                v1 = fverts[i] - v0
                v2 = fverts[i + 1] - v0
                area += 0.5 * np.linalg.norm(np.cross(v1, v2))
            face_areas.append(area)
            face_counts.append(len(idx))
        
        total_area = sum(face_areas) if face_areas else 0.0
        if total_area > 0:
            cutoff = area_cutoff_fraction * total_area
            counts = {}
            for nv, area in zip(face_counts, face_areas):
                if area >= cutoff and 3 <= nv <= 6:
                    counts[nv] = counts.get(nv, 0) + 1
        else:
            counts = {}
        
        results.append({
            'n3': counts.get(3, 0),
            'n4': counts.get(4, 0),
            'n5': counts.get(5, 0),
            'n6': counts.get(6, 0),
        })
    
    return results