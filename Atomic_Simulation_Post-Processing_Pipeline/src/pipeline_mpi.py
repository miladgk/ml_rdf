"""pipeline_mpi.py
=================

MPI-based snapshot parallelization for Atomic_Simulation_Post-Processing_Pipeline.

Why this exists
----------------
The current pipeline uses ProcessPoolExecutor and returns large Python objects
(lists/dicts containing per-atom RDF arrays). This results in heavy pickling/
serialization overhead and limits scaling.

This MPI entrypoint distributes snapshot files across ranks. Each rank processes
its own snapshot subset via the existing `snapshot_processor.process_single_snapshot`.
The rank writes compact partial results to disk; rank 0 merges them and runs the
rest of the pipeline (temporal averaging + peak analysis + spatial analysis).

Implementation notes
---------------------
- This module assumes `mpi4py` is installed.
- Partial results are written as compressed NPZ files containing pickled Python
  objects (object arrays). This preserves existing snapshot dict structure without
  re-writing all downstream modules.
- If you prefer pure text (JSON/pickle), you can change the serialization.

Run
---
mpiexec -n <N> python -m Atomic_Simulation_Post-Processing_Pipeline.src.pipeline_mpi

Or, if running from the repository root:
mpiexec -n <N> python Atomic_Simulation_Post-Processing_Pipeline/src/pipeline_mpi.py

"""

from __future__ import annotations

import os
import time
import glob
import pickle
import logging
import numpy as np

from mpi4py import MPI

import pipeline_orchestrator
import rdf
import temporal_averaging
import peak_analysis
import spatial_analysis_levels
import snapshot_processor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)


def _list_snapshot_files(snapshot_dir: str) -> list[str]:
    """Return all *.lammpstrj under snapshot_dir."""
    pattern = os.path.join(snapshot_dir, "**", "*.lammpstrj")
    files = sorted(glob.glob(pattern, recursive=True))
    return files


def _write_partial_results(output_dir: str, rank: int, partial_data: list[dict]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"partial_rank{rank}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(partial_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path


def _load_partial_results(partial_dir: str) -> list[dict]:
    paths = sorted(glob.glob(os.path.join(partial_dir, "partial_rank*.pkl")))
    all_data: list[dict] = []
    for p in paths:
        with open(p, "rb") as f:
            all_data.extend(pickle.load(f))
    return all_data


def run_mpi_pipeline() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start_time = time.time()

    if rank == 0:
        logging.info("[MPI] Loading config")
        params = pipeline_orchestrator.load_config()
    else:
        params = None

    params = comm.bcast(params, root=0)

    if rank == 0:
        logging.info("[MPI] Resolving snapshot paths")

    # Use same resolution logic as orchestrator, but keep it simple:
    # - prefer params['SNAPSHOT_DIRECTORY']
    # - if empty or invalid, fall back to pipeline_orchestrator.resolve_snapshot_paths
    snapshot_dir = params.get("SNAPSHOT_DIRECTORY", "files/")

    if rank == 0:
        if snapshot_dir and os.path.isdir(snapshot_dir):
            snapshot_files = _list_snapshot_files(snapshot_dir)
        else:
            snapshot_files = pipeline_orchestrator.resolve_snapshot_paths(snapshot_dir)

        if not snapshot_files:
            raise RuntimeError(f"[MPI] No snapshot files found in SNAPSHOT_DIRECTORY={snapshot_dir}")

        logging.info(f"[MPI] Found {len(snapshot_files)} snapshots. Distributing across {size} ranks.")
    else:
        snapshot_files = None

    snapshot_files = comm.bcast(snapshot_files, root=0)

    # Split work
    chunks = np.array_split(np.arange(len(snapshot_files)), size)
    my_idx = chunks[rank]
    my_snapshots = [snapshot_files[i] for i in my_idx]

    if rank == 0:
        os.makedirs("outputs/mpi_partials", exist_ok=True)

    # Precompute RDF bins/volumes on all ranks (cheap)
    bins_for_rdf_calc = rdf.create_adaptive_bins(
        params["R_MIN"],
        params["R_MAX"],
        params["NUM_BINS"],
    )
    bin_volumes_for_rdf_calc = 4 / 3 * np.pi * (bins_for_rdf_calc[1:] ** 3 - bins_for_rdf_calc[:-1] ** 3)

    # Process assigned snapshots
    partial_results: list[dict] = []
    logging.info(f"[MPI][rank {rank}] Processing {len(my_snapshots)} snapshots")

    for s in my_snapshots:
        try:
            res = snapshot_processor.process_single_snapshot(
                s,
                params,
                bins_for_rdf_calc,
                bin_volumes_for_rdf_calc,
            )
            if res:
                partial_results.append(res)
        except Exception as e:
            logging.exception(f"[MPI][rank {rank}] Failed snapshot {s}: {e}")

    # Write partials
    comm.Barrier()
    partial_dir = params.get("MPI_PARTIAL_DIR", "outputs/mpi_partials")

    if rank == 0:
        # Clean old partials for deterministic merges
        for old in glob.glob(os.path.join(partial_dir, "partial_rank*.pkl")):
            try:
                os.remove(old)
            except OSError:
                pass

    comm.Barrier()

    _write_partial_results(partial_dir, rank, partial_results)

    comm.Barrier()

    # Merge + final pipeline on rank 0
    if rank == 0:
        logging.info("[MPI] Merging partial snapshot results")
        all_snapshots_processed_data = _load_partial_results(partial_dir)

        if not all_snapshots_processed_data:
            raise RuntimeError("[MPI] No snapshot results merged. Nothing to do.")

        analysis_level = params.get("ANALYSIS_LEVEL", "temporal_first_spatial")
        params["analysis_level"] = analysis_level

        logging.info("[MPI] Temporal averaging")
        initial_atom_data_list, consolidated_metadata = temporal_averaging.perform_temporal_averaging(
            all_snapshots_processed_data
        )

        if not initial_atom_data_list:
            raise RuntimeError("[MPI] Temporal averaging produced no atoms. Aborting.")

        r_values_bin_centers = peak_analysis.calculate_rdf_bin_centers(np.array(consolidated_metadata.bins))

        if analysis_level == "temporal_only":
            logging.info("[MPI] Running peak analysis (temporal_only)")
            final_processed_atom_results = []
            for atom_data in initial_atom_data_list:
                peak_results = peak_analysis.analyze_rdf_peaks(
                    r_values_bins=r_values_bin_centers,
                    rdf_data_array=atom_data["g_i_r_temporal_avg"],
                    analysis_parameters=params,
                    atom_id=atom_data["id"],
                )
                result_entry = {**atom_data, **peak_results}
                if isinstance(result_entry.get("g_i_r_temporal_avg"), np.ndarray):
                    result_entry["g_i_r_temporal_avg"] = result_entry["g_i_r_temporal_avg"].tolist()
                final_processed_atom_results.append(result_entry)

        elif analysis_level == "temporal_first_spatial":
            logging.info("[MPI] Running spatial analysis (temporal_first_spatial)")
            final_processed_atom_results = spatial_analysis_levels.perform_temporal_first_spatial_analysis(
                initial_atom_data_list,
                r_values_bin_centers,
                params,
            )
        else:
            raise ValueError(f"[MPI] Unknown ANALYSIS_LEVEL={analysis_level}")

        # Export (reuse existing function)
        pipeline_orchestrator.csv_export(initial_atom_data_list, final_processed_atom_results, params, analysis_level)

        end_time = time.time()
        logging.info(f"[MPI] Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    run_mpi_pipeline()

