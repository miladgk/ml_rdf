"""Add second-level MRO features to snapshot_processor.py and downstream files."""
import os

# =============================================================================
# 1. Add compute_second_level_mro_features function
# =============================================================================
sp_path = 'Atomic_Simulation_Post-Processing_Pipeline/src/snapshot_processor.py'
with open(sp_path, 'r') as f:
    content = f.read()

# Add function after compute_neighbor_averaged_voronoi_features
old = "def process_single_snapshot(snapshot_file, params, bins_for_rdf_calc, bin_volumes_for_rdf_calc):"
new = """def compute_second_level_mro_features(atom_idx, first_shell_indices,
                                           neighbor_lists_by_idx, per_atom_lookup):
    \"\"\"
    Compute second-level MRO by averaging over neighbors of neighbors.
    atom_idx: 0-based index of central atom
    first_shell_indices: list of 0-based neighbor indices (first shell)
    neighbor_lists_by_idx: dict mapping 0-based idx -> list of 0-based neighbor idxs
    per_atom_lookup: dict mapping 0-based idx -> dict of per-atom features
    \"\"\"
    vols, fvs, pents, cns = [], [], [], []
    
    for j_idx in first_shell_indices:
        if j_idx not in neighbor_lists_by_idx:
            continue
        for k_idx in neighbor_lists_by_idx[j_idx]:
            if k_idx == atom_idx:
                continue
            if k_idx not in per_atom_lookup:
                continue
            kf = per_atom_lookup[k_idx]
            if not np.isnan(kf['voronoi_volume']): vols.append(kf['voronoi_volume'])
            if not np.isnan(kf['free_volume']): fvs.append(kf['free_volume'])
            if not np.isnan(kf['pentagon_fraction']): pents.append(kf['pentagon_fraction'])
            if not np.isnan(kf['num_neighbors']): cns.append(kf['num_neighbors'])
    
    return {
        'mean_2nd_neighbor_volume': np.mean(vols) if vols else np.nan,
        'std_2nd_neighbor_volume': np.std(vols) if len(vols) > 1 else np.nan,
        'mean_2nd_neighbor_free_volume': np.mean(fvs) if fvs else np.nan,
        'mean_2nd_neighbor_pentagon_fraction': np.mean(pents) if pents else np.nan,
        'mean_2nd_neighbor_CN': np.mean(cns) if cns else np.nan,
        'n_second_shell_samples': len(vols),
    }


""" + "def process_single_snapshot(snapshot_file, params, bins_for_rdf_calc, bin_volumes_for_rdf_calc):"
content = content.replace(old, new)

# Add Pass 3 after Pass 2
old = "        logging.info(f\"MRO features computed for {snapshot_file}.\")\n\n        # ----------------------------------------------------------------------\n        # Snapshot metadata"
new = """        logging.info(f"MRO features computed for {snapshot_file}.")

        # ----------------------------------------------------------------------
        # Pass 3: Second-level MRO (neighbors of neighbors)
        # ----------------------------------------------------------------------
        logging.info(f"Computing second-level MRO features for {snapshot_file}...")
        
        # Build neighbor lookup by 0-based index (same keys as per_atom_lookup)
        neighbor_lists_by_idx = {
            idx: processed_vor_neighbors[idx] for idx in range(num_atoms)
        }
        
        second_mro_keys = ['mean_2nd_neighbor_volume', 'std_2nd_neighbor_volume',
                           'mean_2nd_neighbor_free_volume', 'mean_2nd_neighbor_pentagon_fraction',
                           'mean_2nd_neighbor_CN', 'n_second_shell_samples']
        second_mro_features = {k: np.full(num_atoms, np.nan) for k in second_mro_keys}
        
        for atom_index in range(num_atoms):
            first_shell = processed_vor_neighbors[atom_index]
            result = compute_second_level_mro_features(
                atom_index, first_shell, neighbor_lists_by_idx, per_atom_lookup
            )
            for k in second_mro_keys:
                second_mro_features[k][atom_index] = result[k]
        
        # Add second-level MRO columns to atom_data
        for ad in atom_data:
            atom_id = ad['id']
            orig_idx = id_to_idx[atom_id]
            for k in second_mro_keys:
                ad[k] = second_mro_features[k][orig_idx]
        
        logging.info(f"Second-level MRO features computed for {snapshot_file}.")

        # ----------------------------------------------------------------------
        # Snapshot metadata"""
content = content.replace(old, new)

with open(sp_path, 'w') as f:
    f.write(content)
print("Updated snapshot_processor.py with second-level MRO")

# =============================================================================
# 2. Update temporal_averaging.py
# =============================================================================
ta_path = 'Atomic_Simulation_Post-Processing_Pipeline/src/temporal_averaging.py'
with open(ta_path, 'r') as f:
    content = f.read()

old = "    mean_neighbor_asphericity_avg = grouped_atoms['mean_neighbor_asphericity'].mean().to_dict()"
new = old + "\n    mean_2nd_neighbor_volume_avg = grouped_atoms['mean_2nd_neighbor_volume'].mean().to_dict()\n    std_2nd_neighbor_volume_avg = grouped_atoms['std_2nd_neighbor_volume'].mean().to_dict()\n    mean_2nd_neighbor_free_volume_avg = grouped_atoms['mean_2nd_neighbor_free_volume'].mean().to_dict()\n    mean_2nd_neighbor_pentagon_fraction_avg = grouped_atoms['mean_2nd_neighbor_pentagon_fraction'].mean().to_dict()\n    mean_2nd_neighbor_CN_avg = grouped_atoms['mean_2nd_neighbor_CN'].mean().to_dict()"
content = content.replace(old, new)

old = "            averaged_entry['mean_neighbor_asphericity_temporal'] = mean_neighbor_asphericity_avg.get(atom_id, np.nan)"
new = old + "\n            averaged_entry['mean_2nd_neighbor_volume_temporal'] = mean_2nd_neighbor_volume_avg.get(atom_id, np.nan)\n            averaged_entry['std_2nd_neighbor_volume_temporal'] = std_2nd_neighbor_volume_avg.get(atom_id, np.nan)\n            averaged_entry['mean_2nd_neighbor_free_volume_temporal'] = mean_2nd_neighbor_free_volume_avg.get(atom_id, np.nan)\n            averaged_entry['mean_2nd_neighbor_pentagon_fraction_temporal'] = mean_2nd_neighbor_pentagon_fraction_avg.get(atom_id, np.nan)\n            averaged_entry['mean_2nd_neighbor_CN_temporal'] = mean_2nd_neighbor_CN_avg.get(atom_id, np.nan)"
content = content.replace(old, new)

with open(ta_path, 'w') as f:
    f.write(content)
print("Updated temporal_averaging.py")

# =============================================================================
# 3. Update pipeline_orchestrator.py
# =============================================================================
po_path = 'Atomic_Simulation_Post-Processing_Pipeline/src/pipeline_orchestrator.py'
with open(po_path, 'r') as f:
    content = f.read()

old = "            'mean_neighbor_asphericity_temporal': original_temporal_data.get('mean_neighbor_asphericity_temporal', np.nan),"
new = old + "\n            'mean_2nd_neighbor_volume_temporal': original_temporal_data.get('mean_2nd_neighbor_volume_temporal', np.nan),\n            'std_2nd_neighbor_volume_temporal': original_temporal_data.get('std_2nd_neighbor_volume_temporal', np.nan),\n            'mean_2nd_neighbor_free_volume_temporal': original_temporal_data.get('mean_2nd_neighbor_free_volume_temporal', np.nan),\n            'mean_2nd_neighbor_pentagon_fraction_temporal': original_temporal_data.get('mean_2nd_neighbor_pentagon_fraction_temporal', np.nan),\n            'mean_2nd_neighbor_CN_temporal': original_temporal_data.get('mean_2nd_neighbor_CN_temporal', np.nan),"
content = content.replace(old, new)

old = "        'mean_neighbor_asphericity_temporal',"
new = "        'mean_neighbor_asphericity_temporal',\n        'mean_2nd_neighbor_volume_temporal', 'std_2nd_neighbor_volume_temporal',\n        'mean_2nd_neighbor_free_volume_temporal', 'mean_2nd_neighbor_pentagon_fraction_temporal',\n        'mean_2nd_neighbor_CN_temporal',"
content = content.replace(old, new)

with open(po_path, 'w') as f:
    f.write(content)
print("Updated pipeline_orchestrator.py")

# =============================================================================
# 4. Update spatial_analysis_levels.py
# =============================================================================
sa_path = 'Atomic_Simulation_Post-Processing_Pipeline/src/spatial_analysis_levels.py'
with open(sa_path, 'r') as f:
    content = f.read()

old = "            'mean_neighbor_asphericity_temporal': atom_data.get('mean_neighbor_asphericity_temporal'),"
new = old + "\n            'mean_2nd_neighbor_volume_temporal': atom_data.get('mean_2nd_neighbor_volume_temporal'),\n            'std_2nd_neighbor_volume_temporal': atom_data.get('std_2nd_neighbor_volume_temporal'),\n            'mean_2nd_neighbor_free_volume_temporal': atom_data.get('mean_2nd_neighbor_free_volume_temporal'),\n            'mean_2nd_neighbor_pentagon_fraction_temporal': atom_data.get('mean_2nd_neighbor_pentagon_fraction_temporal'),\n            'mean_2nd_neighbor_CN_temporal': atom_data.get('mean_2nd_neighbor_CN_temporal'),"
content = content.replace(old, new)

with open(sa_path, 'w') as f:
    f.write(content)
print("Updated spatial_analysis_levels.py")

# =============================================================================
# 5. Update feature_builder_data_clean.py
# =============================================================================
fb_path = 'Machine_Learning_Pipeline_for_Materials_Science/src/feature_builder_data_clean.py'
with open(fb_path, 'r') as f:
    content = f.read()

old = "    mean_neighbor_asphericity = df.get(\"mean_neighbor_asphericity_temporal\", np.nan).astype(float)"
new = old + "\n    mean_2nd_neighbor_volume = df.get(\"mean_2nd_neighbor_volume_temporal\", np.nan).astype(float)\n    std_2nd_neighbor_volume = df.get(\"std_2nd_neighbor_volume_temporal\", np.nan).astype(float)\n    mean_2nd_neighbor_free_volume = df.get(\"mean_2nd_neighbor_free_volume_temporal\", np.nan).astype(float)\n    mean_2nd_neighbor_pentagon_fraction = df.get(\"mean_2nd_neighbor_pentagon_fraction_temporal\", np.nan).astype(float)\n    mean_2nd_neighbor_CN = df.get(\"mean_2nd_neighbor_CN_temporal\", np.nan).astype(float)"
content = content.replace(old, new)

old = '        "mean_neighbor_asphericity_temporal": mean_neighbor_asphericity,'
new = old + '\n        "mean_2nd_neighbor_volume_temporal": mean_2nd_neighbor_volume,\n        "std_2nd_neighbor_volume_temporal": std_2nd_neighbor_volume,\n        "mean_2nd_neighbor_free_volume_temporal": mean_2nd_neighbor_free_volume,\n        "mean_2nd_neighbor_pentagon_fraction_temporal": mean_2nd_neighbor_pentagon_fraction,\n        "mean_2nd_neighbor_CN_temporal": mean_2nd_neighbor_CN,'
content = content.replace(old, new)

with open(fb_path, 'w') as f:
    f.write(content)
print("Updated feature_builder_data_clean.py")

# =============================================================================
# 6. Update config.yaml
# =============================================================================
cfg_path = 'Machine_Learning_Pipeline_for_Materials_Science/config.yaml'
with open(cfg_path, 'r') as f:
    content = f.read()

old = "mean_neighbor_asphericity_temporal]"
new = "mean_neighbor_asphericity_temporal, mean_2nd_neighbor_volume_temporal, std_2nd_neighbor_volume_temporal, mean_2nd_neighbor_free_volume_temporal, mean_2nd_neighbor_pentagon_fraction_temporal, mean_2nd_neighbor_CN_temporal]"
content = content.replace(old, new)

with open(cfg_path, 'w') as f:
    f.write(content)
print("Updated config.yaml")

print("\nAll files updated!")