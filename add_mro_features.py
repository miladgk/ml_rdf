"""
Add Medium-Range Order (MRO) neighbor-averaged Voronoi features.
"""
with open('Atomic_Simulation_Post-Processing_Pipeline/src/snapshot_processor.py', 'r') as f:
    content = f.read()

# 1. Add compute_neighbor_averaged_voronoi_features function after compute_bond_angle_features
old = "def process_single_snapshot(snapshot_file, params, bins_for_rdf_calc, bin_volumes_for_rdf_calc):"
new = """def compute_neighbor_averaged_voronoi_features(atom_index, neighbor_indices,
                                                 per_atom_features_dict):
    \"\"\"
    Compute second-shell MRO features from per-atom feature lookup.
    atom_index: int, index of central atom i
    neighbor_indices: list of neighbor indices (row positions)
    per_atom_features_dict: dict[atom_index] -> dict of per-atom features
    \"\"\"
    if len(neighbor_indices) == 0:
        return {k: np.nan for k in [
            'mean_neighbor_volume', 'std_neighbor_volume',
            'mean_neighbor_pentagon_fraction', 'std_neighbor_pentagon_fraction',
            'mean_neighbor_CN', 'mean_neighbor_free_volume', 'mean_neighbor_asphericity'
        ]}
    
    vols, pents, cns, fvs, asps = [], [], [], [], []
    for nb_idx in neighbor_indices:
        if nb_idx not in per_atom_features_dict:
            continue
        nb = per_atom_features_dict[nb_idx]
        if not np.isnan(nb['voronoi_volume']): vols.append(nb['voronoi_volume'])
        if not np.isnan(nb['pentagon_fraction']): pents.append(nb['pentagon_fraction'])
        if not np.isnan(nb['num_neighbors']): cns.append(nb['num_neighbors'])
        if not np.isnan(nb['free_volume']): fvs.append(nb['free_volume'])
        if not np.isnan(nb['asphericity_voronoi']): asps.append(nb['asphericity_voronoi'])
    
    return {
        'mean_neighbor_volume': np.mean(vols) if vols else np.nan,
        'std_neighbor_volume': np.std(vols) if len(vols) > 1 else np.nan,
        'mean_neighbor_pentagon_fraction': np.mean(pents) if pents else np.nan,
        'std_neighbor_pentagon_fraction': np.std(pents) if len(pents) > 1 else np.nan,
        'mean_neighbor_CN': np.mean(cns) if cns else np.nan,
        'mean_neighbor_free_volume': np.mean(fvs) if fvs else np.nan,
        'mean_neighbor_asphericity': np.mean(asps) if asps else np.nan,
    }


""" + "def process_single_snapshot(snapshot_file, params, bins_for_rdf_calc, bin_volumes_for_rdf_calc):"
content = content.replace(old, new)

# 2. After the atom_data assembly loop, add Pass 2 for MRO features
# Find the closing of atom_data assembly
old = "        # ----------------------------------------------------------------------\n        # Snapshot metadata\n        # ----------------------------------------------------------------------"
new = """        # ----------------------------------------------------------------------
        # Pass 2: Compute neighbor-averaged (MRO) features
        # ----------------------------------------------------------------------
        logging.info(f"Computing MRO features for {snapshot_file}...")
        
        # Build lookup dict from atom_data (already assembled)
        per_atom_lookup = {}
        for idx, ad in enumerate(atom_data):
            pentagon = None
            n5 = ad.get('n5_voronoi', 0)
            n_sum = (ad.get('n3_voronoi', 0) + ad.get('n4_voronoi', 0) +
                     n5 + ad.get('n6_voronoi', 0))
            if n_sum > 0:
                pentagon = n5 / n_sum
            
            # free_volume not in per-snapshot data; compute from voronoi_volume and radius
            radius = ad.get('radius', 1.28)
            atomic_vol = (4.0/3.0) * np.pi * (radius ** 3)
            free_vol = ad.get('voronoi_volume', np.nan) - atomic_vol if not np.isnan(ad.get('voronoi_volume', np.nan)) else np.nan
            
            per_atom_lookup[idx] = {
                'voronoi_volume': ad.get('voronoi_volume', np.nan),
                'pentagon_fraction': pentagon,
                'num_neighbors': ad.get('num_neighbors', np.nan),
                'free_volume': free_vol,
                'asphericity_voronoi': ad.get('asphericity_voronoi', np.nan),
            }
        
        mro_keys = ['mean_neighbor_volume', 'std_neighbor_volume',
                    'mean_neighbor_pentagon_fraction', 'std_neighbor_pentagon_fraction',
                    'mean_neighbor_CN', 'mean_neighbor_free_volume', 'mean_neighbor_asphericity']
        mro_features = {k: np.full(num_atoms, np.nan) for k in mro_keys}
        
        for atom_index in range(num_atoms):
            neigh = processed_vor_neighbors[atom_index]
            result = compute_neighbor_averaged_voronoi_features(atom_index, neigh, per_atom_lookup)
            for k in mro_keys:
                mro_features[k][atom_index] = result[k]
        
        # Add MRO columns to atom_data
        for idx, ad in enumerate(atom_data):
            for k in mro_keys:
                ad[k] = mro_features[k][idx]
        
        logging.info(f"MRO features computed for {snapshot_file}.")

        # ----------------------------------------------------------------------
        # Snapshot metadata
        # ----------------------------------------------------------------------"""
content = content.replace(old, new)

with open('Atomic_Simulation_Post-Processing_Pipeline/src/snapshot_processor.py', 'w') as f:
    f.write(content)
print("Updated snapshot_processor.py with MRO features")