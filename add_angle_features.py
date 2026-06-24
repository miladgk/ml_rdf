"""
Implement bond angle distribution features.
"""
import sys
sys.setrecursionlimit(10000)

# Read snapshot_processor.py
with open('Atomic_Simulation_Post-Processing_Pipeline/src/snapshot_processor.py', 'r') as f:
    content = f.read()

# 1. Add scipy.stats.skew import
old = "import logging\nfrom collections import Counter"
new = "import logging\nfrom collections import Counter\nfrom scipy.stats import skew as scipy_skew"
content = content.replace(old, new)

# 2. Add compute_bond_angle_features function BEFORE process_single_snapshot
old = "def process_single_snapshot(snapshot_file, params, bins_for_rdf_calc, bin_volumes_for_rdf_calc):"
new = """def compute_bond_angle_features(atom_pos, neighbor_positions, neighbor_types,
                                  atom_type, box_lengths):
    \"\"\"
    atom_pos: position of central atom i (3,)
    neighbor_positions: positions of all Voronoi neighbors (N, 3)
    neighbor_types: atom types of neighbors (N,)
    atom_type: type of central atom (1 or 2)
    box_lengths: [Lx, Ly, Lz] for minimum image convention
    
    Returns dict of bond angle features.
    \"\"\"
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


""" + "def process_single_snapshot(snapshot_file, params, bins_for_rdf_calc, bin_volumes_for_rdf_calc):"
content = content.replace(old, new)

# 3. Add bond angle computation to the per-atom loop - after CSRO computation
old = "        # ----------------------------------------------------------------------\n        # Compute local RDF"
new = """        # ----------------------------------------------------------------------
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
        # Compute local RDF"""
content = content.replace(old, new)

# 4. Add angle features to per-atom data dictionary - in the atom_data block
old = "                    'csro_like': csro_like_list[atom_index],"
new = """                    'csro_like': csro_like_list[atom_index],
                    'mean_angle_all': angle_features['mean_angle_all'][atom_index],
                    'std_angle_all': angle_features['std_angle_all'][atom_index],
                    'skewness_angle_all': angle_features['skewness_angle_all'][atom_index],
                    'mean_angle_liketype': angle_features['mean_angle_liketype'][atom_index],
                    'mean_angle_unliketype': angle_features['mean_angle_unliketype'][atom_index],
                    'mean_angle_mixedtype': angle_features['mean_angle_mixedtype'][atom_index],
                    'count_liketype_pairs': angle_features['count_liketype_pairs'][atom_index],
                    'count_unliketype_pairs': angle_features['count_unliketype_pairs'][atom_index],"""
content = content.replace(old, new)

with open('Atomic_Simulation_Post-Processing_Pipeline/src/snapshot_processor.py', 'w') as f:
    f.write(content)
print("Updated snapshot_processor.py")