import os

# 1. temporal_averaging.py
ta_path = 'Atomic_Simulation_Post-Processing_Pipeline/src/temporal_averaging.py'
with open(ta_path, 'r') as f:
    content = f.read()

old = "    isb_avg_over_time = grouped_atoms['isb'].mean().to_dict()"
new = old + "\n    mean_neighbor_volume_avg = grouped_atoms['mean_neighbor_volume'].mean().to_dict()\n    std_neighbor_volume_avg = grouped_atoms['std_neighbor_volume'].mean().to_dict()\n    mean_neighbor_pentagon_fraction_avg = grouped_atoms['mean_neighbor_pentagon_fraction'].mean().to_dict()\n    std_neighbor_pentagon_fraction_avg = grouped_atoms['std_neighbor_pentagon_fraction'].mean().to_dict()\n    mean_neighbor_CN_avg = grouped_atoms['mean_neighbor_CN'].mean().to_dict()\n    mean_neighbor_free_volume_avg = grouped_atoms['mean_neighbor_free_volume'].mean().to_dict()\n    mean_neighbor_asphericity_avg = grouped_atoms['mean_neighbor_asphericity'].mean().to_dict()"
content = content.replace(old, new)

old = "            averaged_entry['isb_temporal'] = isb_avg_over_time.get(atom_id, np.nan)"
new = old + "\n            averaged_entry['mean_neighbor_volume_temporal'] = mean_neighbor_volume_avg.get(atom_id, np.nan)\n            averaged_entry['std_neighbor_volume_temporal'] = std_neighbor_volume_avg.get(atom_id, np.nan)\n            averaged_entry['mean_neighbor_pentagon_fraction_temporal'] = mean_neighbor_pentagon_fraction_avg.get(atom_id, np.nan)\n            averaged_entry['std_neighbor_pentagon_fraction_temporal'] = std_neighbor_pentagon_fraction_avg.get(atom_id, np.nan)\n            averaged_entry['mean_neighbor_CN_temporal'] = mean_neighbor_CN_avg.get(atom_id, np.nan)\n            averaged_entry['mean_neighbor_free_volume_temporal'] = mean_neighbor_free_volume_avg.get(atom_id, np.nan)\n            averaged_entry['mean_neighbor_asphericity_temporal'] = mean_neighbor_asphericity_avg.get(atom_id, np.nan)"
content = content.replace(old, new)

with open(ta_path, 'w') as f:
    f.write(content)
print("Updated temporal_averaging.py")

# 2. pipeline_orchestrator.py
po_path = 'Atomic_Simulation_Post-Processing_Pipeline/src/pipeline_orchestrator.py'
with open(po_path, 'r') as f:
    content = f.read()

old = "            'isb_temporal': original_temporal_data.get('isb_temporal', np.nan),"
new = old + "\n            'mean_neighbor_volume_temporal': original_temporal_data.get('mean_neighbor_volume_temporal', np.nan),\n            'std_neighbor_volume_temporal': original_temporal_data.get('std_neighbor_volume_temporal', np.nan),\n            'mean_neighbor_pentagon_fraction_temporal': original_temporal_data.get('mean_neighbor_pentagon_fraction_temporal', np.nan),\n            'std_neighbor_pentagon_fraction_temporal': original_temporal_data.get('std_neighbor_pentagon_fraction_temporal', np.nan),\n            'mean_neighbor_CN_temporal': original_temporal_data.get('mean_neighbor_CN_temporal', np.nan),\n            'mean_neighbor_free_volume_temporal': original_temporal_data.get('mean_neighbor_free_volume_temporal', np.nan),\n            'mean_neighbor_asphericity_temporal': original_temporal_data.get('mean_neighbor_asphericity_temporal', np.nan),"
content = content.replace(old, new)

old = "'csro_unlike_temporal', 'csro_like_temporal', 'csro_unlike_std_temporal',"
new = "'csro_unlike_temporal', 'csro_like_temporal', 'csro_unlike_std_temporal',\n        'mean_neighbor_volume_temporal', 'std_neighbor_volume_temporal',\n        'mean_neighbor_pentagon_fraction_temporal', 'std_neighbor_pentagon_fraction_temporal',\n        'mean_neighbor_CN_temporal', 'mean_neighbor_free_volume_temporal',\n        'mean_neighbor_asphericity_temporal',"
content = content.replace(old, new)

with open(po_path, 'w') as f:
    f.write(content)
print("Updated pipeline_orchestrator.py")

# 3. spatial_analysis_levels.py
sa_path = 'Atomic_Simulation_Post-Processing_Pipeline/src/spatial_analysis_levels.py'
with open(sa_path, 'r') as f:
    content = f.read()

old = "            'isb_temporal': atom_data.get('isb_temporal'),"
new = old + "\n            'mean_neighbor_volume_temporal': atom_data.get('mean_neighbor_volume_temporal'),\n            'std_neighbor_volume_temporal': atom_data.get('std_neighbor_volume_temporal'),\n            'mean_neighbor_pentagon_fraction_temporal': atom_data.get('mean_neighbor_pentagon_fraction_temporal'),\n            'std_neighbor_pentagon_fraction_temporal': atom_data.get('std_neighbor_pentagon_fraction_temporal'),\n            'mean_neighbor_CN_temporal': atom_data.get('mean_neighbor_CN_temporal'),\n            'mean_neighbor_free_volume_temporal': atom_data.get('mean_neighbor_free_volume_temporal'),\n            'mean_neighbor_asphericity_temporal': atom_data.get('mean_neighbor_asphericity_temporal'),"
content = content.replace(old, new)

with open(sa_path, 'w') as f:
    f.write(content)
print("Updated spatial_analysis_levels.py")

# 4. feature_builder_data_clean.py
fb_path = 'Machine_Learning_Pipeline_for_Materials_Science/src/feature_builder_data_clean.py'
with open(fb_path, 'r') as f:
    content = f.read()

old = "    csro_unlike_std = df.get(\"csro_unlike_std_temporal\", np.nan).astype(float)"
new = old + "\n    mean_neighbor_volume = df.get(\"mean_neighbor_volume_temporal\", np.nan).astype(float)\n    std_neighbor_volume = df.get(\"std_neighbor_volume_temporal\", np.nan).astype(float)\n    mean_neighbor_pentagon_fraction = df.get(\"mean_neighbor_pentagon_fraction_temporal\", np.nan).astype(float)\n    std_neighbor_pentagon_fraction = df.get(\"std_neighbor_pentagon_fraction_temporal\", np.nan).astype(float)\n    mean_neighbor_CN = df.get(\"mean_neighbor_CN_temporal\", np.nan).astype(float)\n    mean_neighbor_free_volume = df.get(\"mean_neighbor_free_volume_temporal\", np.nan).astype(float)\n    mean_neighbor_asphericity = df.get(\"mean_neighbor_asphericity_temporal\", np.nan).astype(float)"
content = content.replace(old, new)

old = '        "csro_unlike_std_temporal": csro_unlike_std,'
new = old + '\n        "mean_neighbor_volume_temporal": mean_neighbor_volume,\n        "std_neighbor_volume_temporal": std_neighbor_volume,\n        "mean_neighbor_pentagon_fraction_temporal": mean_neighbor_pentagon_fraction,\n        "std_neighbor_pentagon_fraction_temporal": std_neighbor_pentagon_fraction,\n        "mean_neighbor_CN_temporal": mean_neighbor_CN,\n        "mean_neighbor_free_volume_temporal": mean_neighbor_free_volume,\n        "mean_neighbor_asphericity_temporal": mean_neighbor_asphericity,'
content = content.replace(old, new)

with open(fb_path, 'w') as f:
    f.write(content)
print("Updated feature_builder_data_clean.py")

print("\nAll downstream files updated!")