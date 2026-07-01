"""
Add angle features to downstream files.
"""
import os

# 1. temporal_averaging.py
ta_path = 'Atomic_Simulation_Post-Processing_Pipeline/src/temporal_averaging.py'
with open(ta_path, 'r') as f:
    content = f.read()

# Add after csro_unlike_std_over_time line
old = "    csro_unlike_std_over_time = grouped_atoms['csro_unlike'].std().to_dict()"
new = old + "\n    mean_angle_all_avg = grouped_atoms['mean_angle_all'].mean().to_dict()\n    std_angle_all_avg = grouped_atoms['std_angle_all'].mean().to_dict()\n    skewness_angle_all_avg = grouped_atoms['skewness_angle_all'].mean().to_dict()\n    mean_angle_liketype_avg = grouped_atoms['mean_angle_liketype'].mean().to_dict()\n    mean_angle_unliketype_avg = grouped_atoms['mean_angle_unliketype'].mean().to_dict()\n    mean_angle_mixedtype_avg = grouped_atoms['mean_angle_mixedtype'].mean().to_dict()"
content = content.replace(old, new)

# Add output entries after csro_unlike_std_temporal
old = "            averaged_entry['csro_unlike_std_temporal'] = csro_unlike_std_over_time.get(atom_id, np.nan)"
new = old + "\n            averaged_entry['mean_angle_all_temporal'] = mean_angle_all_avg.get(atom_id, np.nan)\n            averaged_entry['std_angle_all_temporal'] = std_angle_all_avg.get(atom_id, np.nan)\n            averaged_entry['skewness_angle_all_temporal'] = skewness_angle_all_avg.get(atom_id, np.nan)\n            averaged_entry['mean_angle_liketype_temporal'] = mean_angle_liketype_avg.get(atom_id, np.nan)\n            averaged_entry['mean_angle_unliketype_temporal'] = mean_angle_unliketype_avg.get(atom_id, np.nan)\n            averaged_entry['mean_angle_mixedtype_temporal'] = mean_angle_mixedtype_avg.get(atom_id, np.nan)"
content = content.replace(old, new)

with open(ta_path, 'w') as f:
    f.write(content)
print("Updated temporal_averaging.py")

# 2. pipeline_orchestrator.py
po_path = 'Atomic_Simulation_Post-Processing_Pipeline/src/pipeline_orchestrator.py'
with open(po_path, 'r') as f:
    content = f.read()

old = "            'csro_unlike_std_temporal': original_temporal_data.get('csro_unlike_std_temporal', np.nan),"
new = old + "\n            'mean_angle_all_temporal': original_temporal_data.get('mean_angle_all_temporal', np.nan),\n            'std_angle_all_temporal': original_temporal_data.get('std_angle_all_temporal', np.nan),\n            'skewness_angle_all_temporal': original_temporal_data.get('skewness_angle_all_temporal', np.nan),\n            'mean_angle_liketype_temporal': original_temporal_data.get('mean_angle_liketype_temporal', np.nan),\n            'mean_angle_unliketype_temporal': original_temporal_data.get('mean_angle_unliketype_temporal', np.nan),\n            'mean_angle_mixedtype_temporal': original_temporal_data.get('mean_angle_mixedtype_temporal', np.nan),"
content = content.replace(old, new)

old = "'csro_unlike_temporal', 'csro_like_temporal', 'csro_unlike_std_temporal',"
new = "'csro_unlike_temporal', 'csro_like_temporal', 'csro_unlike_std_temporal',\n        'mean_angle_all_temporal', 'std_angle_all_temporal', 'skewness_angle_all_temporal',\n        'mean_angle_liketype_temporal', 'mean_angle_unliketype_temporal', 'mean_angle_mixedtype_temporal',"
content = content.replace(old, new)

with open(po_path, 'w') as f:
    f.write(content)
print("Updated pipeline_orchestrator.py")

# 3. spatial_analysis_levels.py
sa_path = 'Atomic_Simulation_Post-Processing_Pipeline/src/spatial_analysis_levels.py'
with open(sa_path, 'r') as f:
    content = f.read()

old = "            'csro_unlike_std_temporal': atom_data.get('csro_unlike_std_temporal'),"
new = old + "\n            'mean_angle_all_temporal': atom_data.get('mean_angle_all_temporal'),\n            'std_angle_all_temporal': atom_data.get('std_angle_all_temporal'),\n            'skewness_angle_all_temporal': atom_data.get('skewness_angle_all_temporal'),\n            'mean_angle_liketype_temporal': atom_data.get('mean_angle_liketype_temporal'),\n            'mean_angle_unliketype_temporal': atom_data.get('mean_angle_unliketype_temporal'),\n            'mean_angle_mixedtype_temporal': atom_data.get('mean_angle_mixedtype_temporal'),"
content = content.replace(old, new)

with open(sa_path, 'w') as f:
    f.write(content)
print("Updated spatial_analysis_levels.py")

# 4. feature_builder_data_clean.py
fb_path = 'Machine_Learning_Pipeline_for_Materials_Science/src/feature_builder_data_clean.py'
with open(fb_path, 'r') as f:
    content = f.read()

# Add extraction after csro_unlike_std line
old = "    csro_unlike_std = df.get(\"csro_unlike_std_temporal\", np.nan).astype(float)"
new = old + "\n    mean_angle_all = df.get(\"mean_angle_all_temporal\", np.nan).astype(float)\n    std_angle_all = df.get(\"std_angle_all_temporal\", np.nan).astype(float)\n    skewness_angle_all = df.get(\"skewness_angle_all_temporal\", np.nan).astype(float)\n    mean_angle_liketype = df.get(\"mean_angle_liketype_temporal\", np.nan).astype(float)\n    mean_angle_unliketype = df.get(\"mean_angle_unliketype_temporal\", np.nan).astype(float)\n    mean_angle_mixedtype = df.get(\"mean_angle_mixedtype_temporal\", np.nan).astype(float)"
content = content.replace(old, new)

# Add to DataFrame
old = '        "csro_unlike_std_temporal": csro_unlike_std,'
new = old + '\n        "mean_angle_all_temporal": mean_angle_all,\n        "std_angle_all_temporal": std_angle_all,\n        "skewness_angle_all_temporal": skewness_angle_all,\n        "mean_angle_liketype_temporal": mean_angle_liketype,\n        "mean_angle_unliketype_temporal": mean_angle_unliketype,\n        "mean_angle_mixedtype_temporal": mean_angle_mixedtype,'
content = content.replace(old, new)

with open(fb_path, 'w') as f:
    f.write(content)
print("Updated feature_builder_data_clean.py")

print("\nAll downstream files updated!")