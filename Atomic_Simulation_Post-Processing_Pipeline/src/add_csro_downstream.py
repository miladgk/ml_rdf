"""
Add CSRO columns to all downstream files.
"""
import os

# 1. temporal_averaging.py - add averaging variables
ta_path = 'Atomic_Simulation_Post-Processing_Pipeline/src/temporal_averaging.py'
with open(ta_path, 'r') as f:
    content = f.read()

# Add after isb_avg_over_time line
old = "    isb_avg_over_time = grouped_atoms['isb'].mean().to_dict()"
new = old + "\n    csro_unlike_avg_over_time = grouped_atoms['csro_unlike'].mean().to_dict()\n    csro_like_avg_over_time = grouped_atoms['csro_like'].mean().to_dict()\n    csro_unlike_std_over_time = grouped_atoms['csro_unlike'].std().to_dict()"
content = content.replace(old, new)

# Add output entries after isb_temporal
old = "            averaged_entry['isb_temporal'] = isb_avg_over_time.get(atom_id, np.nan)"
new = old + "\n            averaged_entry['csro_unlike_temporal'] = csro_unlike_avg_over_time.get(atom_id, np.nan)\n            averaged_entry['csro_like_temporal'] = csro_like_avg_over_time.get(atom_id, np.nan)\n            averaged_entry['csro_unlike_std_temporal'] = csro_unlike_std_over_time.get(atom_id, np.nan)"
content = content.replace(old, new)

with open(ta_path, 'w') as f:
    f.write(content)
print("Updated temporal_averaging.py")

# 2. pipeline_orchestrator.py - add CSV columns
po_path = 'Atomic_Simulation_Post-Processing_Pipeline/src/pipeline_orchestrator.py'
with open(po_path, 'r') as f:
    content = f.read()

old = "            'isb_temporal': original_temporal_data.get('isb_temporal', np.nan),"
new = old + "\n            'csro_unlike_temporal': original_temporal_data.get('csro_unlike_temporal', np.nan),\n            'csro_like_temporal': original_temporal_data.get('csro_like_temporal', np.nan),\n            'csro_unlike_std_temporal': original_temporal_data.get('csro_unlike_std_temporal', np.nan),"
content = content.replace(old, new)

old = "'s2_entropy_temporal', 's2_entropy_avg_temporal', 'isb_temporal',"
new = "'s2_entropy_temporal', 's2_entropy_avg_temporal', 'isb_temporal',\n        'csro_unlike_temporal', 'csro_like_temporal', 'csro_unlike_std_temporal',"
content = content.replace(old, new)

with open(po_path, 'w') as f:
    f.write(content)
print("Updated pipeline_orchestrator.py")

# 3. spatial_analysis_levels.py
sa_path = 'Atomic_Simulation_Post-Processing_Pipeline/src/spatial_analysis_levels.py'
with open(sa_path, 'r') as f:
    content = f.read()

old = "            'isb_temporal': atom_data.get('isb_temporal'),"
new = old + "\n            'csro_unlike_temporal': atom_data.get('csro_unlike_temporal'),\n            'csro_like_temporal': atom_data.get('csro_like_temporal'),\n            'csro_unlike_std_temporal': atom_data.get('csro_unlike_std_temporal'),"
content = content.replace(old, new)

with open(sa_path, 'w') as f:
    f.write(content)
print("Updated spatial_analysis_levels.py")

# 4. feature_builder_data_clean.py
fb_path = 'Machine_Learning_Pipeline_for_Materials_Science/src/feature_builder_data_clean.py'
with open(fb_path, 'r') as f:
    content = f.read()

# Add extraction after isb line
old = "    isb = df.get(\"isb_temporal\", np.nan).astype(float)"
new = old + "\n    csro_unlike = df.get(\"csro_unlike_temporal\", np.nan).astype(float)\n    csro_like = df.get(\"csro_like_temporal\", np.nan).astype(float)\n    csro_unlike_std = df.get(\"csro_unlike_std_temporal\", np.nan).astype(float)"
content = content.replace(old, new)

# Add to DataFrame
old = '        "isb_temporal": isb,'
new = old + '\n        "csro_unlike_temporal": csro_unlike,\n        "csro_like_temporal": csro_like,\n        "csro_unlike_std_temporal": csro_unlike_std,'
content = content.replace(old, new)

with open(fb_path, 'w') as f:
    f.write(content)
print("Updated feature_builder_data_clean.py")

print("\nAll files updated!")