import os
import pandas as pd

preds_csv = "Machine_Learning_Pipeline_for_Materials_Science/outputs/polyamorphous_fixed_predictions.csv"
dump_path = "Atomic_Simulation_Post-Processing_Pipeline/files/snapshots_polyamorphous/combined_final_dump.lammpstrj"
out_dump = "Atomic_Simulation_Post-Processing_Pipeline/files/polyamorphous_phase_labeled_FINAL.lammpstrj"

print("Loading predictions...")
df = pd.read_csv(preds_csv)

# Create mapping atom_id -> (pred, conf, true_label)
lookup = {}
for _, row in df.iterrows():
    aid = int(row['id'])
    lookup[aid] = (int(row['pred']), f"{row['conf']:.4f}", int(row['true_label']))

print("Reading LAMMPS dump and attaching aligned labels...")
with open(dump_path, 'r') as f:
    lines = f.readlines()

header = lines[:8]
atoms_header = lines[8].strip()

cols = atoms_header.replace("ITEM: ATOMS", "").strip().split()
id_idx = cols.index('id')

with open(out_dump, 'w') as out:
    for h in header:
        out.write(h)
    out.write(atoms_header + " pred_phase confidence true_phase\n")
    
    atom_count = 0
    for line in lines[9:]:
        parts = line.strip().split()
        if not parts:
            continue
        aid = int(parts[id_idx])
        pred, conf, true_label = lookup.get(aid, (-1, "0.0000", -1))
        out.write(" ".join(parts) + f" {pred} {conf} {true_label}\n")
        atom_count += 1

print(f"Successfully wrote {atom_count} atoms to {out_dump}")
