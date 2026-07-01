import os
import yaml
import subprocess

config_path = "Atomic_Simulation_Post-Processing_Pipeline/config.yaml"
datasets = ["5050", "4654", "6436"]

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

for ds in datasets:
    print(f"==========================================")
    print(f"Regenerating training features for {ds}...")
    print(f"==========================================")
    
    cfg["SNAPSHOT_DIRECTORY"] = f"files/snapshots_{ds}/"
    cfg["output_csv"] = f"outputs/features_{ds}.csv"
    cfg["PLOT_OUTPUT_FILE"] = f"outputs/sample_peak_{ds}.png"
    
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
        
    cmd = ["conda", "run", "-n", "materials-sim-ml", "python3", "Atomic_Simulation_Post-Processing_Pipeline/src/pipeline_orchestrator.py", "--config", config_path]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"ERROR on {ds}")
        break

# Restore config to polyamorphous
cfg["SNAPSHOT_DIRECTORY"] = "files/snapshots_polyamorphous/"
cfg["output_csv"] = "outputs/features_polyamorphous.csv"
cfg["PLOT_OUTPUT_FILE"] = "outputs/sample_peak_polyamorphous.png"
with open(config_path, "w") as f:
    yaml.dump(cfg, f)
print("\nTraining regeneration complete and config restored.")
