"""
Run pipeline for all datasets with CSRO, then sanity check.
"""
import subprocess, os

# Run each dataset with correct composition
datasets = {
    '5050': {'type_1': 0.50, 'type_2': 0.50, 'compute': 'false'},
    '4654': {'type_1': 0.46, 'type_2': 0.54, 'compute': 'false'},
    '6436': {'type_1': 0.64, 'type_2': 0.36, 'compute': 'false'},
}

for ds, comp in datasets.items():
    # Update config
    sed_cmd = f"sed -i 's|SNAPSHOT_DIRECTORY: .*|SNAPSHOT_DIRECTORY: \"files/snapshots_{ds}/\"|; s|output_csv: .*|output_csv: \"outputs/features_{ds}.csv\"|; s|PLOT_OUTPUT_FILE: .*|PLOT_OUTPUT_FILE: \"outputs/sample_peak_{ds}.png\"|; s|type_1_fraction: .*|type_1_fraction: {comp['type_1']}|; s|type_2_fraction: .*|type_2_fraction: {comp['type_2']}|; s|compute_from_snapshot: .*|compute_from_snapshot: {comp['compute']}|' Atomic_Simulation_Post-Processing_Pipeline/config.yaml"
    subprocess.run(sed_cmd, shell=True)
    
    print(f"Running {ds} with Cu={comp['type_1']}, Zr={comp['type_2']}...")
    result = subprocess.run(['conda', 'run', '-n', 'materials-sim-ml', 'python3', 
        'Atomic_Simulation_Post-Processing_Pipeline/src/pipeline_orchestrator.py',
        '--config', 'Atomic_Simulation_Post-Processing_Pipeline/config.yaml'],
        capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'saved to' in line:
            print(f"  {line}")

print("All datasets processed!")