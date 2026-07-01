#!/bin/bash
CONFIG="Atomic_Simulation_Post-Processing_Pipeline/config.yaml"
ENV_PY="/home/m/miniconda3/envs/materials-sim-ml/bin/python3"

for DS in 5050 4654 6436; do
    echo "=========================================="
    echo "Processing $DS (with 2nd-level MRO)"
    echo "=========================================="
    sed -i "s|SNAPSHOT_DIRECTORY: .*|SNAPSHOT_DIRECTORY: \"files/snapshots_${DS}/\"|" "$CONFIG"
    sed -i "s|output_csv: .*|output_csv: \"outputs/features_${DS}.csv\"|" "$CONFIG"
    $ENV_PY Atomic_Simulation_Post-Processing_Pipeline/src/pipeline_orchestrator.py --config "$CONFIG" 2>&1 | tail -3
    echo "Done with $DS"
done
echo "All datasets complete"