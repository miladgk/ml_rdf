#!/bin/bash
# Re-run all datasets with the FIXED MRO implementation
CONFIG="Atomic_Simulation_Post-Processing_Pipeline/config.yaml"
DATASETS=("5050" "4654" "6436")

for DS in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Processing dataset: $DS  (fixed MRO)"
    echo "=========================================="
    
    sed -i "s|SNAPSHOT_DIRECTORY: .*|SNAPSHOT_DIRECTORY: \"files/snapshots_${DS}/\"|" "$CONFIG"
    sed -i "s|output_csv: .*|output_csv: \"outputs/features_${DS}.csv\"|" "$CONFIG"
    sed -i "s|PLOT_OUTPUT_FILE: .*|PLOT_OUTPUT_FILE: \"outputs/sample_peak_${DS}.png\"|" "$CONFIG"
    
    /home/m/miniconda3/envs/materials-sim-ml/bin/python3 -u Atomic_Simulation_Post-Processing_Pipeline/src/pipeline_orchestrator.py --config "$CONFIG" 2>&1 | tail -3
    
    echo "Done with $DS"
    echo ""
done

# Restore config
sed -i "s|SNAPSHOT_DIRECTORY: .*|SNAPSHOT_DIRECTORY: \"files/snapshots_polyamorphous/\"|" "$CONFIG"
sed -i "s|output_csv: .*|output_csv: \"outputs/features_polyamorphous.csv\"|" "$CONFIG"
sed -i "s|PLOT_OUTPUT_FILE: .*|PLOT_OUTPUT_FILE: \"outputs/sample_peak_polyamorphous.png\"|" "$CONFIG"
echo "Config restored to polyamorphous"