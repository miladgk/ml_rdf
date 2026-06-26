#!/bin/bash
# Run 4654 and 6436 datasets to regenerate with MRO columns
CONFIG="Atomic_Simulation_Post-Processing_Pipeline/config.yaml"

for DS in 4654 6436; do
    echo "=========================================="
    echo "Processing dataset: $DS"
    echo "=========================================="
    
    sed -i "s|SNAPSHOT_DIRECTORY: .*|SNAPSHOT_DIRECTORY: \"files/snapshots_${DS}/\"|" "$CONFIG"
    sed -i "s|output_csv: .*|output_csv: \"outputs/features_${DS}.csv\"|" "$CONFIG"
    sed -i "s|PLOT_OUTPUT_FILE: .*|PLOT_OUTPUT_FILE: \"outputs/sample_peak_${DS}.png\"|" "$CONFIG"
    
    python3 Atomic_Simulation_Post-Processing_Pipeline/src/pipeline_orchestrator.py --config "$CONFIG" 2>&1 | tail -5
    
    echo "Done with $DS"
    echo ""
done