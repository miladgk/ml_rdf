#!/bin/bash
# Generate feature CSVs for all three datasets

DATASETS=("5050" "4654" "6436")
CONFIG="Atomic_Simulation_Post-Processing_Pipeline/config.yaml"

for DS in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Processing dataset: $DS"
    echo "=========================================="
    
    # Update config for this dataset
    sed -i "s|SNAPSHOT_DIRECTORY: .*|SNAPSHOT_DIRECTORY: \"files/snapshots_${DS}/\"|" "$CONFIG"
    sed -i "s|output_csv: .*|output_csv: \"outputs/features_${DS}.csv\"|" "$CONFIG"
    sed -i "s|PLOT_OUTPUT_FILE: .*|PLOT_OUTPUT_FILE: \"outputs/sample_peak_${DS}.png\"|" "$CONFIG"
    
    # Run pipeline
    conda run -n materials-sim-ml python3 Atomic_Simulation_Post-Processing_Pipeline/src/pipeline_orchestrator.py \
        --config "$CONFIG" 2>&1 | tail -5
    
    echo "Done with $DS"
    echo ""
done

# Restore to polyamorphous
sed -i "s|SNAPSHOT_DIRECTORY: .*|SNAPSHOT_DIRECTORY: \"files/snapshots_polyamorphous/\"|" "$CONFIG"
sed -i "s|output_csv: .*|output_csv: \"outputs/features_polyamorphous.csv\"|" "$CONFIG"
sed -i "s|PLOT_OUTPUT_FILE: .*|PLOT_OUTPUT_FILE: \"outputs/sample_peak_polyamorphous.png\"|" "$CONFIG"
echo "Config restored to polyamorphous"