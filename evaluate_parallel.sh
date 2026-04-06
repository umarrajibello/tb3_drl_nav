#!/bin/bash
OUTPUT_DIR="./eval_results"
mkdir -p "$OUTPUT_DIR"
MAX_CONCURRENT=6   # adjust to number of CPU cores (e.g., 6)

# Find all model directories
model_dirs=(./models/dense_*_final)

# Function to evaluate one model
eval_one() {
    model_dir=$1
    name=$(basename "$model_dir")
    if [[ "$name" == *"_hybrid_"* ]]; then
        hybrid_flag="--hybrid"
    else
        hybrid_flag=""
    fi
    # Extract algorithm
    if [[ "$name" == *"lstm-sac"* ]]; then
        algo="lstm-sac"
    elif [[ "$name" == *"tx-sac"* ]]; then
        algo="tx-sac"
    elif [[ "$name" == *"sac"* ]]; then
        algo="sac"
    elif [[ "$name" == *"ppo"* ]]; then
        algo="ppo"
    else
        return
    fi
    output="$OUTPUT_DIR/${name}.csv"
    if [ -f "$output" ]; then
        echo "Skipping $name (already evaluated)"
        return
    fi
    echo "Evaluating $name"
    python eval.py \
        --model_path "$model_dir/model.zip" \
        --algorithm "$algo" \
        --env_type dense \
        $hybrid_flag \
        --num_episodes 100 \
        --output "$output"
}

# Run in parallel with job control
for model_dir in "${model_dirs[@]}"; do
    eval_one "$model_dir" &
    # Limit concurrency
    while [ $(jobs -r | wc -l) -ge $MAX_CONCURRENT ]; do
        sleep 5
    done
done
wait
echo "All evaluations finished"