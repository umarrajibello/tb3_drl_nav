#!/bin/bash

OUTPUT_DIR="./eval_results"
mkdir -p "$OUTPUT_DIR"

for model_dir in ./models/*_final; do
  [ -d "$model_dir" ] || continue
  name=$(basename "$model_dir")
  
  # Example: dense_sac_end2end_seed42_independent_final
  IFS='_' read -r env_type algo rep seed_part suffix _ <<< "$name"
  
  seed="${seed_part#seed}"
  
  if [ "$rep" == "hybrid" ]; then
    hybrid_flag="--hybrid"
  else
    hybrid_flag=""
  fi
  
  # Fix algorithm names with dash
  if [[ "$algo" == "lstm"* ]]; then
    algo="lstm-sac"
  elif [[ "$algo" == "tx"* ]]; then
    algo="tx-sac"
  fi
  
  echo "Evaluating: $name"
  python eval.py \
    --model_path "$model_dir/model.zip" \
    --algorithm "$algo" \
    --env_type dense \
    $hybrid_flag \
    --num_episodes 100 \
    --output "$OUTPUT_DIR/${name}.csv"
done

echo "All evaluations complete."