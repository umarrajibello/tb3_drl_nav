#!/bin/bash
# Usage: 
#   ./run_independent_dense.sh <algo> <seed> <num_envs>           (no hybrid)
#   ./run_independent_dense.sh <algo> --hybrid <seed> <num_envs>  (with hybrid)

ALGO=$1
if [ "$2" == "--hybrid" ]; then
    HYBRID="--hybrid"
    SEED=$3
    NUM_ENVS=$4
else
    HYBRID=""
    SEED=$2
    NUM_ENVS=$3
fi

cmd=(python train.py --algorithm "$ALGO" --env_type dense --total_timesteps 500000 --seed "$SEED" --num_envs "$NUM_ENVS" --gpu --exp_suffix independent)
if [ -n "$HYBRID" ]; then
    cmd+=("$HYBRID")
fi
"${cmd[@]}"