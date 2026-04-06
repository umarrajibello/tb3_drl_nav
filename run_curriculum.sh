#!/bin/bash
# Usage:
#   ./run_curriculum.sh <algo> <seed> <num_envs>           (no hybrid)
#   ./run_curriculum.sh <algo> --hybrid <seed> <num_envs>  (with hybrid)

ALGO=$1
if [ "$2" == "--hybrid" ]; then
    HYBRID="--hybrid"
    SEED=$3
    NUM_ENVS=$4
    REP="hybrid"
else
    HYBRID=""
    SEED=$2
    NUM_ENVS=$3
    REP="end2end"
fi

# Base command array with suffix
base_cmd=(python train.py --algorithm "$ALGO" --seed "$SEED" --num_envs "$NUM_ENVS" --gpu --exp_suffix curriculum)
if [ -n "$HYBRID" ]; then
    base_cmd+=("$HYBRID")
fi

# Stage 1: static
echo "Stage 1: static"
cmd1=("${base_cmd[@]}" --env_type static --total_timesteps 150000)
"${cmd1[@]}"

STATIC_DIR="./models/static_${ALGO}_${REP}_seed${SEED}_curriculum_final"
STATIC_MODEL="$STATIC_DIR/model.zip"

# Stage 2: mixed
echo "Stage 2: mixed"
cmd2=("${base_cmd[@]}" --env_type mixed --total_timesteps 150000 --pretrained "$STATIC_MODEL")
"${cmd2[@]}"

MIXED_DIR="./models/mixed_${ALGO}_${REP}_seed${SEED}_curriculum_final"
MIXED_MODEL="$MIXED_DIR/model.zip"

# Stage 3: dense
echo "Stage 3: dense"
cmd3=("${base_cmd[@]}" --env_type dense --total_timesteps 200000 --pretrained "$MIXED_MODEL")
"${cmd3[@]}"