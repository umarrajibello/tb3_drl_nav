#!/bin/bash

ALGOS=(sac ppo lstm-sac tx-sac)
HYBRIDS=("" "--hybrid")
TRAININGS=(independent curriculum)
SEEDS=(42 123 456)

NUM_ENVS=24
MAX_CONCURRENT=3
GPU_LIST=(0 1 2 3)

declare -a CMDS
for algo in "${ALGOS[@]}"; do
  for hybrid in "${HYBRIDS[@]}"; do
    for training in "${TRAININGS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        if [ "$training" == "independent" ]; then
          if [ -n "$hybrid" ]; then
            cmd="./run_independent_dense.sh $algo $hybrid $seed $NUM_ENVS"
          else
            cmd="./run_independent_dense.sh $algo $seed $NUM_ENVS"
          fi
        else
          if [ -n "$hybrid" ]; then
            cmd="./run_curriculum.sh $algo $hybrid $seed $NUM_ENVS"
          else
            cmd="./run_curriculum.sh $algo $seed $NUM_ENVS"
          fi
        fi
        CMDS+=("$cmd")
      done
    done
  done
done

running() {
  pgrep -f "python train.py" | wc -l
}

gpu_idx=0
for cmd in "${CMDS[@]}"; do
  while [ $(running) -ge $MAX_CONCURRENT ]; do
    sleep 10
  done
  gpu=${GPU_LIST[$gpu_idx]}
  gpu_idx=$(( (gpu_idx + 1) % ${#GPU_LIST[@]} ))
  echo "Launching on GPU $gpu: $cmd"
  CUDA_VISIBLE_DEVICES=$gpu $cmd &
  sleep 2
done

wait
echo "All training finished!"