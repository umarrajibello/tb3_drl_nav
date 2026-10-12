#!/usr/bin/env bash
set -euo pipefail

# TB3 THESIS — PRIMARY M0–M3 DYNAMIC/MIXED CAMPAIGN
# GPU 0->seed42, GPU 1->seed43, GPU 2->seed44, GPU 3->seed45
# Within each seed: M0 -> M1 -> M2 -> M3 sequentially.
# All M-conditions use the same mixed environment, goal range,
# training budget, evaluation frequency, and seed-specific settings.
#
# The final budget is not yet scientifically frozen.
# Default = 1,000,000 steps for the convergence/budget pilot.
# Override with e.g.:
#   STEPS=500000 ./run_mseries_mixed_4gpu.sh
#   STEPS=750000 ./run_mseries_mixed_4gpu.sh

GPU_IDS=(0 1 2 3)
SEEDS=(42 43 44 45)

PYTHON="${PYTHON:-python}"
TRAIN="${TRAIN:-train.py}"
WANDB_PROJECT="${WANDB_PROJECT:-tb3-nav-research}"
NUM_ENVS="${NUM_ENVS:-1}"
EVAL_FREQ="${EVAL_FREQ:-10000}"

# Keep these identical for every M-condition and every seed.
ENV_TYPE="${ENV_TYPE:-mixed}"
GOAL_DIST_MIN="${GOAL_DIST_MIN:-1.0}"
GOAL_DIST_MAX="${GOAL_DIST_MAX:-2.0}"
STEPS="${STEPS:-1000000}"

export PYTHONUNBUFFERED=1
mkdir -p shell_logs campaign_logs

echo "============================================================"
echo "TB3 THESIS — M0–M3 MIXED DYNAMIC CAMPAIGN"
echo "============================================================"
echo "Environment:       ${ENV_TYPE}"
echo "Goal distance:     ${GOAL_DIST_MIN}–${GOAL_DIST_MAX} m"
echo "Training steps:    ${STEPS}"
echo "Eval frequency:    ${EVAL_FREQ}"
echo "Num envs:          ${NUM_ENVS}"
echo "Seeds:             ${SEEDS[*]}"
echo "GPUs:              ${GPU_IDS[*]}"
echo "============================================================"
echo "GPU mapping:  GPU0->42  GPU1->43  GPU2->44  GPU3->45"
echo "============================================================"

run_seed() {
    local gpu="$1"
    local seed="$2"
    export CUDA_VISIBLE_DEVICES="${gpu}"

    run_stage() {
        local mc="$1"
        local hybrid="$2"
        local lookahead="$3"
        local suffix="${mc}_MIXED"
        local logfile="shell_logs/seed${seed}_${suffix}.log"

        local args=(
            --algorithm tx-sac
            --total_timesteps "${STEPS}"
            --eval_freq "${EVAL_FREQ}"
            --seed "${seed}"
            --num_envs "${NUM_ENVS}"
            --gpu true
            --env_type "${ENV_TYPE}"
            --hybrid "${hybrid}"
            --goal_dist_min "${GOAL_DIST_MIN}"
            --goal_dist_max "${GOAL_DIST_MAX}"
            --wandb_project "${WANDB_PROJECT}"
            --wandb_mode online
            --wandb_run_name "MAIN_${mc}_mixed_seed${seed}"
            --wandb_tags "main" "mseries" "mixed" "${mc}" "seed${seed}"
            --exp_suffix "${suffix}"
        )

        if [[ "${hybrid}" == "true" ]]; then
            args+=(--lookahead_mode "${lookahead}")
        fi

        echo ""
        echo "------------------------------------------------------------"
        echo "SEED ${seed} | GPU ${gpu} | ${mc}"
        echo "Environment: ${ENV_TYPE}"
        echo "Goal range:  ${GOAL_DIST_MIN}-${GOAL_DIST_MAX} m"
        echo "Steps:       ${STEPS}"
        [[ "${hybrid}" == "true" ]] && echo "Lookahead:   ${lookahead}" || echo "Hybrid:      false"
        echo "LOG:         ${logfile}"
        echo "------------------------------------------------------------"

        "${PYTHON}" -u "${TRAIN}" "${args[@]}" 2>&1 | tee "${logfile}"
        echo "FINISHED ${mc} | seed ${seed}"
    }

    run_stage "M0" "false" ""
    run_stage "M1" "true" "fixed"
    run_stage "M2" "true" "velocity"
    run_stage "M3" "true" "velocity_clearance"

    echo "============================================================"
    echo "SEED ${seed} COMPLETE"
    echo "============================================================"
}

PIDS=()
for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[$i]}"
    gpu="${GPU_IDS[$i]}"
    (
        run_seed "${gpu}" "${seed}"
    ) > "campaign_logs/seed${seed}_gpu${gpu}.stdout.log" 2>&1 &
    pid=$!
    PIDS+=("${pid}")
    echo "Launched seed ${seed} on GPU ${gpu} (PID ${pid})"
done

echo ""
echo "ALL FOUR SEEDS LAUNCHED"
echo "Monitor: tail -f campaign_logs/seed42_gpu0.stdout.log"
echo "         tail -f campaign_logs/seed43_gpu1.stdout.log"
echo "         tail -f campaign_logs/seed44_gpu2.stdout.log"
echo "         tail -f campaign_logs/seed45_gpu3.stdout.log"
echo ""
echo "Or:      tail -f campaign_logs/*.stdout.log"

FAILED=0
for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[$i]}"
    gpu="${GPU_IDS[$i]}"
    pid="${PIDS[$i]}"
    if wait "${pid}"; then
        echo "SUCCESS: seed ${seed} on GPU ${gpu}"
    else
        echo "FAILED:  seed ${seed} on GPU ${gpu}" >&2
        FAILED=1
    fi
done

echo "============================================================"
if [[ "${FAILED}" -eq 0 ]]; then
    echo "M0–M3 MIXED CAMPAIGN COMPLETE"
    echo "All seeds 42–45 completed successfully."
else
    echo "M0–M3 MIXED CAMPAIGN FINISHED WITH FAILURES"
    echo "Inspect campaign_logs/*.stdout.log and shell_logs/*.log"
    exit 1
fi
echo "============================================================"
