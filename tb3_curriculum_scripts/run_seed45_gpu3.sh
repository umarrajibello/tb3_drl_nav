#!/usr/bin/env bash
set -euo pipefail

GPU_ID="3"
SEED="45"
PYTHON="python"
TRAIN="train.py"
WANDB_PROJECT="tb3-nav-research"
NUM_ENVS=1
EVAL_FREQ=10000

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
mkdir -p shell_logs

echo "============================================================"
echo "TB3 thesis curriculum campaign"
echo "Seed: ${SEED} | Physical GPU: ${GPU_ID}"
echo "MC0–MC3 | C1–C7"
echo "============================================================"

run_stage() {
    local mc="$1"
    local label="$2"
    local hybrid="$3"
    local lookahead="$4"
    local env="$5"
    local gmin="$6"
    local gmax="$7"
    local steps="$8"
    local pretrained="${9:-}"

    local suffix="${mc}_${label}"
    local logfile="shell_logs/seed${SEED}_${suffix}.log"

    echo ""
    echo "------------------------------------------------------------"
    echo "START ${suffix}: ${env}, goal ${gmin}-${gmax} m, ${steps} steps"
    if [[ -n "${pretrained}" ]]; then
        echo "PRETRAINED: ${pretrained}"
    else
        echo "PRETRAINED: none (fresh initialization)"
    fi
    echo "LOG: ${logfile}"
    echo "------------------------------------------------------------"

    local args=(
        --algorithm tx-sac
        --total_timesteps "${steps}"
        --eval_freq "${EVAL_FREQ}"
        --seed "${SEED}"
        --num_envs "${NUM_ENVS}"
        --gpu true
        --env_type "${env}"
        --hybrid "${hybrid}"
        --goal_dist_min "${gmin}"
        --goal_dist_max "${gmax}"
        --wandb_project "${WANDB_PROJECT}"
        --wandb_mode online
        --wandb_run_name "MAIN_${mc}_${label}_seed${SEED}_${env}"
        --wandb_tags "main" "curriculum" "${mc}" "${label}" "seed${SEED}" "${env}"
        --exp_suffix "${suffix}"
    )

    if [[ "${hybrid}" == "true" ]]; then
        args+=(--lookahead_mode "${lookahead}")
    fi

    if [[ -n "${pretrained}" ]]; then
        args+=(--pretrained "${pretrained}")
    fi

    "${PYTHON}" -u "${TRAIN}" "${args[@]}" 2>&1 | tee "${logfile}"

    echo "FINISHED ${suffix}"
}

run_stage "MC0" "C1" "false" "" "empty" "0.8" "1.5" "150000"
prev="./models/empty_tx-sac_end2end_seed45_MC0_C1_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

run_stage "MC0" "C2" "false" "" "static" "0.8" "1.5" "150000" "${prev}"
prev="./models/static_tx-sac_end2end_seed45_MC0_C2_final/model.zip" 
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

run_stage "MC0" "C3" "false" "" "static" "1.5" "2.5" "200000" "${prev}" 
prev="./models/static_tx-sac_end2end_seed45_MC0_C3_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

run_stage "MC0" "C4" "false" "" "mixed" "1.5" "2.5" "200000" "${prev}"
prev="./models/mixed_tx-sac_end2end_seed45_MC0_C4_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

run_stage "MC0" "C5" "false" "" "mixed" "2.5" "4.0" "250000" "${prev}"
prev="./models/mixed_tx-sac_end2end_seed45_MC0_C5_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

run_stage "MC0" "C6" "false" "" "dense" "2.5" "4.0" "300000" "${prev}"
prev="./models/dense_tx-sac_end2end_seed45_MC0_C6_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

run_stage "MC0" "C7" "false" "" "static" "2.5" "4.0" "500000" "${prev}"
prev="./models/static_tx-sac_end2end_seed45_MC0_C7_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

echo "Completed MC0 for seed ${SEED}."

run_stage "MC1" "C1" "true" "fixed" "empty" "0.8" "1.5" "150000"
prev="./models/empty_tx-sac_hybrid_la-fixed_seed45_MC1_C1_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

run_stage "MC1" "C2" "true" "fixed" "static" "0.8" "1.5" "150000" "${prev}"
prev="./models/static_tx-sac_hybrid_la-fixed_seed45_MC1_C2_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

run_stage "MC1" "C3" "true" "fixed" "static" "1.5" "2.5" "200000" "${prev}"
prev="./models/static_tx-sac_hybrid_la-fixed_seed45_MC1_C3_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

run_stage "MC1" "C4" "true" "fixed" "mixed" "1.5" "2.5" "200000" "${prev}"
prev="./models/mixed_tx-sac_hybrid_la-fixed_seed45_MC1_C4_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

run_stage "MC1" "C5" "true" "fixed" "mixed" "2.5" "4.0" "250000" "${prev}"
prev="./models/mixed_tx-sac_hybrid_la-fixed_seed45_MC1_C5_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

run_stage "MC1" "C6" "true" "fixed" "dense" "2.5" "4.0" "300000" "${prev}"
prev="./models/dense_tx-sac_hybrid_la-fixed_seed45_MC1_C6_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

run_stage "MC1" "C7" "true" "fixed" "static" "2.5" "4.0" "500000" "${prev}"
prev="./models/static_tx-sac_hybrid_la-fixed_seed45_MC1_C7_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

echo "Completed MC1 for seed ${SEED}."

run_stage "MC2" "C1" "true" "velocity" "empty" "0.8" "1.5" "150000"
prev="./models/empty_tx-sac_hybrid_la-velocity_seed45_MC2_C1_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

run_stage "MC2" "C2" "true" "velocity" "static" "0.8" "1.5" "150000" "${prev}"
prev="./models/static_tx-sac_hybrid_la-velocity_seed45_MC2_C2_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

run_stage "MC2" "C3" "true" "velocity" "static" "1.5" "2.5" "200000" "${prev}"
prev="./models/static_tx-sac_hybrid_la-velocity_seed45_MC2_C3_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

run_stage "MC2" "C4" "true" "velocity" "mixed" "1.5" "2.5" "200000" "${prev}"
prev="./models/mixed_tx-sac_hybrid_la-velocity_seed45_MC2_C4_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

run_stage "MC2" "C5" "true" "velocity" "mixed" "2.5" "4.0" "250000" "${prev}"
prev="./models/mixed_tx-sac_hybrid_la-velocity_seed45_MC2_C5_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

run_stage "MC2" "C6" "true" "velocity" "dense" "2.5" "4.0" "300000" "${prev}"
prev="./models/dense_tx-sac_hybrid_la-velocity_seed45_MC2_C6_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

run_stage "MC2" "C7" "true" "velocity" "static" "2.5" "4.0" "500000" "${prev}"
prev="./models/static_tx-sac_hybrid_la-velocity_seed45_MC2_C7_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

echo "Completed MC2 for seed ${SEED}."

run_stage "MC3" "C1" "true" "velocity_clearance" "empty" "0.8" "1.5" "150000"
prev="./models/empty_tx-sac_hybrid_la-velocity_clearance_seed45_MC3_C1_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

run_stage "MC3" "C2" "true" "velocity_clearance" "static" "0.8" "1.5" "150000" "${prev}"
prev="./models/static_tx-sac_hybrid_la-velocity_clearance_seed45_MC3_C2_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

run_stage "MC3" "C3" "true" "velocity_clearance" "static" "1.5" "2.5" "200000" "${prev}"
prev="./models/static_tx-sac_hybrid_la-velocity_clearance_seed45_MC3_C3_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

run_stage "MC3" "C4" "true" "velocity_clearance" "mixed" "1.5" "2.5" "200000" "${prev}"
prev="./models/mixed_tx-sac_hybrid_la-velocity_clearance_seed45_MC3_C4_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

run_stage "MC3" "C5" "true" "velocity_clearance" "mixed" "2.5" "4.0" "250000" "${prev}"
prev="./models/mixed_tx-sac_hybrid_la-velocity_clearance_seed45_MC3_C5_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

run_stage "MC3" "C6" "true" "velocity_clearance" "dense" "2.5" "4.0" "300000" "${prev}"
prev="./models/dense_tx-sac_hybrid_la-velocity_clearance_seed45_MC3_C6_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

run_stage "MC3" "C7" "true" "velocity_clearance" "static" "2.5" "4.0" "500000" "${prev}"
prev="./models/static_tx-sac_hybrid_la-velocity_clearance_seed45_MC3_C7_final/model.zip"
[[ -f "${prev}" ]] || { echo "ERROR: expected checkpoint not found: ${prev}" >&2; exit 1; }

echo "Completed MC3 for seed ${SEED}."

echo ""
echo "============================================================"
echo "SEED ${SEED} CURRICULUM CAMPAIGN COMPLETE"
echo "Final checkpoints are the MC0–MC3 C7 models under ./models/*_final/model.zip"
echo "============================================================"
