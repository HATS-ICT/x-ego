#!/bin/bash
# Sequential CLIP pipeline: for each map, run contrastive training then
# downstream baseline + finetuned for seeds 1 and 2.
# Intended to be run locally on a Linux machine after the server-side CLIP runs failed.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

set -a
source .env
set +a

: "${OUTPUT_BASE_PATH:?OUTPUT_BASE_PATH not set in .env}"

MAPS=(mirage dust2 inferno)
SEEDS=(1 2)

noti -m "Starting CLIP pipeline over maps: ${MAPS[*]} (seeds ${SEEDS[*]})"

run_contrastive() {
    local map_name="$1"
    local run_name="main_contra_with_accu-clip-${map_name}-ui-all"
    echo "=== Contrastive clip/${map_name} ==="
    noti -m "Starting clip contrastive training on ${map_name}"

    uv run python main.py --mode train --task contrastive \
        model.encoder.model_type=clip \
        data.map="${map_name}" \
        data.ui_mask=all \
        data.batch_size=128 \
        data.num_workers=8 \
        training.max_epochs=40 \
        training.accumulate_grad_batches=1 \
        training.contrastive_accumulate_batches=8 \
        training.torch_compile=false \
        meta.exp_name=main_contra_with_accu \
        meta.run_name="${run_name}"

    noti -m "Finished clip contrastive training on ${map_name}"
}

locate_stage1_ckpt() {
    local map_name="$1"
    local ckpt_dir
    ckpt_dir=$(ls -td "${OUTPUT_BASE_PATH}"/main_contra_with_accu-clip-"${map_name}"-ui-all-* 2>/dev/null | head -n 1)
    if [[ -z "${ckpt_dir}" ]]; then
        echo "ERROR: could not locate stage1 checkpoint for clip/${map_name} under ${OUTPUT_BASE_PATH}" >&2
        noti -m "CLIP pipeline aborted: missing clip/${map_name} stage1 checkpoint"
        exit 1
    fi
    echo "$(basename "${ckpt_dir}")/checkpoint/last.ckpt"
}

run_downstream_baseline() {
    local map_name="$1"
    local seed="$2"
    echo "=== Downstream baseline seed=${seed} clip/${map_name} ==="
    noti -m "Starting clip downstream baseline seed=${seed} on ${map_name}"
    uv run python train_all_downstream.py \
        --map "${map_name}" \
        --model-type clip \
        --ui-mask all \
        --extra-overrides model.encoder.trainable=false "meta.seed=${seed}"
    noti -m "Finished clip downstream baseline seed=${seed} on ${map_name}"
}

run_downstream_finetuned() {
    local map_name="$1"
    local seed="$2"
    local stage1_ckpt="$3"
    echo "=== Downstream finetuned seed=${seed} clip/${map_name} ==="
    noti -m "Starting clip downstream finetuned seed=${seed} on ${map_name}"
    uv run python train_all_downstream.py \
        --map "${map_name}" \
        --model-type clip \
        --ui-mask all \
        --stage1-checkpoint "${stage1_ckpt}" \
        --extra-overrides model.encoder.trainable=false "meta.seed=${seed}"
    noti -m "Finished clip downstream finetuned seed=${seed} on ${map_name}"
}

for map_name in "${MAPS[@]}"; do
    echo "############################################################"
    echo "# Map: ${map_name}"
    echo "############################################################"
    noti -m "=== Starting CLIP pipeline for map ${map_name} ==="

    run_contrastive "${map_name}"

    stage1_ckpt=$(locate_stage1_ckpt "${map_name}")
    echo "Using stage1 checkpoint for ${map_name}: ${stage1_ckpt}"
    noti -m "Using stage1 checkpoint ${stage1_ckpt} for ${map_name}"

    for seed in "${SEEDS[@]}"; do
        run_downstream_baseline "${map_name}" "${seed}"
        run_downstream_finetuned "${map_name}" "${seed}" "${stage1_ckpt}"
    done

    noti -m "=== Finished CLIP pipeline for map ${map_name} ==="
done

noti -m "CLIP pipeline complete for maps: ${MAPS[*]}"
echo "All CLIP pipeline stages finished."
