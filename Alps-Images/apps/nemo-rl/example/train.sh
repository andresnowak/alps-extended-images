#!/usr/bin/env bash
set -Eeuo pipefail
# Keep crash dumps out of the submission directory.
ulimit -c 0

usage() {
    printf '%s\n' \
        'Usage: bash train.sh {megatron|vllm|nemo-gym-vllm} [sbatch options]' \
        'Example: bash train.sh vllm --account=YOUR_PROJECT --time=06:00:00' \
        'Use this command rather than direct sbatch: recipe resources are selected before submission.' \
        'Shared user settings: .env beside this launcher. Recipe defaults: <recipe>/recipe.env.' \
        'Overrides: NEMO_RL_ENVIRONMENT, HF_MODEL, MAX_NUM_STEPS, TRAIN_MICRO_BATCH,' \
        'INFERENCE_REQUESTS, NUM_PROMPTS, NRL_MEGATRON_CHECKPOINT_DIR, CHONK_RUN_ROOT.'
}
if [[ $# == 0 || "${1:-}" == -h || "${1:-}" == --help ]]; then
    usage
    exit 0
fi
recipe="$1"
shift
case "$recipe" in megatron|vllm|nemo-gym-vllm) ;; *) echo "Unknown recipe: $recipe" >&2; usage >&2; exit 2 ;; esac
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
mode=submit
case "${1:-}" in
    --batch|--server|--driver) mode="${1#--}"; shift ;;
esac
if [[ "$mode" == submit ]]; then
    [[ -z "${SLURM_JOB_ID:-}" ]] || {
        echo 'Use bash train.sh from outside an allocation, not direct sbatch.' >&2
        exit 2
    }
    LAUNCHER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
else
    [[ $# == 0 ]] || { echo 'Unexpected arguments in internal job mode.' >&2; exit 2; }
    : "${SLURM_JOB_ID:?Internal job modes require Slurm}"
    # Slurm executes a spooled script, so its BASH_SOURCE is not the repository path.
    : "${CHONK_EXAMPLES_DIR:?Submit with bash train.sh RECIPE}"
    LAUNCHER_DIR="$CHONK_EXAMPLES_DIR"
fi
LAUNCHER="$LAUNCHER_DIR/train.sh"

# Shared .env overrides include W&B project/entity/group and credentials.
# Load at submission/batch startup, not again in Ray child modes.
if [[ "$mode" == submit || "$mode" == batch ]] && [[ -f "$LAUNCHER_DIR/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$LAUNCHER_DIR/.env"
    set +a
fi
# Public recipe values select resources, batch defaults and the source-PR stack.
# shellcheck source=/dev/null
source "$LAUNCHER_DIR/$recipe/recipe.env"
TOTAL_NODES=$((TRAIN_NODES + INFERENCE_NODES))
TOTAL_GPUS=$((TOTAL_NODES * 4))
EXAMPLE_DIR="${EXAMPLE_DIR:-$LAUNCHER_DIR/$recipe}"
EXAMPLE_DIR="$(cd -- "$EXAMPLE_DIR" && pwd)"

# User overrides: source repositories, model, container and run directory.
NEMO_RL_SOURCE="${NEMO_RL_SOURCE:-https://github.com/swiss-ai/Nemo-RL.git}"
BRIDGE_SOURCE="${BRIDGE_SOURCE:-https://github.com/swiss-ai/Megatron-Bridge.git}"
MEGATRON_SOURCE="${MEGATRON_SOURCE:-https://github.com/swiss-ai/Megatron-LM-MoE.git}"
HF_MODEL="${HF_MODEL:-/capstor/store/cscs/swissai/infra01/apertus_checkpoints/v2/hf/chonk/pretraining/long_context/long-context-iter-0003576}"
ENV_FILE="${NEMO_RL_ENVIRONMENT:-$LAUNCHER_DIR/environment.toml}"
RUN_ROOT="${CHONK_RUN_ROOT:-/iopsstor/scratch/cscs/$USER/tmp/$RUN_NAME}"
if [[ "$GYM_ENABLED" == 1 ]]; then
    NEMO_GYM_SOURCE="${NEMO_GYM_SOURCE:-https://github.com/swiss-ai/nemo-gym.git}"
    NEMO_GYM_REF="${NEMO_GYM_REF:-93f232f02074deda2fec13a6848f22e5dca741f5}"
    export NEMO_GYM_SOURCE NEMO_GYM_REF
fi

# Shared source pins; the profile lists the recipe-specific NeMo PRs.
NEMO_REF=f196ef0a1ccb54d6ce71cd63277fb6a0e01168db
NEMO_FP8_REF=19843e5a523254ebf3bed49a47a1e852dca29a70
MCORE_REF=0536034901962160aac9742d134a84c55e28d224
BRIDGE_REF=077224531ad4d6d0caef767a84e619b256aa5cb6

# Batch/step overrides. GSM8K_VAL_PERIOD and GYM_VAL_PERIOD are resolved by YAML.
TRAIN_MICRO_BATCH="${TRAIN_MICRO_BATCH:-$DEFAULT_MICRO_BATCH}"
INFERENCE_REQUESTS="${INFERENCE_REQUESTS:-32}"
MAX_NUM_STEPS="${MAX_NUM_STEPS:-$DEFAULT_MAX_NUM_STEPS}"
NUM_PROMPTS="${NUM_PROMPTS:-$DEFAULT_NUM_PROMPTS}"
(( TRAIN_MICRO_BATCH > 0 && INFERENCE_REQUESTS > 0 && NUM_PROMPTS > 0 && MAX_NUM_STEPS > 0 )) || {
    echo 'Batch settings and steps must be positive.' >&2; exit 2;
}
# TP1/PP4/CP1 on four GPUs per node gives TRAIN_NODES data-parallel ranks.
(( TRAIN_GLOBAL_BATCH % (TRAIN_NODES * TRAIN_MICRO_BATCH) == 0 )) || {
    echo "Global batch $TRAIN_GLOBAL_BATCH must divide into complete training microbatches." >&2; exit 2;
}
(( (NUM_PROMPTS * GENERATIONS_PER_PROMPT) % TRAIN_GLOBAL_BATCH == 0 )) || {
    echo 'Rollout batch must divide into training batches.' >&2; exit 2;
}
[[ -f "$ENV_FILE" ]] || { echo "EDF not found: $ENV_FILE" >&2; exit 2; }
[[ -f "$EXAMPLE_DIR/$CONFIG_FILE" ]] || { echo "Config not found: $EXAMPLE_DIR/$CONFIG_FILE" >&2; exit 2; }
case "$RUN_ROOT/" in "$HOME/"*) echo 'RUN_ROOT must not be under HOME.' >&2; exit 2 ;; esac

export CHONK_EXAMPLES_DIR="$LAUNCHER_DIR" EXAMPLE_DIR HF_MODEL
export NEMO_RL_SOURCE BRIDGE_SOURCE MEGATRON_SOURCE
export NEMO_RL_ENVIRONMENT="$ENV_FILE" CHONK_RUN_ROOT="$RUN_ROOT"
export TRAIN_NODES INFERENCE_NODES TOTAL_NODES TRAIN_GLOBAL_BATCH GENERATIONS_PER_PROMPT
export TRAIN_MICRO_BATCH INFERENCE_REQUESTS MAX_NUM_STEPS NUM_PROMPTS GYM_ENABLED

if [[ "$mode" == submit ]]; then
    command -v sbatch >/dev/null || { echo 'sbatch is not available.' >&2; exit 127; }
    sources=("$NEMO_RL_SOURCE" "$BRIDGE_SOURCE" "$MEGATRON_SOURCE")
    if [[ "$GYM_ENABLED" == 1 ]]; then sources+=("$NEMO_GYM_SOURCE"); fi
    for source in "${sources[@]}"; do git ls-remote "$source" HEAD >/dev/null || exit 2; done
    mkdir -p "$RUN_ROOT" "/iopsstor/scratch/cscs/$USER/tmp"
    # Command-line flags work here; #SBATCH directives cannot expand recipe values.
    exec sbatch \
        --nodes="$TOTAL_NODES" --ntasks-per-node=1 --cpus-per-task=288 --gres=gpu:4 \
        --exclusive --account=csstaff --partition=normal --time="$JOB_TIME" \
        --job-name="$JOB_NAME" --output="/iopsstor/scratch/cscs/$USER/tmp/$JOB_NAME-%j.out" \
        --export=ALL "$@" "$LAUNCHER" "$recipe" --batch
fi

: "${SCRATCH:?SCRATCH must point to scratch storage}"
[[ "${SLURM_JOB_NUM_NODES:-$TOTAL_NODES}" == "$TOTAL_NODES" ]] || {
    echo "Requires exactly $TOTAL_NODES nodes." >&2; exit 2;
}
RUN_DIR="$RUN_ROOT/job-${SLURM_JOB_ID}"
MEGATRON_PATH="/tmp/chonk-mcore-$SLURM_JOB_ID"
if [[ "$mode" == batch ]]; then
    mkdir -p "$RUN_DIR"
    [[ ! -e "$RUN_DIR/sources" ]] || { echo "Refusing to overwrite snapshots: $RUN_DIR/sources" >&2; exit 1; }
    mkdir -p "$RUN_DIR/sources/megatron"
    git clone --no-checkout "$NEMO_RL_SOURCE" "$RUN_DIR/sources/nemo-rl"
    git -C "$RUN_DIR/sources/nemo-rl" checkout --detach "$NEMO_REF"
    for pr in "${NEMO_PRS[@]}"; do
        git -C "$RUN_DIR/sources/nemo-rl" fetch https://github.com/swiss-ai/Nemo-RL.git "pull/$pr/head"
        printf 'NeMo PR #%s ' "$pr" >> "$RUN_DIR/applied-prs.txt"
        git -C "$RUN_DIR/sources/nemo-rl" rev-parse FETCH_HEAD >> "$RUN_DIR/applied-prs.txt"
        git -C "$RUN_DIR/sources/nemo-rl" -c user.name='Alps example' \
            -c user.email=alps@example.invalid -c commit.gpgsign=false merge --no-edit FETCH_HEAD
    done
    if [[ "$GENERATION_BACKEND" == vllm ]]; then
        git -C "$RUN_DIR/sources/nemo-rl" fetch https://github.com/NVIDIA-NeMo/RL.git pull/2517/head
        printf 'NeMo upstream PR #2517 ' >> "$RUN_DIR/applied-prs.txt"
        git -C "$RUN_DIR/sources/nemo-rl" rev-parse FETCH_HEAD >> "$RUN_DIR/applied-prs.txt"
        # Keep the PR's EP-only multiprocessing executor and the fork's nightly-test budget.
        git -C "$RUN_DIR/sources/nemo-rl" -c user.name='Alps example' \
            -c user.email=alps@example.invalid -c commit.gpgsign=false merge --no-commit --no-ff -Xtheirs FETCH_HEAD
        git -C "$RUN_DIR/sources/nemo-rl" restore --source=HEAD -- tests/unit/test_recipes_and_test_suites.py
        git -C "$RUN_DIR/sources/nemo-rl" add tests/unit/test_recipes_and_test_suites.py
        git -C "$RUN_DIR/sources/nemo-rl" -c user.name='Alps example' \
            -c user.email=alps@example.invalid -c commit.gpgsign=false \
            commit -m 'Merge upstream vLLM DP PR 2517' \
            -m 'Use the PR multiprocessing executor and retain the fork nightly-test budget.'
    fi
    # PR #8 refreshes FP8 expert caches after optimizer and reference/policy updates.
    git -C "$RUN_DIR/sources/nemo-rl" fetch https://github.com/swiss-ai/Nemo-RL.git pull/8/head
    git -C "$RUN_DIR/sources/nemo-rl" cat-file -e "$NEMO_FP8_REF^{commit}"
    git -C "$RUN_DIR/sources/nemo-rl" -c user.name='Alps example' \
        -c user.email=alps@example.invalid -c commit.gpgsign=false merge --no-edit "$NEMO_FP8_REF"
    printf 'NeMo PR #8 %s\n' "$NEMO_FP8_REF" >> "$RUN_DIR/applied-prs.txt"

    if [[ "$GYM_ENABLED" == 1 ]]; then
        NEMO_GYM_PATH="$RUN_DIR/sources/nemo-rl/3rdparty/Gym-workspace/Gym"
        git clone --no-checkout "$NEMO_GYM_SOURCE" "$NEMO_GYM_PATH"
        git -C "$NEMO_GYM_PATH" checkout --detach "$NEMO_GYM_REF"
        printf 'NeMo Gym %s\n' "$NEMO_GYM_REF" >> "$RUN_DIR/applied-prs.txt"
        export NEMO_GYM_PATH NEMO_GYM_VENV_DIR="$RUN_DIR/gym-venvs"
    fi
    git clone --no-checkout "$MEGATRON_SOURCE" "$RUN_DIR/megatron-repo"
    git -C "$RUN_DIR/megatron-repo" fetch https://github.com/swiss-ai/Megatron-LM-MoE.git pull/88/head
    printf 'Megatron PR #88 ' >> "$RUN_DIR/applied-prs.txt"
    git -C "$RUN_DIR/megatron-repo" rev-parse FETCH_HEAD >> "$RUN_DIR/applied-prs.txt"
    git -C "$RUN_DIR/megatron-repo" cat-file -e "$MCORE_REF^{commit}"
    git -C "$RUN_DIR/megatron-repo" archive "$MCORE_REF" |
        tar --extract --touch --directory="$RUN_DIR/sources/megatron"
    git clone --no-checkout "$BRIDGE_SOURCE" "$RUN_DIR/sources/bridge"
    git -C "$RUN_DIR/sources/bridge" checkout --detach "$BRIDGE_REF"
    git -C "$RUN_DIR/sources/bridge" fetch https://github.com/swiss-ai/Megatron-Bridge.git pull/5/head
    git -C "$RUN_DIR/sources/bridge" merge --ff-only FETCH_HEAD
    {
        printf 'NeMo-RL %s\nMegatron-Core %s\nBridge base %s\n' "$NEMO_REF" "$MCORE_REF" "$BRIDGE_REF"
        cat "$RUN_DIR/applied-prs.txt"
        printf 'NeMo result tree '
        git -C "$RUN_DIR/sources/nemo-rl" rev-parse 'HEAD^{tree}'
        printf 'Bridge PR #5 '
        git -C "$RUN_DIR/sources/bridge" rev-parse HEAD
    } > "$RUN_DIR/source-provenance.txt"

    NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-$RUN_DIR/hf-import}"
    export RUN_DIR NRL_MEGATRON_CHECKPOINT_DIR MEGATRON_PATH
    export NEMO_RL_PATH="$RUN_DIR/sources/nemo-rl" MEGATRON_BRIDGE_PATH="$RUN_DIR/sources/bridge/src"
    # W&B job-local paths. The shared artifact cache is set in environment.toml.
    export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$RUN_DIR/wandb/config}"
    export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$RUN_DIR/wandb/data}"
    export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$RUN_DIR/wandb/artifacts}"
    if [[ "$GYM_ENABLED" == 1 ]]; then
        export GYM_MATH_URL="${GYM_MATH_URL:-https://gym-dev.swissai.svc.cscs.ch/math_with_judge}"
        export GYM_TRAIN_DATA="${GYM_TRAIN_DATA:-$RUN_DIR/data/openmathreasoning-train.jsonl}"
        export GYM_VAL_DATA="${GYM_VAL_DATA:-$RUN_DIR/data/openmathreasoning-validation.jsonl}"
        GYM_CHAT_TEMPLATE="$(< "$EXAMPLE_DIR/math-chat-template.jinja")"
        export GYM_CHAT_TEMPLATE
    fi
    export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
    export RAY_USAGE_STATS_ENABLED=0 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 PMIX_MCA_psec=native
    if [[ "$GENERATION_BACKEND" == vllm ]]; then export VLLM_WORKER_MULTIPROC_METHOD=spawn; fi
    export NRL_REFIT_BUFFER_MEMORY_RATIO=0.005 NRL_REFIT_NUM_BUFFERS=1
fi

if [[ "$mode" == server || "$mode" == driver ]]; then
    unset BASH_ENV SSH_CONNECTION SSH_CLIENT SSH_TTY
    cache_root="/tmp/chonk-caches-$SLURM_JOB_ID"
    export CUDA_CACHE_PATH="$cache_root/cuda" TRITON_CACHE_DIR="$cache_root/triton"
    export TRITON_HOME="$cache_root/triton-home" TORCHINDUCTOR_CACHE_DIR="$cache_root/inductor"
    export TORCH_EXTENSIONS_DIR="$cache_root/extensions" FLASHINFER_WORKSPACE_BASE="$cache_root/flashinfer"
    export DG_JIT_CACHE_DIR="$cache_root/deep-gemm"
    export MEGATRON_CONFIG_LOCK_DIR="${MEGATRON_CONFIG_LOCK_DIR:-$cache_root/megatron-config-locks}"
    mkdir -p "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
        "$TORCH_EXTENSIONS_DIR" "$FLASHINFER_WORKSPACE_BASE" "$MEGATRON_CONFIG_LOCK_DIR"
    if [[ ! -f "$MEGATRON_PATH/megatron/core/__init__.py" ]]; then
        mkdir -p "$MEGATRON_PATH"
        cp -R --no-preserve=mode,ownership "$RUN_DIR/sources/megatron/." "$MEGATRON_PATH/"
    fi
    export PYTHONPATH="$RUN_DIR/dependencies:$NEMO_RL_PATH:$MEGATRON_BRIDGE_PATH:$MEGATRON_PATH:$EXAMPLE_DIR"
    if [[ "$GYM_ENABLED" == 1 ]]; then
        export PYTHONPATH="$RUN_DIR/dependencies:$NEMO_RL_PATH:$MEGATRON_BRIDGE_PATH:$MEGATRON_PATH:$NEMO_GYM_PATH:$EXAMPLE_DIR"
    fi
fi

# One server per node. Rank zero starts the driver in the head's container.
if [[ "$mode" == server ]]; then
    node_ip="$(hostname -I | awk '{print $1}')"
    [[ -n "$node_ip" ]] || { echo 'Cannot determine node IP.' >&2; exit 1; }
    args=(start --node-ip-address="$node_ip" --num-cpus="${SLURM_CPUS_PER_TASK:-288}" --num-gpus=4 --disable-usage-stats)
    if [[ "$SLURM_PROCID" == 0 ]]; then
        if [[ "$GYM_ENABLED" == 1 ]]; then
            uv run --no-project --python /usr/bin/python3.12 python -c \
                'from nemo_gym.cli import RunHelper; from nemo_gym.rollout_collection import RolloutCollectionHelper; import responses_api_agents.simple_agent.app; import responses_api_models.vllm_model.app; print("GYM_CLIENT_PREFLIGHT_PASS")'
        fi
        args+=(--head --port="${RAY_PORT:-6379}" --include-dashboard=false --temp-dir="/tmp/chonk-ray-$SLURM_JOB_ID")
        printf '%s:%s\n' "$node_ip" "${RAY_PORT:-6379}" > "$RUN_DIR/ray-address.txt.tmp"
        mv "$RUN_DIR/ray-address.txt.tmp" "$RUN_DIR/ray-address.txt"
    else
        deadline=$((SECONDS + 900))
        until [[ -s "$RUN_DIR/ray-address.txt" ]]; do
            (( SECONDS < deadline )) || { echo 'Timed out waiting for Ray head address.' >&2; exit 1; }
            sleep 2
        done
        read -r RAY_ADDRESS < "$RUN_DIR/ray-address.txt"
        args+=(--address="$RAY_ADDRESS")
    fi
    ray "${args[@]}"
    if [[ "$SLURM_PROCID" == 0 ]]; then
        if [[ "$GYM_ENABLED" == 1 ]]; then
            curl --fail --silent --show-error --max-time 15 "$GYM_MATH_URL/" > "$RUN_DIR/gym-health.json"
        fi
        set +e
        bash "$LAUNCHER" "$recipe" --driver
        driver_status=$?
        set -e
        printf '%s\n' "$driver_status" > "$RUN_DIR/driver-exit.txt.tmp"
        mv "$RUN_DIR/driver-exit.txt.tmp" "$RUN_DIR/driver-exit.txt"
    else
        until [[ -s "$RUN_DIR/driver-exit.txt" ]]; do sleep 2; done
        read -r driver_status < "$RUN_DIR/driver-exit.txt"
    fi
    exit "$driver_status"
fi

# Run GRPO beside the head's Raylet, after the full resource set is ready.
if [[ "$mode" == driver ]]; then
    deadline=$((SECONDS + 900))
    ready=0
    while (( SECONDS < deadline )); do
        if [[ -s "$RUN_DIR/ray-address.txt" ]]; then
            read -r RAY_ADDRESS < "$RUN_DIR/ray-address.txt"
            if status="$(timeout 15 ray status --address="$RAY_ADDRESS" 2>&1)"; then
                nodes="$(awk '/^Active:/ {active=1; next} /^Pending:/ {active=0} active && $1 ~ /^[0-9]+$/ && $2 ~ /^node_/ {n += $1} END {print n+0}' <<< "$status")"
                gpus="$(awk '$2 == "GPU" {split($1, a, "/"); print int(a[2]); exit}' <<< "$status")"
                if [[ "$nodes" == "$TOTAL_NODES" && "${gpus:-0}" == "$TOTAL_GPUS" ]]; then ready=1; break; fi
                echo "Waiting for Ray: $nodes/$TOTAL_NODES nodes, ${gpus:-0}/$TOTAL_GPUS GPUs"
            fi
        fi
        sleep 5
    done
    (( ready )) || { echo "Timed out waiting for all $TOTAL_NODES Ray nodes and $TOTAL_GPUS GPUs." >&2; exit 1; }
    export RAY_ADDRESS
    exec uv run --no-project --python /usr/bin/python3.12 python \
        "$NEMO_RL_PATH/$ENTRYPOINT" --config "$EXAMPLE_DIR/$CONFIG_FILE"
fi

# Install only job-local additions; preserve the image's Torch stack.
srun --nodes=1 --ntasks=1 --relative=0 --cpus-per-task=1 --gres=none \
    --mpi=pmix --network=disable_rdzv_get --export=ALL --environment="$ENV_FILE" \
    -u bash -c '
    set -euo pipefail
    extra_args=()
    if [[ "$GYM_ENABLED" == 1 ]]; then extra_args+=(--requirement "$EXAMPLE_DIR/gym-client-requirements.txt"); fi
    uv pip install --python /usr/bin/python3.12 --no-deps --target "$RUN_DIR/dependencies" "nemo-lens[sdk] @ git+https://github.com/NVIDIA-NeMo/Lens.git@b85578fc2b736a1804705e537001b5f45e9c715d" soundfile==0.13.1 "${extra_args[@]}"
    if [[ "$GYM_ENABLED" == 1 ]]; then
        if [[ "$GYM_TRAIN_DATA" == "$RUN_DIR/data/openmathreasoning-train.jsonl" ]]; then
            bash "$EXAMPLE_DIR/prepare-data.sh" train "$GYM_TRAIN_DATA"
        fi
        if [[ "$GYM_VAL_DATA" == "$RUN_DIR/data/openmathreasoning-validation.jsonl" ]]; then
            bash "$EXAMPLE_DIR/prepare-data.sh" validation "$GYM_VAL_DATA"
        fi
        [[ -f "$GYM_TRAIN_DATA" ]] || { echo "Dataset not found: $GYM_TRAIN_DATA" >&2; exit 2; }
        [[ -f "$GYM_VAL_DATA" ]] || { echo "Dataset not found: $GYM_VAL_DATA" >&2; exit 2; }
    fi'

srun --nodes="$TOTAL_NODES" --ntasks="$TOTAL_NODES" --ntasks-per-node=1 \
    --cpus-per-task=288 --gres=gpu:4 --mpi=pmix --network=disable_rdzv_get \
    --export=ALL --environment="$ENV_FILE" --container-mounts="$HF_MODEL:$HF_MODEL:ro" \
    --kill-on-bad-exit=1 -u bash "$LAUNCHER" "$recipe" --server
