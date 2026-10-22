#!/usr/bin/env bash
set -Eeuo pipefail
# Keep crash dumps out of the submission directory.
ulimit -c 0

# ----- CLI and recipe selection -----

die() { printf '%s\n' "$*" >&2; exit 2; }

usage() {
    printf '%s\n' \
        'Usage: bash train.sh RECIPE [--profile NAME] [training options] -- [sbatch options]' \
        'Example: bash train.sh vllm/gsm8k --profile muon --grpo-steps 58 -- --account=YOUR_PROJECT' \
        'RECIPE is a directory relative to this launcher. Select NAME.env with --profile.' \
        'Without --profile, the launcher uses recipe.env.' \
        'Training options: --grpo-steps, --prompts, --generations-per-prompt, --updates-per-rollout,' \
        '  --micro-batch, --train-nodes, --generation-nodes, --inference-requests,' \
        '  --lr, --min-lr, --matrix-lr, --gains-lr, --train-mb-tokens, --logprob-mb-tokens.' \
        'Development overrides: --dev-COMPONENT-checkout PATH, or --dev-COMPONENT-ref REF' \
        '  with optional --dev-COMPONENT-repo URL; COMPONENT is nemo, core, bridge or gym.' \
        '  Repeat --dev-COMPONENT-pr NUMBER_OR_GITHUB_URL, --dev-COMPONENT-merge REF' \
        '  and --dev-COMPONENT-patch FILE as needed (merges first, then patches).' \
        '  Only selected components change; no release patch list is applied automatically.' \
        'Set NEMO_RL_IMAGE to the image containing the consolidated training stack.' \
        'CLI options override the shared .env/environment and profile defaults.' \
        'Use this command rather than direct sbatch; Slurm options must follow --.'
}
if [[ $# == 0 || "${1:-}" == -h || "${1:-}" == --help ]]; then
    usage
    exit 0
fi
recipe="$1"
shift
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
mode=submit
case "${1:-}" in
    --batch|--server|--driver|--prepare-dev) mode="${1#--}"; shift ;;
esac
if [[ "$mode" == submit ]]; then
    [[ -z "${SLURM_JOB_ID:-}" ]] || die 'Use bash train.sh from outside an allocation, not direct sbatch.'
    LAUNCHER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
else
    [[ $# == 0 ]] || die 'Unexpected arguments in internal job mode.'
    : "${SLURM_JOB_ID:?Internal job modes require Slurm}"
    # Slurm executes a spooled script, so its BASH_SOURCE is not the repository path.
    : "${CHONK_EXAMPLES_DIR:?Submit with bash train.sh RECIPE}"
    LAUNCHER_DIR="$CHONK_EXAMPLES_DIR"
fi
LAUNCHER="$LAUNCHER_DIR/train.sh"
declare -A overrides=()
profile="${CHONK_RECIPE_PROFILE:-recipe}"
if [[ "$mode" == submit ]]; then
    profile=recipe
    while (( $# )); do
        case "$1" in
            --) shift; break ;;
            -h|--help) usage; exit 0 ;;
            --*=*) option="${1%%=*}"; value="${1#*=}"; shift ;;
            --*)
                (( $# >= 2 )) || die "Missing value for $1"
                option="$1"; value="$2"; shift 2 ;;
            *) die "Unexpected argument: $1. Put Slurm options after --." ;;
        esac
        if [[ "$option" =~ ^--dev-(nemo|core|bridge|gym)-(checkout|repo|ref|merge|pr|patch)$ ]]; then
            component="${BASH_REMATCH[1]^^}"
            field="${BASH_REMATCH[2]^^}"
            [[ -n "$value" && "$value" != *$'\n'* ]] || die "Invalid value for $option"
            if [[ "$field" == PR ]]; then
                if [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
                    value="refs/pull/$value/head"
                elif [[ "$value" =~ ^https://github.com/([^/]+/[^/]+)/pull/([1-9][0-9]*)/?$ ]]; then
                    value="https://github.com/${BASH_REMATCH[1]}.git#refs/pull/${BASH_REMATCH[2]}/head"
                else
                    die "Invalid PR number or GitHub PR URL: $value"
                fi
            fi
            case "$field" in MERGE|PR) field=MERGES ;; PATCH) field=PATCHES ;; esac
            key="DEV_${component}_${field}"
            if [[ "$field" == MERGES || "$field" == PATCHES ]]; then
                overrides[$key]="${overrides[$key]:+${overrides[$key]}$'\n'}$value"
            else
                overrides[$key]="$value"
            fi
            continue
        fi
        case "$option" in
            --profile) profile="$value"; continue ;;
            --grpo-steps) key=MAX_NUM_STEPS ;;
            --prompts) key=NUM_PROMPTS ;;
            --generations-per-prompt) key=GENERATIONS_PER_PROMPT ;;
            --updates-per-rollout) key=UPDATES_PER_ROLLOUT ;;
            --micro-batch) key=TRAIN_MICRO_BATCH ;;
            --train-nodes) key=TRAIN_NODES ;;
            --generation-nodes) key=INFERENCE_NODES ;;
            --inference-requests) key=INFERENCE_REQUESTS ;;
            --lr) key=BASE_LR ;;
            --min-lr) key=MIN_LR ;;
            --matrix-lr) key=MATRIX_LR ;;
            --gains-lr) key=GAINS_LR ;;
            --train-mb-tokens) key=TRAIN_MB_TOKENS ;;
            --logprob-mb-tokens) key=LOGPROB_MB_TOKENS ;;
            *) die "Unknown training option: $option. Put Slurm options after --." ;;
        esac
        overrides[$key]="$value"
    done
fi

# ----- User settings and recipe defaults -----

# Job processes inherit these settings; do not re-read files after submission.
if [[ "$mode" == submit ]]; then
    [[ "$profile" =~ ^[a-zA-Z0-9_-]+$ ]] || die 'Invalid profile name.'
    recipe_profile="$LAUNCHER_DIR/$recipe/$profile.env"
    [[ -f "$recipe_profile" ]] || die "Profile not found: $recipe/$profile.env"
    if [[ -f "$LAUNCHER_DIR/.env" ]]; then
        set -a
        # shellcheck source=/dev/null
        source "$LAUNCHER_DIR/.env"
        set +a
    fi
    recipe_profile="$(readlink -f -- "$recipe_profile")"
    set -a
    # shellcheck source=/dev/null
    source "$recipe_profile"
    set +a
    for key in "${!overrides[@]}"; do printf -v "$key" '%s' "${overrides[$key]}"; done
    EXAMPLE_DIR="$(cd -- "${EXAMPLE_DIR:-$(dirname -- "$recipe_profile")}" && pwd -P)"
fi

# The older image does not contain the installed source stack.
: "${NEMO_RL_IMAGE:?Set NEMO_RL_IMAGE to the image built with the consolidated training stack}"
HF_MODEL="${HF_MODEL:-/capstor/store/cscs/swissai/infra01/apertus_checkpoints/v2/hf/chonk/pretraining/long_context/long-context-iter-0003576}"
RUN_ROOT="${CHONK_RUN_ROOT:-/iopsstor/scratch/cscs/$USER/tmp/$RUN_NAME}"

# ----- Development source selection -----

DEV_ENABLED=0
for component in NEMO CORE BRIDGE GYM; do
    checkout_key="DEV_${component}_CHECKOUT"
    repo_key="DEV_${component}_REPO"
    ref_key="DEV_${component}_REF"
    patch_key="DEV_${component}_PATCHES"
    [[ -z "${!checkout_key:-}" || ( -z "${!repo_key:-}" && -z "${!ref_key:-}" ) ]] ||
        die "Choose a checkout OR a repository/ref for $component."
    [[ -z "${!repo_key:-}" || -n "${!ref_key:-}" ]] || die "$repo_key requires $ref_key."
    for field in CHECKOUT REPO REF MERGES PATCHES; do
        key="DEV_${component}_${field}"
        [[ -z "${!key:-}" ]] || DEV_ENABLED=1
        export "$key"
    done
    if [[ "$mode" == submit ]]; then
        if [[ -n "${!checkout_key:-}" ]]; then
            [[ -d "${!checkout_key}" ]] || die "Checkout not found: ${!checkout_key}"
            printf -v "$checkout_key" '%s' "$(cd -- "${!checkout_key}" && pwd -P)"
            [[ "$(git -C "${!checkout_key}" rev-parse --show-toplevel)" == "${!checkout_key}" ]] ||
                die "$checkout_key must be a Git checkout root."
        fi
        if [[ -n "${!patch_key:-}" ]]; then
            patches=''
            while IFS= read -r patch; do
                [[ -n "$patch" ]] || continue
                [[ -f "$patch" ]] || die "Patch not found: $patch"
                patch="$(readlink -f -- "$patch")"
                patches+="${patches:+$'\n'}$patch"
            done <<< "${!patch_key}"
            printf -v "$patch_key" '%s' "$patches"
        fi
    fi
done

# ----- Batch sizing and validation -----
TRAIN_MICRO_BATCH="${TRAIN_MICRO_BATCH:-$DEFAULT_MICRO_BATCH}"
INFERENCE_REQUESTS="${INFERENCE_REQUESTS:-32}"
MAX_NUM_STEPS="${MAX_NUM_STEPS:-$DEFAULT_MAX_NUM_STEPS}"
NUM_PROMPTS="${NUM_PROMPTS:-$DEFAULT_NUM_PROMPTS}"
# Check integers before Bash arithmetic; invalid input must fail before submission.
for key in TRAIN_NODES INFERENCE_NODES TRAIN_MICRO_BATCH \
    INFERENCE_REQUESTS NUM_PROMPTS MAX_NUM_STEPS GENERATIONS_PER_PROMPT \
    UPDATES_PER_ROLLOUT TRAIN_EXPERT_PARALLEL_SIZE GENERATION_EXPERT_PARALLEL_SIZE \
    TRAIN_MB_TOKENS LOGPROB_MB_TOKENS; do
    [[ ! -v "$key" ]] || [[ "${!key}" =~ ^[1-9][0-9]*$ ]] || die "$key must be a positive integer."
done
for key in BASE_LR MIN_LR MATRIX_LR GAINS_LR; do
    [[ ! -v "$key" ]] || [[ "${!key}" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] ||
        die "$key must be a non-negative number."
done
if [[ -n "${BASE_LR:-}" && -n "${MIN_LR:-}" ]]; then
    awk -v base="$BASE_LR" -v minimum="$MIN_LR" 'BEGIN { exit(minimum > base) }' ||
        die 'Minimum learning rate must not exceed the base learning rate.'
fi
TOTAL_NODES=$((TRAIN_NODES + INFERENCE_NODES))
TOTAL_GPUS=$((TOTAL_NODES * 4))
if [[ -n "${UPDATES_PER_ROLLOUT:-}" ]]; then
    (( (NUM_PROMPTS * GENERATIONS_PER_PROMPT) % UPDATES_PER_ROLLOUT == 0 )) ||
        die 'Rollout batch must divide into the requested optimizer updates.'
    TRAIN_GLOBAL_BATCH=$((NUM_PROMPTS * GENERATIONS_PER_PROMPT / UPDATES_PER_ROLLOUT))
fi
[[ "${TRAIN_GLOBAL_BATCH:-}" =~ ^[1-9][0-9]*$ ]] || die 'Derived global batch must be positive.'
(( (NUM_PROMPTS * GENERATIONS_PER_PROMPT) % TRAIN_GLOBAL_BATCH == 0 )) ||
    die 'Rollout batch must divide into training batches.'
UPDATES_PER_ROLLOUT=$((NUM_PROMPTS * GENERATIONS_PER_PROMPT / TRAIN_GLOBAL_BATCH))
[[ -f "$EXAMPLE_DIR/$CONFIG_FILE" ]] || die "Config not found: $EXAMPLE_DIR/$CONFIG_FILE"
case "$RUN_ROOT/" in "$HOME/"*) die 'RUN_ROOT must not be under HOME.' ;; esac
case "$RUN_ROOT" in *\"*|*\\*|*:*|*$'\n'*)
    die 'RUN_ROOT cannot contain quotes, backslashes, colons or newlines.' ;;
esac

export CHONK_EXAMPLES_DIR="$LAUNCHER_DIR" EXAMPLE_DIR HF_MODEL
export NEMO_RL_IMAGE CHONK_RUN_ROOT="$RUN_ROOT"
export TRAIN_NODES INFERENCE_NODES TOTAL_NODES TRAIN_GLOBAL_BATCH GENERATIONS_PER_PROMPT
export TRAIN_MICRO_BATCH INFERENCE_REQUESTS MAX_NUM_STEPS NUM_PROMPTS GYM_ENABLED
export UPDATES_PER_ROLLOUT TRAIN_EXPERT_PARALLEL_SIZE TRAIN_MB_TOKENS LOGPROB_MB_TOKENS
export BASE_LR MIN_LR MATRIX_LR GAINS_LR
export CHONK_RECIPE_PROFILE="$profile"

# ----- Submission -----

if [[ "$mode" == submit ]]; then
    printf '%s GRPO steps × %s updates = %s optimizer updates; global batch %s; %s training + %s generation nodes\n' \
        "$MAX_NUM_STEPS" "$UPDATES_PER_ROLLOUT" "$((MAX_NUM_STEPS * UPDATES_PER_ROLLOUT))" \
        "$TRAIN_GLOBAL_BATCH" "$TRAIN_NODES" "$INFERENCE_NODES" >&2
    command -v sbatch >/dev/null || { echo 'sbatch is not available.' >&2; exit 127; }
    mkdir -p "$RUN_ROOT" "/iopsstor/scratch/cscs/$USER/tmp"
    # Command-line flags work here; #SBATCH directives cannot expand recipe values.
    exec sbatch \
        --nodes="$TOTAL_NODES" --ntasks-per-node=1 --cpus-per-task=288 --gres=gpu:4 \
        --exclusive --account=csstaff --partition=normal --time="$JOB_TIME" \
        --job-name="$JOB_NAME" --output="/iopsstor/scratch/cscs/$USER/tmp/$JOB_NAME-%j.out" \
        --export=ALL "$@" "$LAUNCHER" "$recipe" --batch
fi

# ----- Job-local container environment-----

: "${SCRATCH:?SCRATCH must point to scratch storage}"
[[ "${SLURM_JOB_NUM_NODES:-$TOTAL_NODES}" == "$TOTAL_NODES" ]] || die "Requires exactly $TOTAL_NODES nodes."
RUN_DIR="$RUN_ROOT/job-${SLURM_JOB_ID}"

# Prepare overrides once in a container, then mount those snapshots on every node.
prepare_development_sources() {
    mkdir "$RUN_DIR/dev"
    export GIT_AUTHOR_NAME='Alps development' GIT_COMMITTER_NAME='Alps development'
    export GIT_AUTHOR_EMAIL=alps@example.invalid GIT_COMMITTER_EMAIL=alps@example.invalid
    export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=commit.gpgsign GIT_CONFIG_VALUE_0=false
    : > "$RUN_DIR/development-provenance.txt"
    : > "$RUN_DIR/development-applied-patches.txt"
    for component in NEMO CORE BRIDGE GYM; do
        checkout_key="DEV_${component}_CHECKOUT"
        repo_key="DEV_${component}_REPO"
        ref_key="DEV_${component}_REF"
        merge_key="DEV_${component}_MERGES"
        patch_key="DEV_${component}_PATCHES"
        [[ -n "${!checkout_key:-}${!repo_key:-}${!ref_key:-}${!merge_key:-}${!patch_key:-}" ]] || continue
        case "$component" in
            NEMO) image_path=/opt/nemo-rl; repository=https://github.com/swiss-ai/Nemo-RL.git ;;
            CORE) image_path=/opt/megatron-lm; repository=https://github.com/swiss-ai/Megatron-LM-MoE.git ;;
            BRIDGE) image_path=/opt/megatron-bridge; repository=https://github.com/swiss-ai/Megatron-Bridge.git ;;
            GYM) image_path=/opt/nemo-gym; repository=https://github.com/swiss-ai/nemo-gym.git ;;
        esac
        repository="${!repo_key:-$repository}"
        target="$RUN_DIR/dev/${component,,}"
        base_kind=image
        [[ -z "${!checkout_key:-}" ]] || base_kind=checkout
        if [[ -n "${!ref_key:-}" ]]; then
            base_kind=ref
            git clone --no-checkout "$repository" "$target"
            git -C "$target" fetch origin "${!ref_key}"
            git -C "$target" checkout --detach FETCH_HEAD
        else
            source_path="${!checkout_key:-$image_path}"
            git -c safe.directory="$source_path" clone --no-hardlinks --no-checkout "$source_path" "$target"
            git -C "$target" reset --mixed --quiet HEAD
            tar_excludes=(--exclude=.git --exclude='.env*' --exclude=.venv --exclude=__pycache__)
            git_excludes=()
            # Select submodules separately; do not copy them with the parent.
            while IFS= read -r -d '' entry; do
                [[ "$entry" == '160000 '* ]] || continue
                submodule="${entry#*$'\t'}"
                tar_excludes+=("--exclude=./$submodule")
                git_excludes+=(":(exclude,literal)$submodule")
                if [[ -e "$source_path/$submodule/.git" ]]; then
                    echo "Skipping $component submodule $submodule; override that repository separately if needed." >&2
                fi
            done < <(git -C "$source_path" ls-files --stage -z)
            tar -C "$source_path" "${tar_excludes[@]}" --dereference -cf - . | tar -C "$target" -xf -
            git -C "$target" add -A -- . "${git_excludes[@]}"
            if ! git -C "$target" diff --cached --quiet; then
                git -C "$target" commit -m 'Snapshot development working tree'
            fi
        fi
        printf '%s base %s %s\n' "$component" "$base_kind" "$(git -C "$target" rev-parse HEAD)" >> "$RUN_DIR/development-provenance.txt"
        while IFS= read -r ref; do
            [[ -n "$ref" ]] || continue
            merge_repository="$repository"
            if [[ "$ref" == *'#'* ]]; then
                merge_repository="${ref%%#*}"
                ref="${ref#*#}"
            fi
            git -C "$target" fetch "$merge_repository" "$ref"
            printf '%s merge %s\n' "$component" "$(git -C "$target" rev-parse FETCH_HEAD)" >> "$RUN_DIR/development-provenance.txt"
            git -C "$target" merge --no-edit FETCH_HEAD
        done <<< "${!merge_key:-}"
        while IFS= read -r patch; do
            [[ -n "$patch" ]] || continue
            git -C "$target" apply --index "$patch"
            printf '%s patch ' "$component" >> "$RUN_DIR/development-provenance.txt"
            sha256sum "$patch" | tee -a "$RUN_DIR/development-applied-patches.txt" >> "$RUN_DIR/development-provenance.txt"
        done <<< "${!patch_key:-}"
        printf '%s tree %s\n' "$component" "$(git -C "$target" write-tree)" >> "$RUN_DIR/development-provenance.txt"
    done
}
if [[ "$mode" == prepare-dev ]]; then prepare_development_sources; exit 0; fi

# ----- Create environment toml -----
ENV_FILE="$RUN_DIR/environment.toml"
write_environment() {
    local mounts="\"/capstor/\", \"/iopsstor/\", \"/ritom/\", \"${HOME}:${HOME}\"" component target
    for component in nemo core bridge gym; do
        [[ "$DEV_ENABLED" == 1 && -d "$RUN_DIR/dev/$component" ]] || continue
        case "$component" in
            nemo) target=/opt/nemo-rl ;; core) target=/opt/megatron-lm ;;
            bridge) target=/opt/megatron-bridge ;; gym) target=/opt/nemo-gym ;;
        esac
        mounts+=", \"$RUN_DIR/dev/$component:$target:ro\""
    done
    cat > "$ENV_FILE" <<EOF
image = "$NEMO_RL_IMAGE"
mounts = [$mounts]
workdir = "/tmp"
writable = true

[env]
PATH = "/usr/local/lib/python3.12/dist-packages/torch_tensorrt/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/mpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/amazon/efa/bin:/opt/tensorrt/bin"
PMIX_MCA_psec = "native"
NCCL_PXN_DISABLE = "1"
NCCL_CROSS_NIC = "0"
UCCL_EP_TRANSPORT = "cxi"
XDG_CACHE_HOME = "/ritom/scratch/cscs/${USER}/cache"
HF_HOME = "/ritom/scratch/cscs/${USER}/cache/huggingface"
HF_HUB_CACHE = "/ritom/scratch/cscs/${USER}/cache/huggingface/hub"
TORCH_HOME = "/ritom/scratch/cscs/${USER}/cache/torch"
VLLM_CACHE_ROOT = "/ritom/scratch/cscs/${USER}/cache/vllm"
WANDB_CACHE_DIR = "/ritom/scratch/cscs/${USER}/cache/wandb"
FLASHINFER_WORKSPACE_BASE = "/ritom/scratch/cscs/${USER}/cache/flashinfer"
PIP_CACHE_DIR = "/ritom/scratch/cscs/${USER}/cache/pip"
UV_CACHE_DIR = "/ritom/scratch/cscs/${USER}/cache/uv"

[annotations]
com.hooks.cxi.enabled = "false"
com.hooks.netstack.source = "host"
EOF
}
if [[ "$mode" == batch ]]; then
    mkdir -p "$RUN_DIR"
    [[ "$DEV_ENABLED" == 0 || ! -e "$RUN_DIR/dev" ]] ||
        die "Refusing to overwrite development snapshots: $RUN_DIR/dev"
    write_environment
    if [[ "$DEV_ENABLED" == 1 ]]; then
        srun --nodes=1 --ntasks=1 --relative=0 --cpus-per-task=1 --gres=none \
            --mpi=pmix --network=disable_rdzv_get --export=ALL --environment="$ENV_FILE" \
            -u bash "$LAUNCHER" "$recipe" --prepare-dev
        write_environment
    fi
    NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-$RUN_DIR/hf-import}"
    export RUN_DIR NRL_MEGATRON_CHECKPOINT_DIR
    export NEMO_GYM_VENV_DIR="$RUN_DIR/gym-venvs"
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

# ----- Runtime environment and caches -----

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
    : "${NEMO_RL_PATH:?The image must include the consolidated training stack}"
    if [[ "$mode" == server && "$SLURM_PROCID" == 0 ]]; then
        if [[ "$DEV_ENABLED" == 1 ]]; then
            cp /opt/alps/source-provenance.txt "$RUN_DIR/image-source-provenance.txt"
            cp /opt/alps/applied-patches.txt "$RUN_DIR/image-applied-patches.txt"
            printf 'Image defaults: image-source-provenance.txt; selected replacements follow.\n' > "$RUN_DIR/source-provenance.txt"
            cat "$RUN_DIR/development-provenance.txt" >> "$RUN_DIR/source-provenance.txt"
            cp "$RUN_DIR/development-applied-patches.txt" "$RUN_DIR/applied-patches.txt"
        else
            cp /opt/alps/source-provenance.txt "$RUN_DIR/source-provenance.txt"
            cp /opt/alps/applied-patches.txt "$RUN_DIR/applied-patches.txt"
        fi
    fi
    export PYTHONPATH="$EXAMPLE_DIR"
fi

# ----- Ray servers -----

# Rank zero starts the driver in the head's container.
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

# ----- Training driver -----

# Run beside the head's Raylet after the full resource set is ready.
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

# ----- Gym client and data -----

if [[ "$GYM_ENABLED" == 1 ]]; then
    srun --nodes=1 --ntasks=1 --relative=0 --cpus-per-task=1 --gres=none \
        --mpi=pmix --network=disable_rdzv_get --export=ALL --environment="$ENV_FILE" \
        -u bash -c '
        set -euo pipefail
        if [[ "$GYM_TRAIN_DATA" == "$RUN_DIR/data/openmathreasoning-train.jsonl" ]]; then
            bash "$EXAMPLE_DIR/prepare-data.sh" train "$GYM_TRAIN_DATA"
        fi
        if [[ "$GYM_VAL_DATA" == "$RUN_DIR/data/openmathreasoning-validation.jsonl" ]]; then
            bash "$EXAMPLE_DIR/prepare-data.sh" validation "$GYM_VAL_DATA"
        fi
        for path in "$GYM_TRAIN_DATA" "$GYM_VAL_DATA"; do
            [[ -f "$path" ]] || { echo "Dataset not found: $path" >&2; exit 2; }
        done'
fi

# ----- Launch -----

srun --nodes="$TOTAL_NODES" --ntasks="$TOTAL_NODES" --ntasks-per-node=1 \
    --cpus-per-task=288 --gres=gpu:4 --mpi=pmix --network=disable_rdzv_get \
    --export=ALL --environment="$ENV_FILE" --container-mounts="$HF_MODEL:$HF_MODEL:ro" \
    --kill-on-bad-exit=1 -u bash "$LAUNCHER" "$recipe" --server
