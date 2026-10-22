# Chonk GRPO on Alps

## Choose a recipe

| Recipe | Profile | Configuration | Defaults |
| --- | --- | --- | --- |
| `vllm/gsm8k` | `adamw` | [YAML](vllm/gsm8k/config.yaml) | [adamw.env](vllm/gsm8k/adamw.env) |
| `vllm/gsm8k` | `muon` | [YAML](vllm/gsm8k/config.yaml) | [muon.env](vllm/gsm8k/muon.env) |
| `megatron/gsm8k` | `adamw` | [YAML](megatron/gsm8k/config.yaml) | [adamw.env](megatron/gsm8k/adamw.env) |
| `megatron/gsm8k` | `muon` | [YAML](megatron/gsm8k/config.yaml) | [muon.env](megatron/gsm8k/muon.env) |
| `vllm/nemo-gym` (k8s) | `recipe` (default) | [YAML — sync GRPO](vllm/nemo-gym/grpo-chonk.yaml) | [recipe.env](vllm/nemo-gym/recipe.env) |

The GSM8K [shared YAML](common/gsm8k.yaml) defines training and data settings. Each
backend YAML defines its generation engine. Each `.env` profile is self-contained.

## Launch

Set `NEMO_RL_IMAGE` to the image built with the consolidated training stack
(in your shell or `.env`). The older runtime-patched image cannot run this launcher.
From this directory, choose one:

```bash
bash train.sh vllm/gsm8k --profile muon \
  --grpo-steps 58 --prompts 128 --generations-per-prompt 16 --updates-per-rollout 4 \
  -- --account=YOUR_PROJECT --partition=YOUR_PARTITION --time=07:00:00
# Or use native generation:
bash train.sh megatron/gsm8k --profile adamw \
  -- --account=YOUR_PROJECT --partition=YOUR_PARTITION --time=07:00:00
```

Put training options before `--` and Slurm options after it. Run `bash train.sh --help`
for all options. CLI values override the shared `.env` and selected profile.
The launcher also works from another directory when called by its full path.

Use `bash train.sh`, **not direct `sbatch`**: the launcher selects resources before
submitting. For GSM8K, AdamW uses 8 training nodes and 4 generation nodes.
Muon uses 16 training nodes and 4 generation nodes. Each node has four GPUs.
Check partition availability with `sinfo -a` before submitting.

## Development overrides

Normal runs use the image's installed stack. To replace a component, use
`--dev-COMPONENT-checkout PATH` or `--dev-COMPONENT-ref REF`. Replace `COMPONENT`
with `nemo`, `core`, `bridge` or `gym`. Use `--dev-COMPONENT-repo URL` to select
another repository with a ref. The default repositories are the Swiss AI forks.
Use a commit SHA for repeatable runs.

For example, start with a clean Core revision and apply only the requested fixes:

```bash
bash train.sh vllm/gsm8k --profile adamw \
  --dev-core-ref 0536034901962160aac9742d134a84c55e28d224 \
  --dev-core-patch "$PWD/../patches/megatron/0001-distrib-optimizer-detach-fp32-param-shards.patch" \
  -- --account=YOUR_PROJECT --partition=YOUR_PARTITION
```

Or use a working checkout and merge explicit PRs before applying patches:

```bash
bash train.sh megatron/gsm8k --profile muon \
  --dev-nemo-checkout /ritom/scratch/cscs/$USER/tmp/nemo-rl-dev \
  --dev-nemo-pr 5 --dev-nemo-pr 1 \
  --dev-nemo-pr https://github.com/swiss-ai/Nemo-RL/pull/6 \
  --dev-nemo-patch /ritom/scratch/cscs/$USER/tmp/nemo-rl-dev-patches/generation.patch \
  -- --account=YOUR_PROJECT --partition=YOUR_PARTITION
```

PR numbers refer to the selected repository. Use a full GitHub PR URL for
another fork. Repeat `--dev-COMPONENT-merge REF` to merge branches or commits.
The launcher merges the selected refs in order, then applies the patches in
order. A conflict stops the job before training. Resolve conflicts in a checkout.

The launcher replaces only the components you select. So, a clean checkout or ref
**doesn't inherit the image's patches** and instead only uses what you provide it (for Megatron, NeMo, Bridge and Gym).
If you select only PRs, merges or patches, then the launcher starts from the image's version, and its existing fixes remain.

At job start, one container prepares source copies under `$RUN_DIR/dev/`.
Each node mounts these copies read-only. The launcher does not change the
original checkouts. It copies working-tree changes, deletions and untracked
files, but excludes `.git`, `.env*`, `.venv` and Python caches. It does not copy
submodules with a parent checkout; select each repository separately.

Keep checkouts and patch files on shared storage that the container can access.
The launcher copies their contents when the job starts, not when you submit it.
`development-provenance.txt` records base versions, merges, patch checksums and
tree hashes. In development mode, `applied-patches.txt` lists only the patches
added for that job. The `image-*` provenance files record the baked defaults.
Overrides do not install dependencies or rebuild extensions. Use source versions
that are compatible with the image's Python/CUDA stack.
Normal runs do not fetch or patch code.

## Shared training configuration

- UCCL provides the DeepEP-compatible training dispatcher over Slingshot/CXI
  (`UCCL_EP_TRANSPORT=cxi`). Token dispatch uses BF16.
- Router weights are frozen (`freeze_moe_router`), as are quantile-balancing
  thresholds (`moe_router_quantile_balancing_freeze`).

## Settings

- Edit the shared or backend **YAML** for structured model and engine settings.
- Edit the selected **`.env` profile** for resource, batch, learning-rate and run-budget defaults.
- Use `--micro-batch`, `--lr`, `--matrix-lr`, `--gains-lr`, `--train-nodes` or
  `--generation-nodes` to override those defaults for one run.
- `NEMO_RL_IMAGE` selects the container image. [train.sh](train.sh) embeds the
  mounts and cache settings and writes a job-local `environment.toml`.
- `HF_MODEL` overrides the read-only starting checkpoint.
- `CHONK_RUN_ROOT` selects where job directories, checkpoints and logs are written.
  The initial HF-to-Megatron conversion is stored under each job's `hf-import/`.
- Global batch is derived as `prompts × generations per prompt ÷ updates per rollout`.
  The launcher rejects fractional batches. NeMo-RL/Megatron checks parallelism
  and microbatch compatibility.
- The GSM8K default budget is **58 GRPO steps × 4 updates = 232 optimizer updates**.
  The requested GRPO step count controls the run; there is no separate one-epoch cap.

The default HF checkpoint is:

```text
/capstor/store/cscs/swissai/infra01/apertus_checkpoints/v2/hf/chonk/pretraining/long_context/long-context-iter-0003576
```

### Local environment and W&B

Create a local settings file without replacing an existing one:

```bash
cp -n .env.example .env
chmod 600 .env
```

Set `WANDB_ENABLED=true`, `WANDB_PROJECT`, and `WANDB_API_KEY` in `.env`.
Set `WANDB_ENTITY` if you need a specific user or team namespace.
You can also set a run root there, for example:

```bash
CHONK_RUN_ROOT=/ritom/scratch/cscs/$USER/tmp/chonk-grpo/runs
```

The launcher sources `.env`, so use only a trusted file. Its values override
matching environment variables. It is Git-ignored; never commit API keys.
Keep run directories and caches outside `$HOME`.

## GSM8K

The active recipes validate on all 1,319 GSM8K test questions.
`GSM8K_VAL_PERIOD` overrides the validation interval. Weight-checkpoint settings
include a final save, no optimizer state, a save period of
1,000 steps, and a checkpoint deadline of 4h40m. All outputs stay under the run root.

By default, each GSM8K rollout batch has **128 prompts × 16 responses = 2,048 samples**.
It is split into **four groups of 512 samples**, with one optimizer update per
group: **four optimizer updates per GRPO step**. These are distinct training
batches, not four passes over the same samples.

GSM8K has 7,473 training prompts. With `drop_last=True`:

| Prompts per GRPO step | GRPO steps per epoch | Epochs at 150 steps |
| --- | --- | --- |
| 64 | 116 | 1.29 |
| 128 | 58 | 2.59 |

Epochs count prompt batches, regardless of responses per prompt or optimizer updates.

### Reference results

These supplied reference results are not measurements of the local examples.

| Metric | vLLM AdamW | Megatron AdamW | vLLM MD | Megatron MD |
| --- | --- | --- | --- | --- |
| Job ID | 3686420 | 3686429 | 3686113 | 3689066 |
| Training + generation nodes | 8 + 4 | 8 + 4 | 16 + 4 | 16 + 4 |
| Run time (min) | 78 | 132 | 74 | 110 |
| Evaluation before | 49.1% | 48.1% | 48.2% | 47.2% |
| Evaluation after | **90.7%** | **91.1%** | **78.5%** | **78.7%** |
| `gen_kl` max | 0.0021 | 0.0018 | 0.0007 | 0.0006 |
| Trainer time (s) | 60.8 | 59.4 | 55.7 | 54.5 |
| Generation time (s) | 33.9 | 174 | 33.2 | 143 |
| Throughput (tokens/s/GPU) | 432 | 322 | 263 | 206 |
| Time per GRPO step (s) | 69.8 | 125.9 | 65.9 | 106.7 |
| Average refit time (s) | 7.6 | 3.8 | 8.7 | 2.8 |
| Trajectory age (steps) | 1 | 1 | 1 | 1 |

## NeMo Gym (k8s) results (synchronous GRPO)

Run [5066314 / cfgsgu5r](https://forge.coreweave.com/wandb/apertus/smoke-apertus2-chonk-grpo/runs/cfgsgu5r).
Results below are from GRPO step 245 unless another step is stated.
Training reward is not validation accuracy.

| Metric | Result |
| --- | --- |
| Optimizer | Muon / MD-decoupling |
| Logged GRPO steps | 245 |
| Optimizer updates | 490 |
| Prompts × responses per rollout | 128 × 8 |
| Global batch | 512 |
| Beginning training reward (step 1) | **1.37%** |
| Last training reward (step 245) | **5.57%** |
| Loss | 0.0286 |
| Generation KL error | 0.0293 |
| Time per GRPO step (s) | 95.90 |
| Trainer time, two optimizer updates (s) | 28.89 |
| Generation time (s) | 18.15 |
| Policy/reference log-probability inference (s) | 26.09 |
| Refit time (s) | 10.36 |

## Source revisions

The image installs the fixed revisions in
[`install-training-stack.sh`](../sources/install-training-stack.sh), including NeMo PRs
5, 1, 6, 7 and 9, the PR 8 cache fix, and the tested upstream PR 2517 revision.
The [NeMo patches](../patches/nemo-rl/) and [Core patches](../patches/megatron/)
are applied when building the image, not when launching a job. NeMo-RL, Core,
Bridge and the Gym client are already installed; normal runs need no runtime
Git access or package installation.

The launcher copies the image's `source-provenance.txt` and `applied-patches.txt`
to the run directory. Recipe YAML and profiles remain editable outside the image.
The default HF checkpoint is read-only.

## Run more experiments

For running and comparing RL experiments, try [rl-infra](https://github.com/swiss-ai/rl-infra),
created by [@davmels](https://github.com/davmels).
