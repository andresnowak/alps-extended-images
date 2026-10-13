# Chonk GRPO on Alps

Three standalone Slurm/Ray examples use Megatron training:

| Directory | Generation backend | Configuration |
| --- | --- | --- |
| [`vllm/`](vllm/) | vLLM, standard all-gather/reduce-scatter | `grpo-chonk-vllm.yaml` |
| [`megatron/`](megatron/) | Megatron dynamic inference | `grpo-chonk-megatron.yaml` |
| [`nemo-gym-vllm/`](nemo-gym-vllm/) | vLLM, remote NeMo Gym math verifier | `grpo-chonk.yaml` |

The first two examples use GSM8K. The `nemo-gym-vllm/` example includes an
OpenMathReasoning JSONL subset.

Each GSM8K directory has its own Bash launcher, YAML and five-shot prompt.
The examples share this README and [`environment.toml`](environment.toml).
Neither GSM8K example needs files from the other backend directory.

## Submit

Choose a backend and submit from its directory:

```bash
cd vllm
sbatch train-gsm8k-chonk.sh
```

Or:

```bash
cd megatron
sbatch train-gsm8k-chonk.sh
```

The defaults are account `csstaff`, partition `normal`, 24 nodes, two updates
and a two-hour limit. Use `sinfo -a` to check which partitions are available.
Set `--partition` and other Slurm options for your account:

```bash
sbatch --account=YOUR_PROJECT --time=02:00:00 train-gsm8k-chonk.sh
```

The driver runs in the Ray head's container. Worker tasks wait for the driver
to finish. They return the driver's exit status.

## Shared configuration

- Training: 16 nodes / 64 GPUs, TP1/PP4/CP1/EP16, no virtual pipeline.
  The four pipeline stages contain 8/12/12/10 layers.
- Generation: 8 nodes / 32 GPUs, TP1/PP1/EP8. Generation uses separate nodes.
- Training microbatch: 1/GPU; global batch: 128.
- Training-worker allocator: `PYTORCH_ALLOC_CONF=expandable_segments:True`.
  Generation workers do not inherit this setting.
- Generation batch: 32; 32 prompts × 8 generations = 256 responses per update.
- BF16 model parameters, FP32 training gradient buffers and synchronous GRPO.
  The KL penalty is 0.01. Checkpoint saves are disabled. The DeepGEMM FP8
  expert backward accumulation kernel needs FP32 gradient storage.
- Training expert offloading: coarse-grained, 8 chunks, 2 stages, in-place FP8
  parameters and extra parameter storage.
- Training dispatcher: `flex` with the image's UCCL-backed `deepep` interface.
  Token dispatch uses BF16. The examples disable expert-communication overlap
  and delayed weight-gradient computation. UCCL settings match
  [`pretrain/chonk.sh`](https://github.com/swiss-ai/pretrain/blob/6cb927e8b153e2fe9cd660fe0a6332c2ee7ce2a2/apertus-2/pretraining/chonk/chonk.sh):
  - `UCCL_EP_TRANSPORT=cxi`
  - `UCCL_EP_DISPATCH_CONFIG=24,40,512,32,512`
  - `UCCL_EP_COMBINE_CONFIG=24,7,512,32,512`
  - `UCCL_EP_CPU_TIMEOUT_SECS=6000`
  - `CUDA_DEVICE_MAX_CONNECTIONS=1`
  These settings apply to training workers only. Generation keeps its existing
  dispatcher.
- Weight refit: ordinary NeMo packed NCCL with Bridge conversion for all examples.

The default read-only HF model is:

```text
/capstor/store/cscs/swissai/infra01/apertus_checkpoints/v2/hf/chonk/pretraining/long_context/long-context-iter-0003576
```

The launchers use the shared `../environment.toml`. Set
`NEMO_RL_ENVIRONMENT=/path/to/other.toml` to use another EDF. Set
`HF_MODEL`, `MAX_NUM_STEPS`, `TRAIN_MICRO_BATCH`, `INFERENCE_REQUESTS`,
`NUM_PROMPTS` or `NRL_MEGATRON_CHECKPOINT_DIR` in the submission environment
to change these defaults.

The default vLLM run directory is
`/iopsstor/scratch/cscs/$USER/tmp/nemo-rl-gsm8k-chonk/`.
Megatron uses `nemo-rl-gsm8k-chonk-megatron/` in the same scratch `tmp` directory.
Set `CHONK_RUN_ROOT` to use another run directory.

Slurm writes startup logs to `gsm8k-chonk-JOBID.out` or
`gsm8k-chonk-megatron-JOBID.out` in that scratch `tmp` directory.
Create this directory before submission. The launchers do not change the
original checkouts or model files.

The launcher and EDF set the cache locations. Bridge config locks use
`/tmp/chonk-caches-JOBID/megatron-config-locks` by default.
Set `MEGATRON_CONFIG_LOCK_DIR` to use another location.

## Generation backends

### vLLM

The vLLM examples use an asynchronous engine with DP8/EP8 cohorts and BF16.
They use 80% GPU memory, a maximum model length of 1280 and dummy initial weights.
Communication uses `allgather_reducescatter`, not DeepEP/UCCL.
GRPO updates stay synchronous. Asynchronous generation does not enable
asynchronous GRPO. CUDA graphs are allowed (`enforce_eager: false`).
Multiprocessing uses `spawn`.

### Megatron

The Megatron example uses native dynamic inference with local, layer-scope
CUDA graphs and MoE padding. With the current EP8/NCCL dispatcher, decode uses
CUDA graphs. Prefill runs without CUDA graphs.

## Validation

- Megatron: the native configuration completed two GRPO updates (job 5039868).
- vLLM: async generation and packed NCCL refit completed two GRPO updates
  (job 5045801).
- NeMo Gym vLLM: remote math verification and packed NCCL refit completed two
  GRPO updates (job 5046558); see its [validation notes](nemo-gym-vllm/README.md).

These are operational smoke tests, not model-quality or numerical-parity checks.

## Sources

The launchers define repository URLs, commit pins and PR refs. Use the launcher
as the source of truth for versions:

- [vLLM launcher](vllm/train-gsm8k-chonk.sh)
- [Megatron launcher](megatron/train-gsm8k-chonk.sh)
- [NeMo Gym launcher](nemo-gym-vllm/train-chonk.sh)

Each launcher downloads public repositories into the job's run directory.
It records the fetched revisions in `applied-prs.txt` and the source trees in
`source-provenance.txt`. Check these files to identify the code used by a run.
No developer checkout, bundled patch or Python launcher helper is needed.

The launchers install Lens and soundfile in a job-local package directory.
The Gym example also installs the packages in `gym-client-requirements.txt`.
Set `NEMO_RL_SOURCE`, `BRIDGE_SOURCE` or `MEGATRON_SOURCE` to use compatible Git
URLs or local checkouts. These checkouts must contain the required revisions.
