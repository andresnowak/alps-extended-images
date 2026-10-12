# Run Chonk and Megachonk with vLLM

These Bash scripts start vLLM on Alps GH200 nodes. Each node has four GPUs.
You do not need Spellbook, Ray, a source checkout, or extra patches.
The shared `environment.toml` selects the tested image by digest.

| Model | Script | Nodes / GPUs | TP / DP / EP | Context limit |
| --- | --- | --- | --- | --- |
| Chonk | `chonk/serve.sh` | 4 / 16 | 1 / 16 / 16 | 65,536 tokens |
| Megachonk | `megachonk/serve.sh` | 8 / 32 | 2 / 16 / 32 | 32,768 tokens |

Both scripts use these settings:

- BF16 weights and `allgather_reducescatter` communication.
- Triton experts, CUDA graphs, and native NUMA binding.
- GPU memory utilization of 0.95.
- Eight sequences per DP engine and 512 batched tokens.
- Distributed RunAI loading, concurrency 4, and a 16 GiB loader memory limit.

Chonk has seven KV heads. Do not use TP2 for this checkpoint.
The context limit includes input and output tokens.

## Start a server

Check that your account can use the selected partition.
Create the output directory before you submit the job.

```bash
mkdir -p /iopsstor/scratch/cscs/$USER/tmp
cd chonk  # Use megachonk for the larger model.
sbatch --account=infra01 --partition=normal serve.sh
```

The scripts select shared Alps HF checkpoints by default.
To use another checkpoint, set `MODEL`:

```bash
MODEL=/path/to/hf/checkpoint sbatch serve.sh
```

Submit the job from the model directory. If you submit from another directory,
set `CONTAINER_EDF` to the absolute path of `environment.toml`.

You can change the context limit:

```bash
MAX_MODEL_LEN=32768 sbatch serve.sh
```

You can change the node count with `sbatch --nodes=N`. The script calculates
DP from the allocation. Only the node counts in the table above were tested.

Wait for the server to start. The tested Megachonk run needed about eight
minutes to become ready. The server URL is in `server-url.txt`.

## Send a request

Replace `HEAD_IP` with the address in the server URL.

```bash
curl http://HEAD_IP:8000/health

curl http://HEAD_IP:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"chonk","prompt":"The capital of Switzerland is","max_tokens":32,"temperature":0}'
```

Use `"model":"megachonk"` for Megachonk.
The API listens on the cluster network. For access from outside the cluster,
use an authenticated SSH tunnel. Do not expose the API through a public proxy.

## Speed

**All speeds below are tokens per second, per request—not totals across GPUs.**
The tests used 32 concurrent requests. Each request generated 128 output tokens.

| Model | GPUs | Input tokens per request | Prefill estimate: input tokens/s per request | Decode: output tokens/s per request |
| --- | --- | ---: | ---: | ---: |
| Chonk | 16 | 32,512 | **1,957** | **14.2** |
| Megachonk | 32 | 32,512 | **747** | **5.9** |
| Chonk | 16 | 65,280 | **470** | **16.4** |

- **Prefill estimate:** input tokens divided by median time to first token.
  This includes scheduling and queue time; it is not pure GPU prefill speed.
- **Decode speed:** 1,000 divided by median time per output token in milliseconds.
  This measures output generation after the first token.

The near-32k values use median timings from two October 10 runs with a
NUMA guard. The near-64k values use one October 12 run with native NUMA binding.
The models use different GPU counts.

> **Chonk near-64k — 32 concurrent requests on 16 GPUs**
>
> - **Median time to first token: 138.8 seconds.**
> - **Median time per output token: 60.8 ms.**
> - Decode speed: about **16.4 tokens/s per request**.
> - Full test time: **147.3 seconds**.

## Logs and stop

Slurm writes its log to:

```text
/iopsstor/scratch/cscs/$USER/tmp/vllm-<model>-<job>.out
```

The corresponding `vllm-<model>-<job>/` directory contains `server.log` and
`server-url.txt`. Runtime and compiler caches use node-local `/tmp`.
The job does not write output to HOME.

The server runs until its two-hour time limit. To stop it earlier:

```bash
scancel JOB_ID
```
