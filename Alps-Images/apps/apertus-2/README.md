# Apertus2 Megatron-LM

One Apertus2 software stack for Chonk and Megachonk training on Alps GH200
nodes. The image provides CUDA kernels and communication libraries. Megatron-LM,
pretrain, data and checkpoints remain in external directories.

Build target: `cuda`. Image: `apertus-2-cuda:alps7-dev`.

## Software stack

Base: `pytorch-cuda:25.12-py3` (Python 3.12, CUDA 13.1).

| Package | Version | Source | Commit |
| --- | --- | --- | --- |
| UCCL EP and `deep_ep` | source checkout | `swiss-ai/uccl` | `fa4c325c6113b798ae0a009ef8239e07352264fb` |
| Transformer Engine | `2.12.0` | `NVIDIA/TransformerEngine` | `5671fd3675906cda1ade26c24a65d3dedd88eb89` |
| DeepGEMM | `2.6.1` | `deepseek-ai/DeepGEMM` | `559d79fb6994a58b8a15b4b93bf13ccc16edf247` |
| nv_grouped_gemm | `1.1.4.post8` | `swiss-ai/grouped_gemm` | `45118e547923803ed52f80f240cf974620c9bdc5` |
| Emerging-Optimizers | `0.2.0` | `FFGGSSJJ/Emerging-Optimizers` | `cc1385eec1966aea2aebfb6dd6fc66a898312a55` |
| flash-linear-attention | `0.5.2` | `swiss-ai/flash-linear-attention` | `1820dba7e15fb927294779f23ac5097eb3927b25` |
| causal-conv1d | `1.7.0` | `Dao-AILab/causal-conv1d` | `cd81f0413cad2fc1e6f17e785ac39f59aae690cd` |
| nvidia-resiliency-ext | `0.6.0` | package index | — |

Native packages are built from pinned sources against the base image's Torch.
Torch, Torchvision, Triton and NumPy versions are kept as installation
constraints. Source pins and base constraints are stored in
`/opt/alps/apertus2-*.txt`.

The SwissAI 0.5.2 FLA fork adds per-channel `A_log` support for KDA decay gates.

UCCL is built for SM90, four NVLink peers per node and libfabric CXI.
The runtime environment must set `UCCL_EP_TRANSPORT=cxi`.

### Emerging-Optimizers fork

The pinned fork adds Muon's `"moonlight"` scaling mode:

```text
0.2 * sqrt(max(size_out, size_in))
```

Upstream supports the same scaling with `scale_mode="spectral"` and
`extra_scale_factor=0.2`, but does not accept `scale_mode="moonlight"` directly.
The image keeps the fork to preserve this configuration without changes.

### Python dependencies

The image uses Transformers `5.17.0`, Tokenizers `0.23.2`, Hugging Face Hub
`1.31.0`, W&B `0.30.0`, gRPC `1.84.0` and Protobuf `7.36.1`.
The OpenTelemetry SDK and HTTP exporter are also included.
All runtime additions are pinned in [`requirements.txt`](sources/requirements.txt).

## Build

From the repository root:

```bash
BUILD_DIR=/iopsstor/scratch/cscs/$USER/tmp/apertus2-image-build
mkdir -p "$BUILD_DIR"
manual-build/manual-build.sh app apertus-2 cuda "$BUILD_DIR/build.sh"
bash "$BUILD_DIR/build.sh"
```

The build requires an aarch64 GH200 compute node with Podman. The generated
script selects the canonical, content-addressed base image.
See the [manual build guide](../../../manual-build/README.md).

## Run Chonk pretraining

[`example/chonk.sh`](example/chonk.sh) is based on the
[Chonk launcher in `swiss-ai/pretrain`](https://github.com/swiss-ai/pretrain/blob/6cb927e8b153e2fe9cd660fe0a6332c2ee7ce2a2/apertus-2/pretraining/chonk/chonk.sh).
It uses [`environments/chonk.toml`](environments/chonk.toml), which selects the
built image by its content-addressed tag. `CONTAINER_ENV` can select another EDF.

**This is a production configuration:** training requests 256 nodes and 1,024
GPUs. The warm-up requests 32 nodes. It is not a small-scale test.

### Set the paths and account

Start from the repository root. Replace the input paths with existing directories
that are accessible through the EDF mounts:

```bash
SCRIPT=$(realpath Alps-Images/apps/apertus-2/example/chonk.sh)
ACCOUNT=your-slurm-account
PARTITION=your-slurm-partition

export MEGATRON_LM_DIR=/path/to/Megatron-LM-MoE
export TOKENIZER_MODEL=/path/to/tokenizer
export DATA_ROOT=/path/to/indexed-data
export WORK_DIR="$SCRATCH/tmp/chonk-pretraining"

mkdir -p "$WORK_DIR/logs"
cd "$WORK_DIR"
```

- Use the [`swiss-ai/Megatron-LM-MoE` fork](https://github.com/swiss-ai/Megatron-LM-MoE).
  Set `MEGATRON_LM_DIR` to its local checkout, which must contain `pretrain_gpt.py`.
- Use the tokenizer from [`swiss-ai/apertus-tokenizer-development`](https://github.com/swiss-ai/apertus-tokenizer-development).
  Set `TOKENIZER_MODEL` to its local tokenizer directory, which must contain
  `tokenizer.json`.
- `DATA_ROOT` must contain Megatron indexed `.bin` and `.idx` shards.
- Use a new `WORK_DIR` for a new run. Training resumes from its checkpoint
  directory when a checkpoint exists.
- Select an available partition that permits the requested node count and time.
  Training requests 12 hours. `sinfo -a -o "%P %a"` lists partition availability.

Always select the account at submission. The script has no default account.
Submit from `WORK_DIR` so Slurm writes logs under `WORK_DIR/logs`.

### Check the arguments

```bash
DRY_RUN=1 bash "$SCRIPT"
```

This prints the training arguments and task environment. It does not submit a
job or test the image on a GPU.

### Compile the Triton kernels

```bash
SBATCH_ACCOUNT="$ACCOUNT" SBATCH_PARTITION="$PARTITION" bash "$SCRIPT" --warmup
```

The command submits a warm-up job. Wait for it to finish and check its log.
Repeat until the log reports `SATURATED`. The script saves the kernel archive
under `WORK_DIR/triton-warm/`. Training requires this archive by default.

### Submit training

```bash
sbatch -A "$ACCOUNT" --partition="$PARTITION" "$SCRIPT"
```

The script starts one task per GPU. Checkpoints are stored under
`WORK_DIR/results/ckpts/`. Slurm logs are stored under `WORK_DIR/logs/`.

The example retains the upstream runtime package installs. These installs can
shadow packages in the image. The example has not been validated with a full
training run on this image.

## Tests

- vetnode hardware and software checks;
- package versions, GPU imports and Muon scale checks;
- Transformer Engine and per-channel KDA gate forward/backward tests;
- UCCL dispatch/combine on two GH200 nodes with eight tasks.

These tests do not cover training convergence, all FP8/KDA kernels or an
end-to-end Megatron training run.
