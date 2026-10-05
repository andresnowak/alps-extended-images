# `nemo-rl` container image

[NeMo-RL](https://github.com/NVIDIA-NeMo/RL) with a Megatron policy and the Megatron
generation backend, plus the MoE kernel stack the Apertus2 MoE checkpoints need.
The image also includes the Apertus2 vLLM fork for vLLM generation configurations;
Megatron generation remains supported and does not require weight refit.

Everything NeMo-RL imports at run time is installed in the image, so jobs need no
`PYTHONPATH` overlays. The previous Megatron-only pin set was validated 2026-09-14
with a two-node GRPO run through NeMo Gym on 8x GH200. The aligned UCCL/vLLM pin
set still needs an image build and GPU validation.

## Building locally

The build context is the repository root, not this directory: the
`COPY Alps-Images/...` lines only resolve from there.

```bash
podman build -f Alps-Images/apps/nemo-rl/Containerfile \
  --build-arg BASE_IMAGE=jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/pytorch-cuda:26.06-py3-alps7-dev \
  -t nemo-rl:local .
```

## What the image contains

| Component | Pin | Why |
|---|---|---|
| UCCL | `uccl-project/uccl@540cc775ba8ab231122f9a98a6d832699f239f70` | complete CUDA 13 wheel, EP and P2P extensions, public P2P library/headers and packaged DeepEP wrapper; same full build and 8 jobs as `app/vllm-apertus2` |
| NIXL | `ai-dynamo/nixl@de8115ca97d3f8fb63a4988e9b4d4a038b2e0f72` (`1.3.2`) | CUDA 13 bindings and UCCL plugin, built with the same Meson options as `app/vllm-apertus2` |
| vLLM | `swiss-ai/vllm@d4d41485a1cc1aee0d906d19f41523d2fdc67463` | Apertus2 KDA support; NGC torch, 32 build jobs and 4 NVCC threads; Rust 1.93.0 builds the required Rust frontend and tool parser |
| TransformerEngine | `v2.17` | CUDA graph support; this is megachonk's TE, one minor above the `te212` sibling image |
| DeepGEMM | `deepseek-ai/DeepGEMM@559d79fb` | FP8 grouped GEMM |
| grouped_gemm | `FFGGSSJJ/grouped_gemm@45118e54` | MoE GEMM with gradient-accumulation fusion |
| nvidia-resiliency-ext | `0.6.0` | first release containing the commit Megatron-LM pins (`15a85156`); older ones break async checkpoint save |
| Emerging-Optimizers | `FFGGSSJJ@cc1385ee` | decoupled Muon (`md_decoupling`) |
| flash-linear-attention | `swiss-ai/flash-linear-attention@1820dba7e15fb927294779f23ac5097eb3927b25` | v0.5.2 plus one commit adding per-channel `A_log` support in KDA decay gates |
| ray | `2.56.1` | NeMo-RL worker runtime; not in the base image |
| flashinfer-python | `0.6.16.post3` | **required**, not optional: NeMo-RL hardcodes `sampling_backend="flashinfer"` in its Megatron worker, so `InferenceConfig.__post_init__` raises `ImportError` without it |
| openai | `2.7.2` | nemo-gym requires `<=2.7.2` and pins each child server venv to the *parent* version, so a newer parent makes every Gym venv unresolvable |
| transformers / megatron-energon / hydra-core / math-verify / mlflow / tensordict / swanlab | pinned | NeMo-RL import-time dependencies |
| uv | `0.11.23` | installs everything in this image, and is what NeMo Gym shells out to at run time to build its per-server venvs |

Installs go through `uv pip install --system`, not pip: it resolves the whole
dependency set at once rather than package by package. The NeMo-RL runtime packages are pinned to exact versions and source builds to
commit SHAs or release tags. vLLM's transitive requirements retain their upstream
ranges, so their resolved versions can change between builds. The image intentionally does not use `--exclude-newer`: JFrog metadata for
some required build packages does not include upload dates, causing uv to exclude
those packages entirely.
uv is bootstrapped with the repository's `pip_install` helper because nothing
else exists at that point, and `UV_DEFAULT_INDEX` points at the same CSCS JFrog
mirror so Gym venvs built at run time resolve through it too.

Every install is additionally run against `/opt/alps/base-pins.txt`, a constraints
file generated from the selected base image's own
`torch`/`torchvision`/`triton`/`numpy` versions. A transitive dependency therefore
cannot drag in a PyPI torch wheel and shadow that NGC stack. A final build check
verifies both that those versions are unchanged and that every pinned layer still
holds the version it asked for: a later `uv pip install` is free to upgrade an
earlier package, and only a check after the last layer sees it.

## Alignment with `app/vllm-apertus2`

The base is `pytorch-cuda:26.06-py3` (NGC Torch `2.13.0a0`), rather than the original
25.12 base. The selected vLLM commit targets the Torch 2.11 stable C++ ABI, whose
headers and APIs are missing from the original NGC Torch 2.10 build. Torch still
comes exclusively from the NGC base; no replacement PyPI Torch is installed.

vLLM is built in a separate stage so its build dependencies do not alter the
NeMo-RL runtime stack. Rust 1.93.0 handles the source's Cargo `resolver=3`
workspace, and `VLLM_REQUIRE_RUST_FRONTEND=1` makes missing Rust components fatal
instead of silently accepting an incomplete wheel. Build checks verify the
`vllm-rs` executable and native tool parser; GPU smoke tests import that parser.
Its requirements are filtered to preserve NGC torch and
the CUDA toolkit. A targeted Torch override also keeps the exact NGC prerelease
when transitive metadata requests a stable Torch release. Runtime resolution
protects NeMo Gym's `openai==2.7.2`
and NeMo-RL's Ray pin. FlashInfer (`0.6.16.post3`), TVM FFI (`0.1.11`), CUTLASS DSL
(`4.6.2`), NVTX (`0.2.15`) and Transformers (`5.17.0`) align with the vLLM stack.

UCCL is pinned to upstream `uccl-project/uccl` main at
`540cc775ba8ab231122f9a98a6d832699f239f70` (2026-10-05), including the P2P OOB
server deadlock fix for connections dropped mid-parse. It uses the complete
`BUILD_TYPE=all` CUDA 13 build, not an EP-only install.
The image includes `uccl.ep`, `uccl.p2p`, the wheel-packaged `deep_ep` wrapper,
and the public P2P headers/library needed to build NIXL's UCCL plugin. Both EP
and P2P select CXI, with `UCCL_CXI_THREADING=safe`. NIXL's `nixl` and `nixl-cu13`
packages come from the same pinned source and plugin build as the vLLM image;
the generic PyPI NIXL wheel is not used. GPU smoke tests check the P2P and DeepEP
APIs, CUDA 13 bindings, and discovery of the UCCL backend by an actual NIXL agent.

## What the image does not contain

NeMo-RL, Megatron-Bridge and Megatron-LM are not vendored: bind-mount the checkouts
and put them on `PYTHONPATH`. They are actively developed forks, and baking them in
would force an image rebuild for every source change.

NeMo Gym's per-server venvs are built at run time too, since they derive from the
Gym checkout's own `requirements.txt`.
