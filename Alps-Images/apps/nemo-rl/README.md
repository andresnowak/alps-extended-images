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
| UCCL-EP | `andresnowak/uccl@0c166355310a04b2344ca0c5c0df818b04d26af0` | same source pin and 8 build jobs as `app/vllm-apertus2`; `uccl.ep` + the `deep_ep` wrapper for Megatron's flex dispatcher |
| vLLM | `swiss-ai/vllm@d4d41485a1cc1aee0d906d19f41523d2fdc67463` | Apertus2 KDA support; built against NGC torch with 32 build jobs and 4 NVCC threads |
| TransformerEngine | `v2.17` | CUDA graph support; this is megachonk's TE, one minor above the `te212` sibling image |
| DeepGEMM | `deepseek-ai/DeepGEMM@559d79fb` | FP8 grouped GEMM |
| grouped_gemm | `FFGGSSJJ/grouped_gemm@45118e54` | MoE GEMM with gradient-accumulation fusion |
| nvidia-resiliency-ext | `0.6.0` | first release containing the commit Megatron-LM pins (`15a85156`); older ones break async checkpoint save |
| Emerging-Optimizers | `FFGGSSJJ@cc1385ee` | decoupled Muon (`md_decoupling`) |
| flash-linear-attention | `v0.5.2` | KDA kernels |
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
NeMo-RL runtime stack. Its requirements are filtered to preserve NGC torch and
the CUDA toolkit. A targeted Torch override also keeps the exact NGC prerelease
when transitive metadata requests a stable Torch release. Runtime resolution
protects NeMo Gym's `openai==2.7.2`
and NeMo-RL's Ray pin. FlashInfer (`0.6.16.post3`), TVM FFI (`0.1.11`), CUTLASS DSL
(`4.6.2`), NVTX (`0.2.15`) and Transformers (`5.17.0`) align with the vLLM stack.

UCCL retains the EP-only build used by Megatron, with per-expert batching enabled.
This does not add the Apertus2 image's full UCCL P2P build or NIXL UCCL plugin;
NeMo-RL's existing NIXL transport package is unchanged.

## What the image does not contain

NeMo-RL, Megatron-Bridge and Megatron-LM are not vendored: bind-mount the checkouts
and put them on `PYTHONPATH`. They are actively developed forks, and baking them in
would force an image rebuild for every source change.

NeMo Gym's per-server venvs are built at run time too, since they derive from the
Gym checkout's own `requirements.txt`.
