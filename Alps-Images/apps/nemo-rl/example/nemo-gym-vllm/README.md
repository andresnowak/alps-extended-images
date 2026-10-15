# Chonk GRPO with NeMo Gym

This example uses Megatron training, async vLLM generation and packed NCCL
weight updates. A remote NeMo Gym server verifies the math answers.
The job starts its own Gym head, agent and policy adapter. It does not start
a local math verifier.

## Run

From the parent example directory:

```bash
cd nemo-gym-vllm
sbatch train-chonk.sh
```

No environment variables are required for the default setup.

- Account: `csstaff`; partition: `normal`; time limit: two hours.
- Training: 16 nodes. Generation: 8 nodes. Each node has 4 GPUs.
- Two GRPO iterations, each with 32 questions and 8 responses per question.
- Shared container environment: `../environment.toml`.
- Math verifier: `https://gym-dev.swissai.svc.cscs.ch/math_with_judge`.

Check that the partition is available before submission. See the
[parent README](../README.md) for training settings. The
[launcher](train-chonk.sh) defines source pins and job-local package installs.
It keeps the image's Torch, Ray and OpenAI versions unchanged.

## Dataset and prompt

`math-smoke.jsonl` contains the first 500 training tasks from NVIDIA's
[Nemotron-RL-math-OpenMathReasoning dataset](https://huggingface.co/datasets/nvidia/Nemotron-RL-math-OpenMathReasoning/tree/219bf0170cd87cd5db2bbf33715ce9ea43a2c59c),
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
It keeps the original questions, answers and prompts. Each row adds `agent_ref`
for the [Swiss AI Gym client](https://github.com/swiss-ai/nemo-gym/tree/swissai-k8s).
The subset is included in this directory. No dataset download is needed.
Two iterations do not use all 500 questions.

The launcher loads `math-chat-template.jinja` for policy tokenization and HTTP
serving. The template asks for a short solution and a boxed final answer.
The verifier uses math verification, not an LLM judge.

## Environment overrides

Set `MAX_NUM_STEPS` to change the maximum number of GRPO iterations
(default: `2`). For example:

```bash
MAX_NUM_STEPS=10 \
sbatch --time=04:00:00 train-chonk.sh
```

See the [shared override table](../README.md#environment-overrides) for model,
batch, container and output settings. This example also accepts:

| Variable | Default | Purpose |
| --- | --- | --- |
| `GYM_MATH_URL` | Math verifier URL shown above | Remote math resources server. |
| `GYM_TRAIN_DATA` | `$EXAMPLE_DIR/math-smoke.jsonl` | Gym-format training dataset. |
| `NEMO_GYM_SOURCE` | `https://github.com/swiss-ai/nemo-gym.git` | Gym Git URL or local checkout. |
| `NEMO_GYM_REF` | Pinned revision in the launcher | Gym revision. Use matching client dependencies. |

To use another dataset:

```bash
GYM_TRAIN_DATA=/path/to/train.jsonl sbatch train-chonk.sh
```

Set `CHONK_RUN_ROOT` to change the default artifact directory:
`/iopsstor/scratch/cscs/$USER/tmp/nemo-rl-nemo-gym-chonk/`.
Slurm's `--output` option selects the log file.

## Check verifier access

Run this command on a compute node. The gateway restricts access by source.

```bash
curl --fail --max-time 15 https://gym-dev.swissai.svc.cscs.ch/math_with_judge/
```

## Test result

Job `5046558` ran this repository launcher and completed both GRPO iterations.
Remote rewards, weight updates and policy training completed. The driver and
Slurm exit codes were zero.

Losses were `0.0134` and `0.0057`. Average rewards were `0.0078` and `0.0039`.
Generation KL errors were `0.0179` and `0.2499`.
This is an integration test, not a model-quality or numerical-parity test.
It does not test stateful environments or concurrent training jobs.
