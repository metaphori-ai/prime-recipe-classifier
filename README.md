# prime-recipe-classifier

Post-training a 2B model to classify recipes against a controlled culinary taxonomy — origin tradition path plus
ingredient-class `contains` — on Prime Intellect's stack (`verifiers` v1 environment, `prime-rl` GRPO), with every arm
evaluated on one human-labeled gold set under one scoring function.

The interesting part is not the classifier. It is what the RL loop exposed about the reward, the data and the training
setup, and how each finding changed the next run. The dated record is in [`LOG.md`](LOG.md); the analysis in
[`TRAINING.md`](TRAINING.md).

![project journey](journey.svg)

## Results — 2B unless noted, reward v2, audited gold

| arm | 439 gold | 350 public subset | notes |
|---|---|---|---|
| A. base (Qwen3.5-2B, v3 prompt) | 0.244 | 0.249 | zero-shot |
| B. SFT, consensus silver (LoRA, 2 epochs, M5) | 0.782 | 0.785 | |
| B. SFT 4B, same data | 0.794 | 0.802 | not distinguishable from 2B (paired diff +0.013 ± 0.010) |
| C. GRPO from base (prime-rl, 1× H200, 23 steps) | 0.517 | — | +0.27 from reward alone; plateaued (entropy 0.39 → 0.03) |
| B″. SFT on Sonnet-5.5 silver | pending | | relabel running |
| D. GRPO from SFT | pending | | |

Reward = 0.55 · origin (hierarchical set-Jaccard over the taxonomy path, influence term) + 0.45 · contains
(recall-weighted, lexicon-grounded evidence). Specification and history in [`TRAINING.md`](TRAINING.md#reward-v2-2026-11-08--after-arm-c-exposed-three-exploits).

## What failed, and what changed

| observed | diagnosed as | changed |
|---|---|---|
| gold 0.46 → 0.12 while train reward rose; rollouts listing every class, quoting unrelated ingredients, always claiming `incomplete` | reward exploits: flat floor, FP cheaper than FN, substring "evidence" | reward v2: `incomplete` must match gold; FP cost 1.0; evidence must be a real ingredient *and* lexicon-supported for the class; spam cap |
| mismatch KL 0.002 → 0.39, grad norm 5 vs clip 1, repetition to the token cap | stale rollouts: single-GPU inference ran ~10× ahead of the trainer | `max_staleness` 8 → 2, batch 64 → 128 episodes |
| train reward flat at ~0.5, entropy → 0.03, one distinct answer per group | premature convergence; no advantage signal | next run: rollout temperature 1.2; add distinct-answers-per-group metric |
| SFT influences 5/32, RL influences 0/32 | labels are the ceiling: consensus silver had 0.006 influences/recipe | teacher bake-off on gold → Sonnet 5.5 (78% exact, 18/32 influences) → relabel |
| 9 gold `contains` labels the verifier could not ground | gold errors (extract-only alcohol, gelatin ≠ pork) | audited 11 rows; all arms re-scored |

Teacher comparison on the 439 gold (`label_origin compare`): Qwen3.6-35B 67.4% exact / 11 influences · DeepSeek V4 68.5% / 12 ·
Claude Sonnet 5.5 78.1% / 18 · qwen3.6:27b (local) pending.

## Layout

```
taxonomy.yaml                     245 origin nodes, 20 contains classes, derived diets, ingredient traps
env/recipe_tax/                   scoring (reward), lexicon, taxonomy loader, dataset build, SFT prep, teacher labeling
env/environments/recipe_classify/ verifiers v1 Taskset/Task; configs/ (vf-run local/ollama/mlx, prime-rl rl/sft/eval)
env/tests/                        12 tests on the reward
scripts/                          pod setup/sync (driver, GPU, port, NCCL gotchas scripted), eval runners, this export
data/                             publishable subset only (see Data)
LOG.md / TRAINING.md / TAXONOMY.md / OVERVIEW.md / PLAN.md
```

## Hardware and where each step ran

- **Mac (M5, 128 GB)**: all 439-recipe evaluations (base models via Ollama, fine-tuned models via `mlx_lm.server`), the size
  curve (Qwen3.5 0.8B → 27B zero-shot), SFT for 2B and 4B (`mlx-lm-lora`, LoRA r8 on 16 layers, ~1.7 h / ~3.2 h), the
  local 27B teacher evaluation, and all data/label tooling. Evals are fast because the ~3k-token system prompt is prefix-cached.
- **Rented 1× H200 (Prime Intellect compute, ~$6.7/h)**: GRPO with `prime-rl`, trainer and vLLM sharing the one GPU
  (`CUDA_VISIBLE_DEVICES=0,0`, filesystem weight broadcast). Two runs, ~2.5 GPU-hours total including the failed one.
- **Hosted APIs**: Prime Inference for the first two origin teachers (DeepSeek V4, Qwen3.6-35B); Anthropic API for the
  Sonnet 5.5 teacher (prompt caching, ~89% of input tokens cached); TypeSafe Jev for `contains` labels.

## Reproduce

```bash
cd env && uv sync
uv run python -m pytest -q tests                                          # reward unit tests
uv run vf-validate recipe-classify                                        # gold self-scores 1.0
# baseline against a local Ollama model:
bash ../scripts/vf_run_ollama.sh qwen3.5:2b 350
# SFT on Apple silicon (mlx-lm-lora; mlx-lm's own trainer OOMs on Qwen3.5's GatedDeltaNet backward):
uv run python -m recipe_tax.sft_prepare && mlx_lm_lora.train --model Qwen/Qwen3.5-2B --train --train-mode sft \
  --data ../data/sft --mask-prompt --batch-size 4 --gradient-accumulation-steps 2 --iters 1600 --learning-rate 1e-5 \
  --max-seq-length 3584 --grad-checkpoint --num-layers 16 --adapter-path ../runs/sft-2b
# GRPO on one GPU with prime-rl (see configs/rl-2b.toml comments for the single-GPU flags):
CUDA_VISIBLE_DEVICES=0,0 uv run rl @ env/environments/recipe_classify/configs/rl-2b.toml --inference.vllm.gpu-memory-utilization 0.45
```

## Data

- **Gold** (`data/gold/gold.jsonl`, 350 of 439): human-labeled by the author; the policy never trains on it.
- **Train** (`data/dataset/`, `data/dataset_dsv4/`): silver labels — `contains` from TypeSafe Jev (0.99 vs gold), origin from
  LLM teachers. Consensus set (two-teacher agreement) and single-teacher set (DeepSeek, no gate).
- **Raw** recipes: Food.com (CC0) and Wikibooks Cookbook (CC BY-SA 3.0; each record carries its `source` URL). The 89 gold and
  ~460 train recipes captured from the author's own app are excluded from this repository; the private 439-recipe numbers are
  reported alongside the 350-recipe public numbers above.
- Labels and derived data in this repo: CC BY 4.0. Code: MIT ([LICENSE](LICENSE)).

## How this was built

Directed, labeled and decided by Jasdeep Jaitla; code and most analysis written with Claude as a pair, over roughly one
weekend. Every decision, mistake and reversal is in [`LOG.md`](LOG.md) with a date — including the model's own errors
(a rename it claimed to have applied and had not; a reward design that RL took apart in 35 steps). That record is the point.
