# Training recipes — recipe-classify, Qwen3.5 2B and 4B

Everything is measured by one function (`recipe_tax.score`) on one set (439 gold). Fifty-recipe numbers are ±0.04; **all
post-training comparisons run on the full 439** (`-n 439`).

## The experiment

| arm | 2B | 4B | what it answers |
|---|---|---|---|
| A. base (v3 prompt) | 0.357* | 0.467* | starting point (*n=50; re-run at n=439) |
| B. SFT on silver | ? | ? | how far imitation of two-teacher-agreed labels gets you |
| C. GRPO from base | ? | ? | can the reward alone teach vocabulary + judgment |
| D. GRPO from SFT | ? | ? | the usual production recipe; does RL add anything on top of imitation |

Report per arm: reward, origin, contains, gate-fail %, unknown-classes/rollout, unsupported-evidence/rollout, and the
origin confusion list. The headline question: **does D beat B, and on which component?** Secondary: C vs B at 2B — if RL
alone fixes the raw-ingredient habit, that's the cleanest demonstration of reward > imitation.

Bonus experiment once something is trained: evaluate the student **without the taxonomy tree in the system prompt**. If it
has memorized the vocabulary, production cost drops ~10× (2,400 tokens/call → ~200).

---

## Recipe B — SFT, locally on the M5 (free, hours)

Data: `data/sft/{train,valid}.jsonl` (chat format, built by `recipe_tax.sft_prepare`; system prompt identical to eval/RL).

Tooling: `mlx-lm` (Apple MLX). LoRA on a 2B/4B at seq ~2.8k fits comfortably in 128 GB.

```bash
uv tool install mlx-lm             # gives mlx_lm.lora / mlx_lm.fuse / mlx_lm.server on PATH
mlx_lm.lora --help                 # confirm flags below still exist

# 2B. --model takes the HF id directly (mlx-lm converts on load) or an mlx-community repo.
mlx_lm.lora --model Qwen/Qwen3.5-2B --train --data data/sft \
  --fine-tune-type lora --num-layers 16 \
  --mask-prompt \
  --batch-size 4 --grad-accumulation-steps 2 --iters 800 --learning-rate 1e-5 \
  --max-seq-length 3072 --grad-checkpoint \
  --steps-per-eval 100 --save-every 200 --adapter-path runs/sft-2b

mlx_lm.fuse --model Qwen/Qwen3.5-2B --adapter-path runs/sft-2b --save-path runs/sft-2b-fused

# serve it OpenAI-compatible and evaluate with the SAME environment on all 439:
mlx_lm.server --model runs/sft-2b-fused --port 8080 &
cd env && OPENAI_API_KEY=x uv run vf-run recipe-classify -m runs/sft-2b-fused -n 439 -r 1 -c 4 \
   @ environments/recipe_classify/configs/local.toml \
   --client.base_url http://localhost:8080/v1 --client.api_key_var OPENAI_API_KEY
```
**What actually ran** (the block above is the original plan; `mlx_lm.lora` OOMs on Qwen3.5 — see Blockers). Verbatim from shell
history, this produced `runs/sft-2b` (B = 0.782) and, with `4B`/`runs/sft-4b`, the 4B row. It is LoRA rank 8 on 16 layers,
auto-fused into the adapter dir; the pod's `uv run sft` is a *full* fine-tune — keep that in mind when comparing B'' across stacks.
```bash
mlx_lm_lora.train --model Qwen/Qwen3.5-2B --train --train-mode sft \
  --data ~/svml/prime-intellect/data/sft --mask-prompt \
  --batch-size 4 --gradient-accumulation-steps 2 --iters 1600 --learning-rate 1e-5 \
  --max-seq-length 3584 --grad-checkpoint --num-layers 16 \
  --steps-per-eval 200 --save-every 400 \
  --adapter-path ~/svml/prime-intellect/runs/sft-2b
# Sonnet-silver variant (product artifact): --data .../data/sft_sonnet --adapter-path .../runs/sft-2b-sonnet
# then: mlx_lm.server --model runs/sft-2b-sonnet --port 8080 --chat-template-args '{"enable_thinking": false}' &
#       bash scripts/vf_run_mlx.sh runs/sft-2b-sonnet 439
```
Why these numbers:
- `--mask-prompt` is the one flag that matters. The system prompt is ~2,470 tokens and the answer ~60; without masking,
  97 % of the loss is the model memorizing the taxonomy tree it already gets for free, and the label signal drowns.
- 800 iters × (4 × 2 accumulation) = 6,400 examples ≈ 2 epochs of 3,090. Watch `valid` loss every 100 steps; stop when it flattens.
- `--max-seq-length 3072` covers prompt + answer with headroom; `--grad-checkpoint` keeps memory flat (not needed on 128 GB, harmless).
- Expect ~1–2 s/step on the M5 for the 2B → well under an hour. 4B: same command, `Qwen/Qwen3.5-4B`, `runs/sft-4b`.
- `data/sft/` already has `train.jsonl` + `valid.jsonl` in the `messages` format mlx-lm reads natively.

Alternative: hosted SFT (`prime train sft.toml`) — beta, needs access enabled, bills GPU-hours, and checkpoints need a
support-provided `trainer.json` to export. Local is simpler for a 2B/4B.

---

## Recipe C/D — GRPO with prime-rl

Two ways to run it. Both use the same `configs/rl-2b.toml` below.

### Option 1 — hosted (`prime train`)  — least setup, least visibility
- Needs the environment on the **Environments Hub** (hosted runtime resolves `taskset.id`); dataset must ship inside the
  wheel → **publishable subset only** (2,504 train / 350 eval; SaffronSifu captures excluded). Check `prime env push --help`
  for visibility options first.
- 2 GPUs minimum (trainer + inference), billed per GPU-hour. ~50–150 steps → ~1–3 h → tens of dollars.
- Monitoring: dashboard reward curve + rollouts; `prime train logs <id> -c trainer`.
- Export friction: checkpoints are DCP format; converting needs the run's `trainer.json` from support.

### Option 2 — self-hosted prime-rl on a rented 2× H100 (Prime compute) — the instrument panel, unabridged
- Rent a 2-GPU pod (`prime pods create` / dashboard; ~$1.5–4 per GPU-hour), clone prime-rl, `uv sync`.
- `rsync` the repo folder up (env package + `data/dataset/train.jsonl` — all sources, private pod).
- `uv pip install -e environments/recipe_classify` inside prime-rl's venv so `taskset.id = "recipe-classify"` resolves locally.
- Run `uv run rl @ configs/rl-2b.toml` with `[wandb]` on → every curve from OVERVIEW.md's MICRO section is live.
- Checkpoints in HF format on the pod; `rsync` them down; serve with vLLM or convert to MLX/GGUF for local eval.
**This is the recommended path for the first run**: it's the one where you *see* the loop.

### `configs/rl-2b.toml` (starting point — tune after the first 20 steps)
```toml
name = "recipe-classify-2b-grpo"
max_steps = 120
seq_len = 4096                      # ~2.5k prompt + ≤400 completion

[model]
name = "Qwen/Qwen3.5-2B"            # arm C. For arm D: path to the fused SFT weights.

[deployment]
num_train_gpus = 1
num_infer_gpus = 1

[trainer.optim]
lr = 1e-6                           # full fine-tune; 2B fits on one H100

[orchestrator]
batch_size = 64                     # prompts per step
[orchestrator.train]
group_size = 8                      # rollouts per prompt (advantage is within-group)
[orchestrator.train.sampling]
max_completion_tokens = 400
temperature = 1.0
[orchestrator.renderer]
name = "prime-qwen3"                # confirm the Qwen3.5 renderer name in prime-rl

[[orchestrator.train.source]]
name = "recipe-classify"
[orchestrator.train.source.env.taskset]
id = "recipe-classify"
dataset_path = "/workspace/prime-intellect/data/dataset/train.jsonl"   # self-hosted; omit on Hub (bundled default)
[orchestrator.train.source.env.agent.harness]
id = "null"
[orchestrator.train.source.env.agent.runtime]
type = "subprocess"

[ckpt]
interval = 20
keep_last = 3

[wandb]
project = "recipe-classify"
```

### What to watch (first 30 steps), and what each shape means
- **reward ↑ while `unknown_classes` ↓** — the policy is learning the vocabulary. Expected first, fast (steps 1–20 on 2B).
- **`gate_fail` → 0 quickly** — format solved. If it stays >5 % after 30 steps, lower lr or raise group_size.
- **`origin` ↑ slower than `contains`** — expected; hierarchy is harder than membership.
- **reward ↑ but eval (439) flat** — overfitting the train prompts or exploiting the silver's quirks; stop and inspect rollouts.
- **response_chars ↑ steadily** — length exploitation (evidence spam?). The unsupported-evidence penalty should hold it; if not, cap tokens lower.
- **entropy collapse / identical rollouts** — raise temperature or group_size; the advantage goes to zero when siblings agree.
- **KL runaway** — lr too high for this seq_len; halve it.
- **Unplaced › Contemporary / American over-prediction on eval** — the majority-floor hack; raise the cap (`--cap-primary 150`) and rebuild.

### Pre-flight checklist (owner)
```
prime train models                 # is Qwen/Qwen3.5-2B (and 4B) cached? exact id spelling
prime train gpus                   # $/GPU-hour by type
prime env push --help              # visibility: private?  (decides Option 1 viability for non-publishable data)
mlx_lm.lora --help                 # flag names for Recipe B
```

---

## Pre-flight results (2026-11-08)
- `prime train models`: `Qwen/Qwen3.5-2B` and `Qwen/Qwen3.5-4B` pre-cached (also 0.8B, 9B, Qwen3.6-35B-A3B, Qwen3.8-27B). Ids confirmed as written in `rl-2b.toml`.
- `prime train gpus`: H100_80GB $4.79/GPU-hr, RTX_PRO_6000B_96GB $3.89/GPU-hr, both available; billed from pods-ready, prorated to the second.
  → hosted GRPO run, 2 GPUs × ~1.5 h ≈ **$12–15**. Trailing line "No GPU types available. Contact support" is a second
  (reserved/other) table being empty — on-demand is fine.
- `prime env push --visibility PRIVATE` exists → **Option 1 no longer forces the publishable-only subset**. A private Hub
  push can bundle the full train set. (It still sits on Prime's registry; your call on the SaffronSifu captures.)
- `mlx_lm` not installed → `uv tool install mlx-lm` first.

### Known blocker: Qwen3.5 LoRA on mlx-lm OOMs on the first backward (2026-11-08)
Qwen3.5 is a hybrid: 3/4 of its layers are Gated DeltaNet (linear attention), 1/4 full attention. mlx-lm's GDN
backward has no fused kernel, so training falls back to a per-token ops path whose memory blows up with sequence
length — Metal OOM at the first backward regardless of batch size / num-layers / grad-checkpoint, while validation
(forward only) passes. Tracked upstream: ml-explore/mlx-lm#1206, #1956; ml-explore/mlx#3539. Qwen3 (previous gen,
plain transformer) trains fine with the same settings — the architecture is the variable.

Options, in the order to try:
1. **`mlx-lm-lora` — WORKS.** (`uv tool install mlx-lm-lora`; `mlx_lm_lora.train --train-mode sft ...`, flags otherwise
   the same as mlx_lm.lora except `--gradient-accumulation-steps`.) Routes GDN layers through a chunked path with a
   working VJP. 2B, batch 1 × accum 8, seq 3584: 2.2 s/it → 800 iters ≈ 30 min. Val baseline 0.70. Use this for Recipe B.
2. Move SFT onto the same rented GPU as GRPO (prime-rl has an `sft` entrypoint). 2B × 3k examples × 2 epochs on one H100
   ≈ 10–15 min ≈ $1–2. Same pod, same stack, then GRPO from the SFT weights without moving anything. **This is the
   default if (1) fails** — it costs less than the time already spent on Metal.
3. Not an option: dropping to Qwen3-1.7B locally — breaks the 2B/4B comparison and Prime's cached models are Qwen3.5.

## Results log

### Arm B smoke (2B, SFT ¼ epoch = 800 examples, LoRA r8/16 layers) — n=50, same 50 recipes as base
| metric | base 2B | SFT-smoke 2B |
|---|---|---|
| reward | 0.357 | **0.741** |
| origin | 0.473 | 0.684 |
| contains | 0.216 | **0.810** |
| gate_fail | 0.02 | 0.00 |
| unknown_classes / rollout | 2.74 | 0.04 |
| unsupported_evidence / rollout | 1.56 | 0.32 |
| resolved_paths (suffix rescue) | 0.56 | 0.12 |
| response_chars | 356 | 204 |

Reading: vocabulary and format are solved by imitation almost immediately (unknown_classes → 0, gate → 0). `contains`
nearly quadrupled. `origin` moved less — the judgment half. Eval ran in 19 s because mlx_lm.server caches the shared
system-prompt prefix (3.1k tokens processed once, then ~50/recipe).

Failure modes in the bottom 6 (all origin):
- **Influences dropped entirely.** Silver has 0.006 influence tags/recipe vs 0.073 in gold (12× under-represented: the
  teachers rarely emitted them and the two-teacher agreement filter removed most survivors). SFT faithfully learned "never
  emit an influence" → every fusion recipe caps out. This is a ceiling of imitation that the reward does not share — the
  cleanest thing for GRPO (arm D) to fix.
- **Over-specific Indian regional paths** (Gujarati ×3, Goan) for recipes whose gold is Anglo-Indian or Durban Indian. Not
  a frequency attractor (Gujarati = 15/3,252 in train); looks like under-training on the host-table convention. Re-check
  after the 2-epoch run before drawing conclusions.
- One invalid hybrid path ("North American > American > Indian > North Indian") — the model splices two real paths. Soft
  gate rescued it at ×0.8; RL would be penalized for it directly.

### Arm B — 2B SFT, 2 epochs (1,600 × batch 4, LoRA r8/16 layers, 1h42m on M5) — **n=439**
| metric | 2B SFT (B) | notes |
|---|---|---|
| **reward** | **0.786** | base 2B n=50 was 0.357; arm A at n=439 pending |
| origin | 0.704 | primary exact-path match 262/439 = 59.7% (hosted teachers: 67–69% exact) |
| contains | 0.885 | |
| gate_fail | 0.000 | |
| unknown_classes / rollout | 0.002 | vocabulary fully learned |
| unsupported_evidence / rollout | 0.096 | |
| resolved_paths | 0.011 | almost no suffix rescues |
| response_chars | 195 | |
| influence tags | **5 predicted vs 32 gold** | recipes with a gold influence score 0.561 vs 0.786 overall |

Val loss: 0.682 → 0.059 (200) → 0.042 (600) → 0.040 (1600); flat after one epoch, train→0.02, no val uptick.
Reward histogram (deciles 0.0–0.9): 4, 6, 11, 8, 58, 31, 22, 41, 47, 211.

Top confusions (gold → pred): Contemporary→American 8; Southern US→American 6; American→Italian-American 5;
American→French 5; Italian-American→Italian 5; Nigerian sub-traditions (Yoruba/Hausa/Igbo)→Nigerian 10 total;
Contemporary→Foundational 4. Pattern: the model under-commits on depth (stops at the national node) and the
American/Contemporary/Italian-American boundary is where the convention is fuzziest for everyone, humans included.

Reading: a 2B after two local epochs lands at 0.786 — within noise of the local qwen3.6:27B teacher's 0.794 (n=50) and
above both hosted teachers' agreement-filtered silver quality. Imitation's remaining gaps: (1) influences, structurally
absent from silver, (2) depth under-commitment, (3) the American-cluster boundary. (1) is the one the reward sees and the
data does not → the GRPO hypothesis for arm D.

### Arm A — 2B base, v3 prompt — **n=439**
reward **0.273** · origin 0.320 · contains 0.216 · gate_fail 0.039 · unknown_classes 2.98/rollout · unsupported_evidence 1.23 ·
response_chars 374 · influence tags **116 predicted vs 32 gold** (sprays them; SFT took this to 5).
The n=50 figure (0.357) over-estimated by 0.08 — first fifty are easier. Table entries are n=439 only.

| 2B | reward | origin | contains | infl. pred/32 |
|---|---|---|---|---|
| A. base | 0.273 | 0.320 | 0.216 | 116 |
| B. SFT 2 ep | **0.786** | 0.704 | 0.885 | 5 |
| C. GRPO from base | | | | |
| D. GRPO from SFT | | | | |

### Operational rule: one MLX process on the GPU at a time
Running `vf-run` against `mlx_lm.server` while `mlx_lm_lora.train` was mid-run killed the server's generation thread with
a Metal OOM (4B trainer at 34.7 GB peak + server's prompt cache; Metal has no arbitration between processes). The trainer
survived; the server had to be restarted. Evals wait for training to finish. Kill a stale server by port:
`kill $(lsof -ti tcp:8080)` — `%1` job handles only exist in the shell that started the job.
Also added task metrics `influence_pred`, `influence_gold`, `primary_depth` (taskset.py) for RL-time curves.

### Arm C — first live GRPO run (2B from base, single H200, 2026-11-08)
- Gold eval (n=150, same first 150 of eval.jsonl every time): **v0 0.3065 → v10 0.4612**. Train reward first-6-step mean 0.24 → steps 13–18 mean 0.39.
- Trainer: 37–40 s/step after a ~3–5 min first-step kernel compile; entropy ~0.34, mismatch KL ~0.002, grad norm 0.6–1.0, peak 100 GiB.
- `batch_size` counts **episodes**: 64 = 8 prompts × group 8. Small → noisy per-step reward.
- Single-GPU imbalance: vLLM generates ~10× faster than the trainer consumes; orchestrator buffers, then discards ~90% of
  rollouts as stale (`max_staleness=8`). Training stays on fresh data; cost is wasted inference that contends with the trainer.
  For arm D: raise `max_staleness` (policy drift per step is tiny at lr 1e-6) and/or `batch_size` 64→256.
- Task errors fixed during the run: scorer crashed on wrong-shaped `evidence`/`contains` (now coerced, or gate 'malformed');
  `parse_answer(None)` on empty completions (now `or ""`). Both were silently dropping episodes.

### Arm B — 4B SFT, 2 epochs (3h13m on M5) — **n=439**
reward **0.799** · origin 0.724 · contains 0.890 · exact 271/439 (61.7%) · influence 9 pred / 32 gold · primary_depth 2.52.
Paired vs 2B SFT on the same 439: mean diff **+0.013 ± 0.010 (s.e.)** — not significant. 78 recipes better by >0.1, 68 worse.
Val loss ended 0.041 (2B: 0.040). **Both sizes sit on the silver's noise floor**; imitation cannot use the 4B's extra capacity.
New failure flavor: invented hierarchy ("Southeast Asian > Malay > Nyonya", ×4) — rescued by soft gate at ×0.8.
Decision: 2B is the product candidate (same quality, half the cost). RL arms run on the 2B only; 4B GRPO deferred.

| arm | 2B | 4B |
|---|---|---|
| A. base | 0.273 (n=439) | 0.467 (n=50) |
| B. SFT 2 ep | 0.786 | 0.799 |
| C. GRPO from base | running (gold-150: 0.307 → 0.461 @ step 10) | — |
| D. GRPO from SFT | next | deferred |

### Finding: RL on this train set is RL against SILVER labels (2026-11-08)
Gold evals so far (n=150): v0 0.307 · v10 0.461 · v20 0.442. Train reward regime-shifted to ~0.47 at steps 22–25 (after v20).
The reward *function* is ours, but on `train.jsonl` it is evaluated against the teachers' consensus labels. Therefore:
- "Train reward ↑, gold flat" = the policy agreeing more with the teachers, not with the human. Same ceiling as SFT.
- **The influence gap is a label problem, not a method problem.** Silver has 0.006 influence tags/recipe; a predicted influence
  on a train recipe with no silver influence triggers the 0.6/0.4 split with a zero influence term → penalized. RL against
  silver will suppress influences exactly as SFT did. Arm D cannot fix this with the current train labels.
- Fix = better labels, not a better optimizer: relabel train origin (at least the fusion-looking subset) with the local
  qwen3.6:27B (0.794 at n=50, > both hosted teachers, free), then re-run B and D. Raises the ceiling for every arm.
Arm C's live question is therefore: does RL-against-silver beat SFT-on-silver (0.786)? Decided by v30+ on the full 439.

### Reward v2 (2026-11-08) — after arm C exposed three exploits
What the collapsed policy did (traces at v30): always claim `incomplete` (flat 0.5 floor); list every contains class
(FP 0.5 vs FN 2.0 made spam beat honesty); quote any real ingredient as "evidence" (`"egg": "tur dal"`). Then entropy
collapse (0.34→0.13), mismatch-KL 0.002→0.39, grad-norm 5 vs clip 1, repetition to the 400-token cap.
Changes in `recipe_tax/scoring.py`:
- `incomplete`: gold&pred → 1.0; pred-only → **0.0**; gold-only → score against gold's listed classes.
- FP cost 0.5 → **1.0**; duplicate classes count as FPs; > 10 distinct classes → contains 0.
- Evidence is grounded against a lexicon (`recipe_tax/lexicon.py`, shared with the SFT synthesizer, extended) and the TITLE line:
  a claim NOT in gold is a false positive regardless of quote; a claim IN gold whose quote is fabricated (not an ingredient,
  or points at a line with no lexicon word while a supporting line exists) is dropped → miss. Missing evidence is not penalised
  (gold carries none) — known hole, watched by the new `evidence_coverage` metric.
Validation: gold self-scores 1.0 on 439/439. Offline re-score of saved traces: base 2B 0.273→0.244, SFT 2B 0.786→0.779,
SFT 4B 0.799→0.791; the collapsed policy's spam output: contains 0.5→0.0. **All table entries are v2 from here**; v1 numbers above are kept for history.
Gold errors found by the verifier (to fix in gold, not in the reward): 7 `alcohol` labels based only on vanilla/lemon extract
(foodcom-418033, ss-paprika-2703F801…, ss-paprika-377631E6…, ss-paprika-52E66AA9…, ss-paprika-E0252B30…, wikibooks-hot-milk-cake,
wikibooks-microwave-chocolate-cake-i); foodcom-401066 `pork` (gelatin ≠ pork); foodcom-11182 `beef`/`red_meat` with no meat listed.
RL config for the rerun: `batch_size` 64→128 episodes (16 prompts/step), `max_staleness` 8→2.

### Gold audit (2026-11-08) — 11 rows corrected, backup `data/gold/gold.backup-2026-11-08.jsonl`
7× drop `alcohol` (extract only), 1× drop `pork` (gelatin), 1× drop `beef`+`red_meat` (no meat listed), 2× Contemporary→American
(Green Garlic Dip, Fish & Eggplant Stew). `eval.jsonl` copies refreshed. Gold self-scores 1.0 on 439/439.
All saved traces re-scored offline against **reward v2 + corrected gold** (same prompt version as when they were generated):

| arm (reward v2, gold-audited, n=439) | reward | origin | contains | exact | infl (of 32) |
|---|---|---|---|---|---|
| A. base 2B | 0.244 | 0.320 | 0.152 | 52 | 2 |
| B. SFT 2B (consensus silver) | 0.782 | 0.703 | 0.877 | 262 | 5 |
| B. SFT 4B (consensus silver) | 0.794 | 0.724 | 0.880 | 271 | 9 |
| B'. SFT 2B (dsv4, single teacher) | not run (superseded by B'') | | | | |
| **B''. SFT 2B (Sonnet 5.5 silver, full FT on pod)** | **0.821** (vLLM) / 0.817 (MLX) | 0.725 | 0.937 | — | 3 emitted |
| B''-mac. SFT 2B (Sonnet silver, LoRA r8/16 layers, MLX) | 0.782 / 0.769 (two draws) | 0.702 | 0.880 | — | 16 emitted |
| C. GRPO 2B from base (v2 reward, staleness 2, batch 128, stopped @ step 23) | **0.517** | 0.524 | 0.510 | — | 0 |
| **D. GRPO 2B from B'' (T 1.2, 80 steps, reward v2)** | **0.857** | 0.767 | 0.967 | — (traces on pod) | 11 emitted |

B'' notes: prime-rl `sft`, 320 steps × batch 32 (~2 epochs of 4,813), lr 1e-5, full fine-tune, 33 min on the H200; eval via
vLLM T=0.7 (median 62 output tokens, max 133, 0 truncations, gate_fail 0). primary_depth 2.54 (gold ≈ 2.6; arm C was 1.32).
**Labels moved the 2B more than doubling parameters did** (+0.038 vs 2B-consensus, +0.027 vs 4B-consensus). Caveats: B/4B were
evaluated through MLX on the Mac, B'' through vLLM on the pod (both T=0.7); B'' was trained and evaluated with the
Contemporary-is-form sentence in CONVENTIONS, which B never saw. influence_pred 0.0068 (3/439) although 6.6% of training
rows carry an influence: SFT on a 2B treats a rare optional field as "omit" — the one dimension left for arm D to show RL adds something.

### LoRA vs full fine-tune on the same labels (2026-11-09)
Same Sonnet silver, same 2B, same eval T 0.7. LoRA r8 on 16 layers (5.6M trainable, mlx-lm-lora, 1,600 iters ≈ 2.7 epochs, 95 min on
the M5) = **0.782** — identical to the consensus-silver LoRA — vs full FT (1.9B trainable, prime-rl, 2 epochs, 33 min on an H200) =
**0.821**. The gap is mostly `contains` (0.880 vs 0.937), whose labels are the same Jev labels in both, so this reads as a
*capacity* limit of r8/16-layer LoRA on this task rather than a label effect; origin moved less (0.702 vs 0.725). The LoRA emits
influences on 16 recipes (most of any arm) without an origin gain → low precision, the 0.4 influence term nets to ~0.
**Eval noise, measured:** the same LoRA model scored twice on 439 at T 0.7 gave 0.782 and 0.769 (Δ 0.013). Treat any two table
entries within ~0.02 as tied; this covers B 2B vs 4B (0.782 vs 0.794) and v30–v80 of arm D's gold-150 curve.
Eval-stack control (done): the full-FT weights served through mlx_lm.server score **0.817** on 439 vs 0.821 through vLLM — within
the measured noise. Cross-stack entries in the table are comparable, and the LoRA gap is capacity. Product implication: ship the
full-FT weights (they run on the Mac at 0.817) or retry LoRA at higher rank / all layers; r8 on 16 layers caps `contains` at ~0.88.

### Arm D result (2026-11-09)
GRPO from B'' (Sonnet-silver SFT), rollout temperature 1.2, batch 128 episodes (16 prompts × 8), max_staleness 2, lr 1e-6, 80
steps, 2 h 12 m on one H200 (trainer 70–120 s/step at MFU 4–8%; 88% of rollouts discarded as stale on the shared GPU).
Gold-150 during training: v0 .818 → v10 .827 → v20 .841 → v30 .845 → v50 .846 → v70 .866 → v80 .852. Final policy
(broadcasts/step_80 + SFT dir config/tokenizer) on all 439: **0.857** (B'' 0.821): origin 0.725 → 0.767, contains 0.937 → 0.967,
depth 2.54 → 2.48, **influence_pred 0.007 → 0.025 (3 → 11 recipes)**. Entropy 0.07 → 0.019; no_signal groups 18% → ~38%;
grad norm 0.3 → 0.15; mismatch KL 0 throughout; no reward hacking under v2.
Readings: (1) the "SFT then RL" recipe worked as advertised — imitation did the big jump, selection added +0.036 on top with no
new labels; (2) RL amplified a *rare* behaviour (influences) 3.7× — the earlier claim that it could not move something SFT emits on
0.7% of outputs was wrong: rare is enough, absent is not; (3) D's origin 0.767 is still 0.08 under the Sonnet teacher's 0.843 on
the same gold — the label ceiling moved, it did not disappear; the headline 0.857 includes Jev-quality `contains`; (4) depth did
not improve (2.54 → 2.48) — the 10% depth bonus is still too small to pull against the shallow-path safety of the Jaccard.
Not yet done: exact-match count and distinct-answers-per-group from the pulled traces; the v70 checkpoint is not saved (ckpt
interval 20 → step_60/step_80 DCP only).

### Teacher selection on the 439 gold (2026-11-08) — `label_origin compare`, now with influence recall
| teacher | exact | origin score | influences found /32 | emitted | parse errors |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B (Prime Inference) | 67.4% | 0.761 | 11 | 13 | 19 |
| DeepSeek V4 (Prime Inference) | 68.5% | 0.784 | 12 | 14 | 1 |
| **Claude Sonnet 5.5** (Anthropic API, OpenAI-compat) | **78.1%** | **0.843** | **18** | 33 | 0 |
| qwen3.6:27b (local Ollama, thinking, 2k tokens) | 80.1% of **291 answered** | 0.874 | 2 (of 7 in the answered subset) | 4 | **148 truncated (34%)** — survivorship-biased; rerun with thinking off and with 8k budget |
| **qwen3.6:27b (local Ollama, thinking OFF, 600 tokens)** | **72.6%** | **0.808** | **17** | 19 | 1 — clean, comparable row |
| qwen3.6:27b (local Ollama, thinking ON, 6k budget) | 74.5% | 0.826 | 17 | 21 | 0 — all answered; 1,865 out tokens/recipe (24× no-think) for +0.018 |
Sonnet emits influences at the gold rate (33 vs 32) with ~55% precision; the hosted teachers suppress them. Top confusion for
every teacher: gold `Unplaced > Contemporary` → `American` (12 for Sonnet) — likely partly a GOLD inconsistency with our own
convention; review in the label tool. Cost: 2.23M input tokens / 439 recipes (~5k each). Full train relabel ≈ 27M tokens;
adding a native Anthropic path with prompt caching for the system block before running it.
27B with thinking and a budget it can't exhaust: 0.826, +0.018 over no-think and the same 17 influences, at 24× the output
tokens; the 2k-budget run's 0.874 'on survivors' was survivorship bias (full-answer number is 0.826). Thinking moved confidence
both ways (fewer depth collapses, a new over-deepening British → English) rather than adding knowledge. Sonnet still leads.
27B no-think: second place (Sonnet 0.843 > 27B 0.808 > DeepSeek 0.784 > Qwen3.6-35B 0.761). Influence *precision* is the
inverse story: 27B 17/19 = 89% vs Sonnet 18/33 = 55% — it only names an influence when sure. Its confusions are almost all
depth collapses (Yoruba/Hausa → Nigerian ×9, Italian-American/Midwestern/Californian/Hawaiian → American ×13) plus the known
Contemporary↔American blur (7+3); very few wrong-region errors. ~77 output tokens/recipe, 439 in ~9 min on the M5 — a free,
local teacher within 0.035 of Sonnet, which matters for SaffronSifu volume later. Caveat (all rows): teacher chosen BY gold
score, so this table mildly overfits the 439; the choice would need a held-out slice to be clean.
Decision: Sonnet 5.5 is the teacher for the next silver (origin only; Jev keeps `contains`). Dataset → `data/dataset_sonnet`.

### Arm C result (2026-11-08) — 2B GRPO from base, reward v2, 439 audited gold
**0.517** (step-22 policy, evaluated with `uv run eval` on the pod against vLLM). Base was 0.244 → **+0.27 from reward alone**.
Gold-150 curve during training: 0.252 → 0.467 → 0.440 (the first 150 of eval.jsonl are harder than the full 439).
Trajectory: train reward 0.18 → ~0.50 by step 13, flat after; entropy 0.39 → 0.03 by step 22 (premature convergence);
mismatch-KL 0.005–0.02 throughout (staleness 2 held); no reward hacking under v2; truncation ≤3%.
Reading: the reward can teach format + vocabulary from scratch, but once all 8 rollouts per prompt agree the advantage is zero
and learning stops — at ~0.5 train / 0.52 gold, far below SFT (0.782). RL-from-base is not the production path for this task.
Arm D (from SFT) must keep exploration alive: rollout temperature ↑ (1.2), consider `trainer.loss.adv_tau` > 1.
Checkpoint note: prime-rl `checkpoints/step_N` is trainer-format (DCP); vLLM-loadable weights are in `broadcasts/step_N`
(safetensors + index only) — assemble a model dir by adding config/tokenizer files from the base HF snapshot, keeping the
broadcast's own `model.safetensors.index.json`. Saved as `~/prime-intellect/models/grpo-base-v2-step22` on the pod.
The supported route (found 2026-11-09, prime-rl commit cbd3702): `uv run python tools/convert_dcp_to_bf16.py <run>/checkpoints/step_N`
→ `<run>/checkpoints/step_N/weights` with config.json + tokenizer + safetensors, loadable by `uv run inference --vllm.model`.
SFT runs never write HF weights or broadcasts (broadcasts exist only for online eval and need a live receiver), so for SFT
this converter is the *only* export path; a `[ckpt]` block (even empty) is what makes the final DCP save happen. Second half of
the converter bug: with `_tied_weights_keys` None the tied `lm_head.weight` is never dropped, so the export carries a duplicate of
`embed_tokens`. vLLM loads it anyway; mlx-lm refuses ("parameters not in model: language_model.lm_head.weight"). Proper fix for the
tool: pop `lm_head.weight` when `tie_word_embeddings`; local fix for an existing export: drop the tensor from its shard + index. The
broadcast-assembly trick above is only for RL runs, where `broadcasts/step_N` already exists.
Breakdown (prime-rl eval metrics): origin 0.524, contains 0.510, primary_depth **1.32** (gold ≈ 2.6), influence_pred **0.0**,
evidence_coverage 1.00, unsupported_evidence 0.05, gate_fail 0.7%, unknown_classes 0.016, response ~190 chars.
Reward-design lessons from arm C: (1) hierarchical partial credit makes SHALLOW paths a safe local optimum — the ×1.1 depth
bonus does not pay enough for committing; (2) zero influences confirms RL-against-silver inherits silver's suppression;
(3) grounding works: every claim carries evidence and fabrication is rare under v2.
