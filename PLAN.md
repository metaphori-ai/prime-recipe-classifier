# Prime Intellect — verifiers / prime-rl learning project

**Status:** planning · task chosen (SaffronSifu recipe classification) · taxonomy v0.3 (tradition spine) drafted
**Budget posture:** small models only (Qwen3.5 0.8B–4B class); target < $200 total for phases 0–3

## Goal

Go from zero hands-on to: one published `verifiers` environment, several real GRPO runs
on Lab, a trajectory-level analysis of what the model learned vs. what it exploited,
and a write-up that reads as "a run that failed, and what the evidence changed."

The qualitative half (reading trajectories, spotting rubric exploits, rubric design)
is already strong. The gap is the operational half: the stack, the instrument panel,
the loop. Optimize the plan for reps on that half.

## Task selection criteria

- Base small model scores **20–60 %** on the eval before any training (GRPO needs
  within-group variance; 0 % or 95 % = no signal). Measure this first.
- Reward is **part verifiable, part judged** — watch where they disagree.
- **Short episodes** (≤ ~1.5k output tokens) → runs cost $10–25, iterate in hours.
- Rubric has a **known exploitable hole** so reward hacking is observable, on purpose.
- **Not already on the Environments Hub** → publishable.

## Candidate tasks

1. **Constrained rewriting** — hard constraints verifiable (length, reading level,
   banned terms) + judge for meaning preservation. Hacking appears in run 1.
2. **Recipe → controlled taxonomy** (SaffronSifu) — origin paths w/ primary/influence roles,
   dietary pattern (rule verifier), dietary restrictions (asymmetric safety penalty).
   See `taxonomy.yaml`, `TAXONOMY.md`. Real product need; model output goes back into the app.
3. **Multi-turn tool task** — small local knowledge base behind a search tool,
   verifiable answer. Better as environment #2.
4. **Format-grammar compliance** — owned grammar's parser as free verifier + judge for
   semantic quality. Better as environment #2.

**Decision:** **#2 (extraction) first** — clean verifiable loop, learn the panel on trusted plumbing.
**Then #1 (constrained rewriting)** as the real study: layered reward (verifiable band gate → judge signal), graded partial credit early, cheap-vs-strong judge comparison, track verifier/judge disagreement rate as the primary curve.
#3 / #4 after.

## Phases

| # | Phase | Where it runs | Est. cost |
|---|-------|---------------|-----------|
| 0 | Toolchain: `uv` + Python ≥3.11, `verifiers`, `prime` CLI, Prime/OpenRouter key | your terminal | $0 |
| 1 | Build environment (dataset, rollout, rubric) | Claude workspace → this folder | $0 |
| 2 | Baseline evals on 2–3 small models; pick the one in the 20–60 % band | your terminal (API) | $5–20 |
| 3 | GRPO runs on Lab (hosted); 3–6 runs varying rubric / group size / LR | Lab via `prime` CLI | $50–150 |
| 4 | Trajectory analysis: before/after samples, exploit hunt, verifier-vs-judge disagreement | Claude workspace | $0 |
| 5 | Publish env to Hub + write-up | your terminal | $0 |
| 6 | *(optional)* self-hosted `prime-rl` on 2× H100 to see the trainer's panel directly | rented GPUs | $75–175 |

## Instrument panel to learn (phase 3/6)

reward (mean, per-group variance) · KL to reference · entropy · response length ·
clip fraction · grad norm · verifier-vs-judge agreement rate

Known pathological shapes to recognize: entropy collapse, length exploitation,
reward↑ with judge↓ (hacking), KL runaway, loss spike after N steps.

## Division of labor

- **Claude:** environment code, rubric, configs, log/trajectory parsing, curve reading,
  write-up drafts. Everything that's CPU Python lives here.
- **You:** anything that touches Prime's API (CLI, Lab jobs, inference) — not reachable
  from Claude's sandboxes. Paste or drop logs/trajectory dumps into this folder.

## Environment notes

- Device sandbox: Python 3.10, `uv 0.13.4`, PyPI reachable, `api.primeintellect.ai` **not** reachable.
- `verifiers` requires Python ≥ 3.11 → `uv python install 3.12`.

## Open questions

- Task choice (see candidates).
- Which judge model for the rubric half — a hosted small model (cheap, gameable) or a
  frontier model via gateway (expensive, stricter)? Possibly both, to compare.
- Lab's current model list for hosted training — confirm Qwen3.5 sizes available.

## Label QA checklist (before any training run)

1. **Gold is right** — owner labels the 400 candidates *against the file*, not instinct (Hyderabadi → Deccan).
2. **Gold is consistent** — relabel a random 50 blind a week later (or second labeler). Disagreement rate is
   the noise floor and the ceiling on measurable reward. Record it in LOG.md.
3. **Silver is audited** — spot-check 50–100 silver labels. Random noise OK; *systematic* bias (always stops
   at national tier, always misses fish sauce) is not — the student learns the habit. Record error rate.
4. **No leakage** — no gold ids in silver; dedupe on content (normalized title + first 5 ingredients), not id.
   Same recipe from two sources is the common failure.

## Baseline (2026-11-07) — qwen3.5:2b, local Ollama, 50 eval recipes, temp 0.7, thinking off

| reward design | mean | rollouts > 0 | ≥ 0.5 | = 1.0 |
|---|---|---|---|---|
| hard gate (vocabulary slip → 0) | 0.133 | 10/50 | 8 | 3 |
| soft gate (resolve-by-name ×0.8; unknown class = spurious claim) | **0.346** | 41/50 | 20 | 3 |

Same traces, rescored: the +0.21 is **reward shaping, not the model**. Components under soft gate: origin 0.44, contains 0.23.
Dominant failure: raw ingredients emitted as classes (31/50 rollouts, ~5.5 each). Secondary: macro tier dropped (10/50, resolved by name).
**Decision: qwen3.5 2B is the training candidate** (in band). Run 0.8B and 4B locally for the size curve before committing.
Config lessons: thinking models burn `max_tokens` on reasoning → `reasoning_effort = "none"`; default agent is bash+prime sandbox → `null` harness + `subprocess` runtime.

## Teacher selection for origin silver (2026-11-08) — 439 gold recipes, v2 prompt

| teacher | primary exact | origin score | unusable |
|---|---|---|---|
| qwen/qwen3.6-35b-a3b | 67.4 % | 0.761 | 19 (parser bug, now fixed) |
| deepseek/deepseek-v4-flash-0731 | 68.5 % | 0.784 | 1 |
| **both agree (68 % of recipes)** | **82.9 %** | **0.879** | — |
| disagree (32 %) | ~35 % | ~0.55 | — |

Decision: run BOTH on the corpus (~$3), keep agreements as origin silver (~4.5k recipes), drop disagreements.
Contains third-reader check: Jev 99.86 % tile accuracy vs gold; DeepSeek 95.4 %; Qwen 94.1 % → Jev stays the contains labeler.
Prompt v1 → v2 (conventions block) moved Qwen 54.5 % → 67.4 % at zero cost; v3 adds Chifa/Nikkei/Cape Malay dish lines.

## Size curve (2026-11-08) — local Ollama, 50 eval recipes each, v3 prompt, soft gate, thinking off

| model | reward | origin | contains | unknown classes / rollout | perfect |
|---|---|---|---|---|---|
| qwen3.5 0.8B | 0.233 | 0.26 | 0.20 | 2.9 | 1 |
| qwen3.5 2B | 0.357 | 0.47 | 0.22 | 2.7 | 2 |
| qwen3.5 4B | 0.467 | 0.61 | 0.29 | 0.18 | 3 |
| qwen3.5 9B | 0.491 (0.569 on another 50) | 0.58 | 0.39 | 0.30 | 5 |
| qwen3.6 27B | **0.794** | 0.87 | 0.70 | 0.00 | 16 |

- Vocabulary-compliance step between 2B and 4B (raw ingredients emitted as classes: 2.7 → 0.18 per rollout).
- Contains is the weak half at every size; the recall-weighted, evidence-checked metric is strict by design.
- n=50 noise is ±0.04 (two 9B runs). Post-training comparisons use all 439 eval recipes.
- **qwen3.6 27B local ≥ hosted teachers** → first choice for any future silver labeling (free, no rate limits).
- **Decision: train 2B (the RL lesson: reward must teach vocabulary + judgment) and 4B (product candidate: judgment only).**

## Data freeze (2026-11-08)
- `data/dataset/eval.jsonl` — 439 human-verified (never trained on)
- `data/dataset/train.jsonl` — 3,252 (two-teacher-agreed origin + Jev contains; American capped at 300). Uncapped: 4,140 (`--cap-primary 0`).
