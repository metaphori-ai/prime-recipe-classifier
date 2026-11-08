# Macro · Meso · Micro — what we are doing, in the native terminology

Each term is given three ways:
**human** — a pattern people already do · **bridge** — halfway · **native** — how the field says it.

---

## MACRO — the shape of the whole thing

**Post-training**
- human: an apprentice who already knows the trade being coached on one shop's way of doing it
- bridge: adjusting an already-capable model's habits, not teaching it language
- native: any training applied after pretraining — SFT, RL, DPO — to shape behavior rather than knowledge

**Policy**
- human: a person's current way of responding to situations
- bridge: the model, viewed as "given this input, what does it tend to produce"
- native: π(a|s) — the probability distribution over outputs the model assigns, given a prompt; the thing being optimized

**Environment**
- human: the dojo — a place with a task, rules, and someone who says whether you did it
- bridge: a packaged task: data to practice on, how a turn works, how it's scored
- native: a `verifiers` Environment: dataset + rollout protocol + rubric; exposes `rollout()` and returns rewards

**Rollout / Trajectory**
- human: one attempt at the task, start to finish, as it actually happened
- bridge: one sampled answer (or one multi-turn episode), kept verbatim with its score
- native: a sampled sequence of tokens (and tool calls) from the policy on one prompt, with its reward

**Reward**
- human: the mentor's verdict on that attempt
- bridge: a number per rollout saying how good it was, by a rule or a judge
- native: scalar r ∈ ℝ assigned per trajectory by the rubric's reward functions; the only signal RL learns from

**The loop (sample → score → update)**
- human: practice, get feedback, adjust, repeat
- bridge: generate many attempts per prompt, score them, nudge the model toward its own better attempts
- native: for each batch: policy samples G rollouts per prompt → rubric scores → advantages computed → gradient step on the policy → repeat

**GRPO** (Group Relative Policy Optimization)
- human: grading on a curve within one classroom, not against an absolute standard
- bridge: for each prompt, compare the model's own attempts against each other; push toward the above-average ones
- native: advantage = (r − mean_group) / std_group over G samples of the same prompt; no value network; PPO-style clipped objective + KL penalty to a reference

---

## MESO — the parts and where they live

**verifiers**
- human: a standard form for writing a dojo so any dojo can plug into any training hall
- bridge: Prime Intellect's Python library for defining environments and evals
- native: `verifiers` — Environment classes (SingleTurn, MultiTurn, Tool), Parser, Rubric; environments are installable Python packages

**Rubric / Reward functions**
- human: the grading sheet — several criteria, each worth something
- bridge: a list of functions, each scoring one aspect, combined by weights
- native: `Rubric(funcs=[...], weights=[...])`; each func maps (prompt, completion, answer, state) → float

**Verifier vs Judge**
- human: a checklist vs. a connoisseur
- bridge: a rule that can prove the answer (parser, taxonomy lookup) vs. a model asked to assess quality
- native: deterministic reward function vs. LLM-as-judge reward (another model prompted with a rubric); the two can disagree, and the disagreement rate is a diagnostic

**Parser**
- human: the clerk who reads the form before anyone grades it
- bridge: extracts the structured answer from the model's text; if it can't, score is zero
- native: `Parser.parse_answer(completion)`; often enforces a format (XML tags, JSON) and feeds the rubric

**Dataset — gold / silver**
- human: the exam everyone trusts vs. the practice problems
- bridge: a small human-verified set used only to measure, a larger model-labeled set used to train
- native: eval split (held out, never trained on) vs. train split (may be noisy, distilled from a stronger model)

**Baseline**
- human: how you did before any coaching
- bridge: the untrained model's score on the gold set; the number every run is compared to
- native: zero-shot eval of the base checkpoint on the held-out set; must sit in the 20–60 % band for GRPO to have signal

**Environments Hub**
- human: the public library of dojos
- bridge: where environments are published and installed by name
- native: hub.primeintellect.ai; `prime env push` / `prime env install owner/name`

**prime-rl**
- human: the training hall itself — the building where the loop runs
- bridge: the trainer that runs the loop at scale: samples with one engine, trains with another, asynchronously
- native: async RL framework: orchestrator (rollouts via vLLM) + trainer (FSDP2) + weight broadcast; off-policy by design

**Lab (hosted training)**
- human: renting the hall by the hour with staff included
- bridge: Prime runs prime-rl for you; you supply the environment and a config; pay per token
- native: hosted post-training; LoRA adapters on Qwen-class models; billed per input/output/training token

**LoRA**
- human: learning a new accent rather than a new language
- bridge: train a small add-on instead of the whole model; cheap, swappable
- native: low-rank adapters on attention/MLP weights; base frozen; adapter merged or served alongside

**Sandbox**
- human: a padded room where the apprentice can run tools without breaking the shop
- bridge: isolated container where tool calls (code execution, search) actually run during a rollout
- native: Prime Sandboxes — microVMs; used by Tool environments; not needed for env #1

---

## MICRO — the instrument panel and what moves it

**Group / Group size (G)**
- human: how many attempts at the same problem before the mentor compares them
- bridge: rollouts per prompt; more = better comparison, more cost
- native: G samples per prompt per step (e.g. 8–16); advantage normalization is within-group

**Advantage**
- human: "better or worse than you usually do on this one"
- bridge: a rollout's score relative to its siblings; positive = reinforce, negative = suppress
- native: A_i = (r_i − μ_G) / σ_G; if all G rewards are equal, A = 0 and the prompt teaches nothing (the variance problem)

**Reference model / KL divergence**
- human: a tether to who you were, so coaching doesn't turn you into someone else
- bridge: a frozen copy of the starting model; the trainer is penalized for drifting too far from it
- native: KL(π_θ ‖ π_ref) added to the loss with coefficient β; rising KL = the policy is moving; runaway KL = it's leaving the manifold

**Entropy**
- human: how many ways you're still willing to try something
- bridge: how spread-out the model's output distribution is; collapse = it has stopped exploring
- native: −Σ p log p over the token distribution; entropy collapse → identical rollouts → zero advantage → learning stalls or locks in a hack

**Clip ratio / Clip fraction**
- human: a speed limit on how much one lesson can change you
- bridge: cap on how far a single update can move the probability of any token
- native: PPO ratio r = π_θ/π_old clipped to [1−ε, 1+ε]; clip fraction = share of tokens hitting the cap; high = steps too aggressive or policy too stale

**Response length**
- human: whether the apprentice is getting wordier or terser
- bridge: mean tokens per rollout over time; drifts tell you what the reward is secretly paying for
- native: mean completion length per step; length exploitation = reward correlated with length without the rubric intending it

**Learning rate / Step**
- human: how hard each lesson lands, and how many lessons
- bridge: step size of each update; total number of updates
- native: lr (e.g. 1e-6 for full FT, 1e-5–1e-4 for LoRA); step = one batch of prompts × G rollouts → one gradient update

**Off-policy / Staleness**
- human: being graded on last week's attempts after you've already changed
- bridge: in async training, rollouts come from a slightly older policy than the one being updated
- native: samples generated by π_old k steps behind π_θ; corrected by importance ratios; too stale → instability

**Reward hacking (Goodhart)**
- human: studying the grader instead of the subject
- bridge: the model finds a way to raise the score that isn't the behavior you wanted
- native: policy exploits a proxy reward; reward ↑ while the intended metric (judge / held-out eval) is flat or ↓; the central failure mode of RL on learned or imperfect rewards

**Signal / Variance**
- human: a lesson only teaches if some attempts were better than others
- bridge: GRPO needs mixed outcomes within a group; all-pass or all-fail prompts are dead weight
- native: within-group reward variance > 0; prompts with σ_G = 0 contribute zero gradient; curriculum = keeping prompts in the band where σ_G is high

**Verifier–judge disagreement**
- human: the checklist says pass, the connoisseur says no
- bridge: the rate at which rule-based and model-based scores diverge; rising = the model is satisfying the letter
- native: |r_verifier − r_judge| tracked per step or on the eval set; the primary curve for env #1 (constrained rewriting), secondary for env #2

---

## How the three layers connect

MACRO says *what the loop is*. MESO says *which boxes we fill* (environment, rubric, dataset, trainer). MICRO says *what to watch while it runs*. Our order of work is MESO-first (taxonomy → environment → gold set → baseline), then MACRO runs it, then MICRO is how we read what happened.
