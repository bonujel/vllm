# Reproducible sampling (#1199) — work summary

One-file consolidation of the review, the code changes, and the GPU experiments.
Branches: `bonujel/vllm@deterministic-sampling-clean`,
`bonujel/gonka@bonujel/deterministic_sampling_merged`.

## Problem

Two-stage inference validation. Stage-1 (sampling replay) re-derives each
position's token from the signed logprobs + seed and checks it matches; Stage-2
(distance) recomputes logprobs and checks distribution distance. For Stage-1 to
work, the executor (vLLM) and the validators (Python + Go chain) must reduce the
same logprobs to the same token, bit-for-bit, in both languages.

## Code changes

vLLM side (`deterministic-sampling-clean`):
- Executor sampling (`topk_topp_sampler.deterministic_sample`) now uses the
  decimal weight pipeline over the raw logprobs, canonical string token order,
  and a per-position seed `base|pos` — the identical computation the validator
  replays. Replaces float `(probs*2^16).round()` + numeric order + single stream.
- Python validator (`validation_sampling`): sequence-level `verify_sequence`
  (three-valued Honest/Fraud/Inconclusive), `mae_distance` (Stage-2 = MAE over the
  signed support), unbounded-support and seed-domain gates. The serving path
  (`serving_chat`) now calls this; the legacy recompute path
  (`validation_logic`) is retired from serving.
- `sample_categorical_weights` raises on zero-weight input instead of a silent
  fallback (matches the Go validator).

gonka side (`detsample`): Go `VerifySequence` + `MAEDistance` mirror the Python
sequence layer. A shared `conformance_vectors.json` (primitives + sequence +
distance) proves Python↔Go are byte-identical.

## Experiments (GPU, gpt2 / Qwen2.5 on RTX 4090; HF proxy unless noted)

| # | Question | Result |
|---|---|---|
| 1 | float executor false-reject? | 14–77%; decimal pipeline 0/128 |
| 2 | signed support size? | fixed top-256 distorts 7–33% -> sign the filter's kept set |
| 3 | can validator recompute logprobs? | no; batching alone flips 0–8% -> replay signed logprobs |
| 4 | Stage-2 distance metric? | MAE/KL over top-K separates honest vs wrong model ~40–65x; sampled-token delta overlaps |
| 5 | quantized same model = fraud? | precision ladder: int4 caught, int8 gray zone |
| 6 | drift from hardware or batch? | 8 cards bit-identical -> batch-shape only; cross-arch untested |
| 7 | filter-edge rules load-bearing? | boundary hit on 5–46% of positions |
| 8 | E1 live in real vLLM (not proxy)? | 70% divergence = E1 + numeric order combined |
| decomp | which executor fix works? | E1-only 39–59%, order-only 32–57%, both together 0% |
| e2e | full flow in real vLLM | current executor artifact -> FRAUD; after the fix -> HONEST 24/24 |
| tok/perf | tokenizer + speed | same tokenizer reload-stable; decimal ~10–142x float |

Headline: the executor fix (decimal + string order + per-position seed) takes
executor-vs-validator divergence from ~70% to 0, verified end-to-end in a real
vLLM engine (honest inference -> HONEST verdict). `pytest tests/v1/validation
tests/v1/sample` = 45 passed on GPU.

## Status

Done: validator convergence (three-valued, MAE, replay); Python↔Go parity
(primitives + sequence, shared vectors); S2 local decimal context; token order,
temp-0, filter-edge/kept-set, unbounded-support handling; §6.9 raise; seed-domain
gate; executor E1 + §6.6 + per-position seed, verified live.

Open — ours: 3 data items (distance det-vs-baseline, Go perf bench, tokenizer
cross-version); delete `validation_logic.py`; min_p in the deterministic path.

Open — needs a gonka decision: Decision A (signed-field contract + signature
verification), D10 (`inference_id` transport into vLLM, then chain-bound seed
wiring), D2 (signed support size; `DETERMINISTIC_SUPPORT_K=64` is a placeholder),
Decision D (distance threshold value).

Open — infra: cross-architecture reproducibility CI (needs A100/H100 + multi-node);
real production model/backend re-verification; on-chain slashing / verdict
handling.
