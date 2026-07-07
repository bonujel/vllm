# Reproducible Sampling for Inference Validation (gonka #1199)

Landing page for the deterministic-sampling / inference-validation work. Notes
here are working documents; the shipped code + contract live in the code tree
(see "Code deliverables" below).

Issue: https://github.com/gonka-ai/gonka/issues/1199

## Documents in this folder

| File | What it is |
|---|---|
| `ISSUE-1199-讲解.md` | Plain-language explainer of the whole scheme (for onboarding a teammate). |
| `ISSUE-1199-REVIEW-REPORT.md` | The review deliverable, 6-review-point format (bilingual). Answers the issue's six review points + the merge addendum. |
| `ISSUE-1199-REVIEW.md` | Same review, three-layer format (Summary → Current state → Findings → per-point). |
| `ISSUE-1199-HANDOFF-ADDENDUM.md` | Early handoff addendum grounding the original handoff against the real branches. |

The team's posted review on the issue (by @Ryanchen911, @bonujel) is an expanded
version of `ISSUE-1199-REVIEW-REPORT.md`.

## Code deliverables (NOT in this folder — they live with the code)

Branch `bonujel/deterministic_sampling_merged` in both repos.

**vLLM** (`gonka-ai/vllm` fork):
- `docs/DETERMINISTIC_SAMPLING_CONTRACT.md` — the authoritative cross-language
  contract (v1.0.0) + seed-derivation appendix (§8) + open items.
- `vllm/v1/sample/deterministic_utils.py` — RNG, decimal pipeline, and
  `derive_chain_bound_seed` (from gonka-ai/vllm#56).
- `scripts/gen_conformance_vectors.py` → `tests/v1/validation/conformance_vectors.json`
  — the golden vectors (executable form of the contract).
- `tests/v1/validation/test_conformance_vectors.py`,
  `tests/v1/sample/test_chain_bound_seed_domain.py` — Python drift guards.

**gonka** (`decentralized-api/internal/validation/detsample/`, stdlib+apd only,
offline-testable):
- `rng.go` — Sha256CounterRNG + integer categorical sampler (contract §5/§6).
- `pipeline.go` — decimal logprobs→weights via cockroachdb/apd (contract §4).
- `seed.go` — DeriveChainBoundSeed (contract §8).
- `verify.go` — VerifyPosition → three-valued Verdict (Honest/Fraud/Inconclusive)
  with version gating + greedy exemption.
- `*_test.go` — cross-language parity gate against the shared vectors.
- `../testdata/conformance_vectors.json`, `../DETERMINISTIC_SAMPLING_CONTRACT.md`
  — mirrored fixtures at the gonka↔vLLM boundary.

## Progress

```
Stage 1 "reproducible sampling" rollout
├─ Core math cross-language          ✅ done + tested (bit-identical Python↔Go)
│   ├─ contract pinned               ✅
│   ├─ sampling pipeline             ✅
│   └─ chain-bound seed              ✅
├─ Failure taxonomy + version gating ✅ done (VerifyPosition, 3-valued verdict)
├─ Go logic into real chain validator 🟡 public API done; thin wiring into
│                                        inference_validation.go = GPU/network step
├─ Seed into executor (S1 wiring)    ⬜ GPU + inference_id transport + protocol
├─ Executor → decimal weights (E1)   ⬜ GPU (biggest remaining block; de-risked)
├─ Executor emits artifact + orchestration ⬜ GPU
├─ Stage 2 distance-threshold calibration  ⬜ real data
└─ Rollout: log-only → measure → enforce   ⬜ last step
```

## Open items (decisions pending — see contract §"Open items")

1. **inference_id transport**: which field/channel carries the chain
   `inference_id` into vLLM (request field / SamplingParams / header).
2. **Seed hardening**: whether to also bind `executor_addr` + `start_block_hash`
   to close residual user_seed grinding.
3. **Token order**: converge the legacy numeric-id sort in `validation_logic.py`
   onto the contract's lexicographic string order.
4. **Sequence RNG semantics**: one RNG stream across the sequence vs one seed per
   position — not yet pinned; needs a sequence-level conformance vector.

## Key facts verified locally (no GPU)

- Python↔Go bit-identical across all conformance cases for: integer weights
  (incl. softmax `exp()`), sampled token, and the chain-bound seed digest.
- `derive_chain_bound_seed(7, "chain-abc")` ==
  `910b688db5b2061e66385acf0ee665682d5e01bab5d1d8d2cdde9a2612a6e6c2`.
- RNG reference: `iter_u64("reference_seed_v1", 5)[0]` == `4286832458236889005`.
