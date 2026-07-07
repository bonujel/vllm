# Handoff — Reproducible Sampling (gonka #1199)

Status snapshot for handing off the deterministic-sampling / inference-validation
work. Read `README.md` in this folder for the doc index; this file is the
"where we are / what's next".

Issue: https://github.com/gonka-ai/gonka/issues/1199

## One-liner

The cross-language core (Python executor ↔ Go chain validator producing
**bit-identical** results) is **built and proven offline**. The remaining work
is integration into the live paths, which needs a GPU and two pending decisions.

## Branches

Both repos: branch **`bonujel/deterministic_sampling_merged`**.
- vLLM: `gonka-ai/vllm` fork, branched from `tg/deterministic_sampling_merged` (`bd07b130b`).
- gonka: branched from `main` (`3544ca762`). Worked in a git worktree at
  `~/gonka-wt/bonujel-deterministic_sampling_merged` (main checkout untouched).

**Nothing is pushed.** All commits are local.

## What's done (verified offline, no GPU)

### Cross-language contract + parity — DONE
- **Contract** pinned: `docs/DETERMINISTIC_SAMPLING_CONTRACT.md` (v1.0.0) — the
  exact logprobs→weights→token rules + chain-bound seed (§8) all three parties
  (vLLM executor, Python validator, Go validator) must reproduce bit-for-bit.
- **Golden vectors**: `tests/v1/validation/conformance_vectors.json`, generated
  by `scripts/gen_conformance_vectors.py`. Cover sampling cases, RNG reference,
  `uint64_below` (non-power-of-2), and seed derivation.
- **Python** (drift guards): `test_conformance_vectors.py`,
  `test_chain_bound_seed_domain.py`.
- **Go** (`decentralized-api/internal/validation/detsample/`, stdlib + apd only,
  offline-testable — 17 tests green):
  - `rng.go` — SHA256 counter RNG, unbiased categorical sampler (§5/§6).
  - `pipeline.go` — decimal logprobs→weights via cockroachdb/apd (§4).
  - `seed.go` — `DeriveChainBoundSeed` (§8).
  - `verify.go` — `VerifyPosition` → 3-valued verdict (Honest/Fraud/Inconclusive)
    with version gating + greedy/non-positive-temperature exemption.

Proven bit-identical Python↔Go across all cases: integer weights (incl. softmax
`exp()`), sampled token, and the chain-bound seed digest. This closes the
review's #2 risk (S3, "cross-language parity never tested").

### Failure taxonomy + version gating — DONE
`VerifyPosition` returns Honest / Fraud / **Inconclusive**. A version skew,
greedy position, or validator-side replay error is Inconclusive, never Fraud —
so an honest executor is not punished for a validator-side or version problem
(the review's "binary verdict" finding).

### Chain-bound seed primitive — DONE (from gonka-ai/vllm#56)
`derive_chain_bound_seed` (Python) + `DeriveChainBoundSeed` (Go), byte-identical.
Purely additive: **nothing calls it in the generation path yet** (that is S1
wiring, below).

## Commits

vLLM `bonujel/deterministic_sampling_merged` (on `bd07b130b`):
```
95188e5cd conformance: add uint64_below vectors; stop reaching into private tag
fed7db8b5 docs: gather reproducible-sampling dev notes
f177d5bc3 contract: record open items (token-order, sequence RNG semantics)
889128a03 Add chain-bound seed derivation (#56) + seed conformance vectors
8e355d862 Add cross-language deterministic-sampling contract + conformance vectors
```
gonka `bonujel/deterministic_sampling_merged` (on `3544ca762`):
```
e0ea16b2c close review gaps (rejection-path coverage, temp guard)
28f8cfd16 validator-facing replay API with three-valued verdict
fb30b00da Go chain-bound seed derivation + cross-language parity
3a970bc92 repoint conformance-replay pointer to ./detsample
e77c51e29 Go decimal pipeline — full cross-language parity proven
c1b0f806f Go RNG + integer categorical sampler (§5/§6)
577f88860 add conformance vectors + Go harness
```

## What's to-do

```
├─ Core math cross-language          ✅ DONE + tested
├─ Failure taxonomy + version gating ✅ DONE
├─ Go logic into real chain validator 🟡 public API done; wire into
│                                        inference_validation.go (GPU/network)
├─ Seed into executor (S1 wiring)    ⬜ GPU + inference_id transport + protocol
├─ Executor → decimal weights (E1)   ⬜ GPU — biggest remaining block (de-risked)
├─ Executor emits artifact + orchestration ⬜ GPU (serving_chat.py)
├─ Stage 2 distance-threshold calibration  ⬜ real data
└─ Rollout: log-only → measure → enforce   ⬜ last step
```

Recommended order once a GPU is available:
1. **E1** — switch the executor weight path (`topk_topp_sampler.py:317`,
   `(probs*2^16).round()` on GPU float32) to the decimal pipeline (from logprobs).
   Land as shadow mode first (compute both, log divergence), verify on a small
   model. Parity is guaranteed by construction (same pipeline as the validator);
   GPU is for integration, not discovery.
2. **S1 wiring** — replace the seed derivation in `gpu_model_runner.py:708`
   (`f"{seed}|{prompt_repr}"`) with `derive_chain_bound_seed`; needs the chain
   `inference_id` delivered (see decisions).
3. **Wire the validator** — call `detsample.VerifyPosition` from
   `decentralized-api/internal/validation/inference_validation.go`.
4. **serving_chat orchestration** + response fields; then log-only rollout →
   measure false-reject rate → enforce.

## Decisions pending (not code — need you / the team)

1. **inference_id transport**: which field/channel carries the chain
   `inference_id` into vLLM (request field / SamplingParams / header)? Gates S1.
2. **Seed hardening**: also bind `executor_addr` + `start_block_hash` to close
   residual `user_seed` grinding? (#56 binds inference_id only.)
3. **Token order**: converge the legacy numeric-id sort in `validation_logic.py`
   onto the contract's lexicographic string order (latent false-reject).
4. **Sequence RNG semantics**: one RNG stream across a sequence vs one seed per
   position — unpinned; needs a decision + a sequence-level conformance vector.
   (See contract "Open items".)

Also housekeeping: run `go mod tidy` on a networked machine to promote `apd/v2`
from indirect to direct dep.

## How to verify

```bash
# Go (offline, no torch/GPU) — 17 tests
cd decentralized-api && GOPROXY=off go test ./internal/validation/detsample/ -v

# Python drift guards need pytest + the module import; the module itself is
# stdlib-only and can be loaded by path (see gen_conformance_vectors.py).
python scripts/gen_conformance_vectors.py   # regenerates vectors deterministically
```

## Key verified values

- `iter_u64("reference_seed_v1", 5)[0]` == `4286832458236889005`
- `derive_chain_bound_seed(7, "chain-abc")` ==
  `910b688db5b2061e66385acf0ee665682d5e01bab5d1d8d2cdde9a2612a6e6c2`
- Integer weights always sum to exactly `2^16 = 65536`.
