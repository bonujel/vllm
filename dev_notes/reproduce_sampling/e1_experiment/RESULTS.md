# E1 shadow experiment — results

**Question:** does the current float executor weight path (E1) actually diverge
from the decimal validator on real model logprobs, and does switching the
executor to the decimal pipeline fix it?

**Setup:** run on an 8×RTX-4090 box (idle GPU, `qwen` conda env: torch 2.8+cu128,
transformers 4.55). Real logits from `gpt2` and `Qwen/Qwen2.5-0.5B`. Per position:
sample a token with the executor's weights and with the validator's weights under
the **same** SHA256 RNG; a mismatch = a false reject of an honest position.

## Headline result

| Executor | vs decimal validator (false-reject) |
|---|---|
| **Current float** `(probs*2^16).round()` | **14–77%** (varies by model / temp / top_k / top_p) |
| **E1 fix** — decimal pipeline (`logprobs_to_weights`) | **0/128, every configuration** |

`e1_shadow.py` (weight-computation replica), 128 steps each:

```
# gpt2
temp=0.7 topk=20            FLOAT 37.5% | DECIMAL 0 | float weight-sum [65531,65539]
temp=1.0 topk=20            FLOAT 44.5% | DECIMAL 0 | float weight-sum [65532,65540]
temp=1.0 topk=40 top_p=0.9  FLOAT 53.1% | DECIMAL 0
temp=0.7 topk=40 top_p=0.95 FLOAT 37.5% | DECIMAL 0
# Qwen2.5-0.5B (different tokenizer, ~151k vocab)
temp=0.7 topk=20            FLOAT 21.1% | DECIMAL 0
temp=1.0 topk=20            FLOAT 41.4% | DECIMAL 0
temp=1.0 topk=40 top_p=0.9  FLOAT 25.8% | DECIMAL 0
temp=0.7 topk=40 top_p=0.95 FLOAT 14.1% | DECIMAL 0
```

`e1_patch_verify.py` (uses the **actual** `deterministic_sample` logic verbatim
for the float path, and the proposed E1 patch for the decimal path):

```
gpt2 temp=1.0 top_k=40   FLOAT exec vs validator 76.6% | E1 exec vs validator 0/128
gpt2 temp=0.7 top_k=40   FLOAT exec vs validator 53.1% | E1 exec vs validator 0/128
```

## Why the float path fails (mechanism)

The float weights sum to `[65530, 65541]`, **never exactly 65536**. Because the
sum ≠ 2^16, `uint64_below(rng, sum)` reduces the *same* drawn u64 by a different
modulus than the validator's 65536 — and since the u64 is ~2^63, `u64 % sum_float`
and `u64 % 65536` are effectively **uncorrelated**. So the executor and validator
draw independent tokens ~40% of the time. Fine per-token weight rounding matters
too, but the non-integer total is the dominant cause. Only the decimal pipeline
guarantees the weights sum to exactly 2^16, which is why the executor must adopt
it.

## Decision-relevant conclusions

- **E1 is a hard blocker, empirically:** a decimal validator false-rejects
  14–77% of honest positions from the current float executor. Enforcing before
  E1 lands would slash honest participants at a massive rate.
- **The E1 fix works, verified end-to-end on real model output:** once the
  executor computes weights via the decimal pipeline, executor == validator on
  every position (0 false rejects), across two models, four temp/top_k/top_p
  configurations.
- Feeds **D8** (enforcement gating): the shadow false-reject rate must be ≈0
  before enforcing, which requires E1 first.

## The E1 patch (proposed)

`deterministic_sample_e1` in `e1_patch_verify.py` is the proposed replacement for
`vllm/v1/sample/ops/topk_topp_sampler.py::deterministic_sample`. It changes the
input from `probs` to `logits + sampling params`, computes `log_softmax`, and runs
`logprobs_to_weights` over the top-`support_k` logprobs.

**Provisional / open:**
- `support_k` (how many tokens' logprobs enter the pipeline and the artifact) is
  set to 64 as a placeholder — the real value is the signed **support set (D2)**.
- The **caller** (`TopKTopPSampler.forward` / `sample`) must be updated to pass
  `logits` + per-request `temperature/top_p/top_k/min_p` instead of `probs`.
- Full end-to-end in a running vLLM (rather than the function-level verification
  here) is the remaining heavier step, best done after D2 is fixed so the
  artifact format doesn't change twice.
