# E4 — cross-card & run-to-run logit determinism

**Question:** E3 showed batch-*shape* change perturbs logits. Are the other two
suspects — run-to-run (same card) and card-to-card (different physical GPU) — also
sources of drift? This isolates them at fixed batch=1 / fixed shape.

**Setup:** the 8×RTX-4090 box (all cards idle, 0 MiB). Per position, logits computed
on card 0 twice (run-to-run) and on cards 1–7 once each (cross-card), same input,
same shape. Report max|Δlogit|, bit-identity, and deterministic-sample flip vs
card 0. N=64. Script: `e4_crosscard.py`.

## Results — zero drift everywhere

```
gpt2  &  Qwen2.5-0.5B  (identical on both)
run-to-run (card0 x2) : dlogit max=0 | bit-identical 64/64 | sample-flip 0/64
card0 vs card1..7     : dlogit max=0 | bit-identical 64/64 | sample-flip 0/64  (every card)
```

Every comparison is **bit-identical** (0 max abs diff, 64/64 identical, 0 flips),
on both models, across all 8 cards and repeated runs.

## Findings

1. **Run-to-run is bit-exact.** Same card, same input, same shape → identical logits
   (cuBLAS is deterministic for a fixed problem shape). No run-to-run noise.
2. **Cross-card (same architecture) is bit-exact.** All 8 RTX 4090s produce
   byte-identical logits for the same input. No card-to-card noise on a homogeneous
   fleet.
3. **Therefore E3's 0–8% sample-flip is entirely batch-shape noise** (GEMM kernel
   selection changing with batch dims), not hardware or run-to-run. This cleanly
   attributes the one real drift source.

## Decision-relevant conclusions (D12)

- **Within a homogeneous fleet (same GPU model + driver), the only logit
  nondeterminism is batch shape.** Control the shape and recompute is bit-stable;
  but production uses continuous batching, so E3's drift stands → Stage-1 must still
  replay *signed* logprobs (D1/D2), not recompute.
- **Cross-architecture is the remaining open risk.** All 8 cards here are RTX 4090
  on one driver; A100/H100 / different CUDA / different GPU family is NOT covered and
  is exactly where bit-identity is expected to break. This is the "cross-node /
  cross-arch reproducibility CI must be green" gate (Ryan §5) — untestable on this
  box.

## Caveats / open

- Single architecture (RTX 4090), single driver, single node. Cross-arch / cross-
  driver / cross-node determinism is the open item that gates enforcement across
  heterogeneous hardware.
- HF forward; vLLM's own kernels (paged attention, chunked prefill) could introduce
  additional same-arch nondeterminism not seen here (the heavier E3b/vLLM run).
