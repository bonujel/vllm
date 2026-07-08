# E3 — batching-induced logit drift

**Question:** does batch composition perturb a position's logits enough to (a) flip
the deterministic-sampled token, and (b) move recomputed logprobs? I.e. can a
validator *recompute* logits and replay, or must it replay the *signed* logprobs?

**Setup:** RTX 4090 (single card, `CUDA_VISIBLE_DEVICES=0`, `qwen` env). Real logits
from `gpt2` and `Qwen/Qwen2.5-0.5B`. Per position, compute the target position's
logits (a) with the sequence ALONE (batch=1) and (b) RIGHT-PADDED into a batch of K
unrelated filler sequences (K = 1 / 4 / 16). Right-pad + causal mask ⇒ in exact
arithmetic the target's logits are batch-independent; any diff is float batched-GEMM
noise. 64 positions per config. Script: `e3_batch_drift.py`.

## Headline

| model | dlogit p95 | dlogprob(top64) p95 | argmax-flip | **sample-flip** |
|---|---|---|---|---|
| gpt2 | ~0.19 (tail max 2.4) | **~0.12 nats** | 0–1/64 | **2–5/64 = 3–8%** |
| Qwen2.5-0.5B | ~0.05 | **~0.04 nats** | 0–1/64 | **0–1/64 = 0–1.6%** |

Raw:
```
# gpt2
temp=1.0 top_p=0.9  batch=1+1  | dlp p50=5.3e-2 p95=1.2e-1 | argmax 0/64 | sample 3/64
temp=1.0 top_p=0.9  batch=1+4  | dlp p50=5.3e-2 p95=1.2e-1 | argmax 1/64 | sample 2/64
temp=1.0 top_p=0.9  batch=1+16 | dlp p50=5.7e-2 p95=1.2e-1 | argmax 0/64 | sample 4/64
temp=0.7 nofilter   batch=1+1  | dlp p50=4.9e-2 p95=1.2e-1 | argmax 1/64 | sample 5/64
temp=0.7 nofilter   batch=1+4  | dlp p50=4.7e-2 p95=1.1e-1 | argmax 1/64 | sample 4/64
temp=0.7 nofilter   batch=1+16 | dlp p50=5.0e-2 p95=1.1e-1 | argmax 1/64 | sample 3/64
# Qwen2.5-0.5B
temp=1.0 top_p=0.9  batch=1+1  | dlp p50=1.8e-2 p95=3.8e-2 | argmax 0/64 | sample 0/64
temp=1.0 top_p=0.9  batch=1+4  | dlp p50=1.9e-2 p95=3.5e-2 | argmax 0/64 | sample 0/64
temp=1.0 top_p=0.9  batch=1+16 | dlp p50=2.1e-2 p95=3.9e-2 | argmax 0/64 | sample 0/64
temp=0.7 nofilter   batch=1+1  | dlp p50=2.2e-2 p95=3.1e-2 | argmax 0/64 | sample 0/64
temp=0.7 nofilter   batch=1+4  | dlp p50=1.9e-2 p95=3.3e-2 | argmax 1/64 | sample 0/64
temp=0.7 nofilter   batch=1+16 | dlp p50=2.0e-2 p95=3.1e-2 | argmax 1/64 | sample 1/64
```

## Findings

1. **Batch composition alone perturbs a position** — same card, same weights,
   decimal pipeline applied. Logprob drift p95 ~0.04 (Qwen) to ~0.12 nats (gpt2).
2. **It flips the deterministic-sampled token 0–8%** of positions. A validator that
   *recomputes* logits in a different batch context would false-reject at that rate,
   **even after the E1 fix.**
3. **~Flat across batch size** (1+1 ≈ 1+16): a kernel switch triggered by "batched
   vs not," not proportional to batch size — one co-batched sequence is enough.
4. **Model-dependent magnitude:** gpt2 ≈ 4× Qwen (fp16 lm_head + tied embedding →
   larger logit magnitudes → more GEMM noise).
5. argmax (greedy / temp-0) is far more stable (~0–1.5% flip) but not zero — top
   near-ties still flip.

## Decision-relevant conclusions

- **D1/D2 — Stage-1 must replay the *signed* logprobs, not recompute.** Recompute
  false-rejects 0–8% purely from batching. This is the empirical basis for signing
  the support set rather than recomputing it.
- **D12/D13 — Stage-2 distance threshold floor.** Recompute logprob drift reaches
  ~0.12 nats (gpt2 p95) from batching alone; the distance threshold must sit above
  this or honest inferences fail the distance check. (Ryan §5: "distance
  distributions with vs without deterministic mode.")
- **D8 — temp-0** is more robust but not immune to near-tie flips.

## Caveats / open

- **HF proxy.** This captures batched-GEMM noise via HuggingFace. vLLM's real
  executor path (chunked prefill, paged attention, continuous batching) is a
  separate, heavier **E3b** and may drift differently.
- Does not separate batch-shape noise from pure run-to-run fp16 noise (same shape
  repeated) or cross-GPU noise — that is **E4**.
