# E-quant — Stage-2 distance gray zone: quantized same-model vs honest noise

**Question:** Experiment 4 separated honest (same model, batch noise) from a clearly
cheaper model by ~40–65×. The hard case is a **quantized version of the same model**
— an attacker (or an honest node) runs int8/int4 but the artifact is scored against
fp16. Does the distance check catch it, or does quant blend into the honest floor?

**Setup:** single RTX 4090, `qwen` env, `bitsandbytes` 0.49.2. Reference = BASE in
fp16 (validator recompute). Per position, distance vs the fp16 reference for:
`honest` (fp16 recomputed batched), `int8` (load_in_8bit), `int4` (load_in_4bit
nf4), `cheap` (3× smaller model, clear-fraud scale marker). N=96, top-64. Metric
`d_mae` (mean abs logprob diff) and `d_kl`. Script: `equant_grayzone.py`.

## Results — a precision ladder

```
Qwen2.5-1.5B (base)                 gpt2-large (base)
cond   | d_mae p50 / p95           cond   | d_mae p50 / p95
honest | 0.0058 / 0.0112           honest | 0.0027 / 0.0076
int8   | 0.0974 / 0.2004           int8   | 0.0539 / 0.1454
int4   | 0.3644 / 0.6919           int4   | 0.2048 / 0.6358
cheap  | 0.8171 / 2.2602           cheap  | 0.8841 / 2.0444
```

Both families give the same ordering: **honest ≪ int8 < int4 < cheaper model.**

## Findings

1. **The distance metric is a precision ladder, not a binary.** honest fp16 (~0.01),
   int8 (~0.05–0.20), int4 (~0.20–0.69), different model (~0.8–2.3) — monotone, on
   both families.
2. **int4-claiming-fp16 is clearly caught.** d_mae ~0.20–0.36 is ~25–35× the honest
   floor, in the same range as a 3× smaller model. A distance gate flags it.
3. **int8 is a genuine gray zone.** d_mae ~0.05–0.20 is ~7–18× the honest floor but
   well below clear fraud. Experiment 4's clean ~0.1 threshold sits **right at int8's
   p50** — int8 would be flagged much of the time.

## Decision-relevant conclusion (D12)

- **The distance threshold implicitly defines *which precisions count as "the
  model."*** It is not just "honest vs fraud": placing it at ~0.05 accepts only
  fp16; ~0.25 accepts fp16+int8; ~0.75 accepts down to int4. That placement is a
  **policy choice**, not a purely empirical one.
- **This is the real driver of "false-reject rate by quantization" (Ryan §5).** If
  the network standardizes on one precision, the point is moot. If honest nodes may
  run mixed precision, the check must tolerate the **coarsest accepted** precision —
  which raises the honest floor and shrinks the margin to fraud. int4 still clears
  int8, so a threshold near ~0.25 accepts int8 as honest while still catching int4
  and cheaper models.
- **Stage-1 replay is unaffected** by any of this: it replays the executor's *signed*
  logprobs, so it is precision-agnostic regardless of what the executor ran. Quant
  only matters for the Stage-2 distance gate.

## Caveats / open

- One model family shape (Qwen, gpt2), single card, HF; int4 = nf4 specifically.
  Other schemes (gptq/awq/int4 variants) may land at different rungs.
- The honest floor here is fp16 batched-GEMM noise only; vLLM's penalty-pipeline
  reorder is not included and would raise it somewhat.
- "Is int8 fraud?" is a protocol question, not a measurement — the experiment only
  locates each precision on the distance scale; the accept/reject line is D12's call.
