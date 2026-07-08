# E-dist — Stage-2 distance: honest recompute-noise vs fraud

**Question:** does the Stage-2 distance check actually separate honest inferences
(same model, recompute noise) from fraud (logprobs from a different / cheaper
model)? And which distance metric works? Feeds the Stage-2 threshold gap (D12/D13).

**Setup:** single RTX 4090, `qwen` env, fp16. Same-tokenizer model pairs:
`real` = claimed model, `cheap` = pre-fill attacker's generator.
- Honest: real generates; signed = real logprobs (alone), recompute = real logprobs
  (batched with fillers).  → recompute noise only.
- Fraud: cheap generates; signed = cheap logprobs (alone, best case for attacker),
  recompute = real logprobs on the same tokens.  → model-difference distance.
N=96 positions, top-K=64. Script: `edist_distance.py`.

Metrics over the signed top-64: `d_chosen` (sampled token's logprob delta),
`d_mae` (mean abs logprob diff), `d_kl` (KL of renormalized top-64).

## Results

```
Qwen2.5-1.5B (real) vs Qwen2.5-0.5B (cheap)
d_chosen | HONEST p95=0.011 | FRAUD p5=0.006 | OVERLAP
d_mae    | HONEST p95=0.011 | FRAUD p5=0.437 | SEPARATED (~40x)
d_kl     | HONEST p95=0.000 | FRAUD p5=0.006 p50=0.127 | SEPARATED

gpt2-large (real) vs gpt2 (cheap)
d_chosen | HONEST p95=0.008 | FRAUD p5=0.036 | separated (this pair)
d_mae    | HONEST p95=0.008 | FRAUD p5=0.525 | SEPARATED (~65x)
d_kl     | HONEST p95=0.000 | FRAUD p5=0.063 p50=0.312 | SEPARATED
```

## Findings

1. **The distance check works — with the right metric.** `d_mae` (mean abs logprob
   diff over the top-64 support) cleanly separates honest (p95 ~0.01) from fraud
   (p5 ~0.44–0.53), a **~40–65× margin**. Any threshold in ~[0.05, 0.4] separates
   them on both model families. `d_kl` is equally clean (honest ~0 vs fraud ≥0.06).
2. **Do NOT gate on the sampled token alone.** `d_chosen` OVERLAPPED on the Qwen
   pair (honest p95 0.011 vs fraud p5 0.006): a cheap model often agrees on the
   easy/obvious token, so per-token logprob is a weak discriminator. Fraud shows up
   in the **shape of the whole distribution**, not the single chosen token. Stage-2
   must compare the full top-K (MAE/KL), not just the reported token.
3. Honest recompute noise (~0.01 MAE) is consistent with E3's batching drift.

## Decision-relevant conclusions

- **Stage-2 threshold gap (D12/D13):** put the distance threshold on `d_mae` (or
  `d_kl`) **over the top-K support**, set well above the honest floor (~0.01) and
  below the fraud floor (~0.44). A value near ~0.1 MAE gives large margin both ways.
- **Metric choice matters:** the sampled-token delta is insufficient; use the
  aggregate over the signed support set (ties back to D2 — the support set is also
  what Stage-2 measures over).

## Caveats / open

- **Clear fraud only.** 0.5B-vs-1.5B (a 3× size gap) separates by 40–65×. The real
  test is the **hardest fraud**: a quantized version of the *same* model, or a
  similar-size different finetune — the fraud distance shrinks and may approach the
  honest floor. That gray zone is the **quantization sweep** (next todo).
- **HF proxy.** Honest noise here is batched-GEMM only; vLLM's penalty-pipeline
  reorder (Ryan §5 "with vs without deterministic mode") is not included and may add
  to the honest floor.
- KL is directional (signed‖recompute); the production metric/direction is a design
  choice, but any of MAE/KL over the support works far better than d_chosen.
