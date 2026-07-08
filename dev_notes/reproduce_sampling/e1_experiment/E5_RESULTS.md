# E5 — filter-edge instability (D5)

**Question:** are the top_p / min_p boundary rules load-bearing? When a sample flips
under executor-vs-validator logprob noise, is it because a token crossed the filter
edge (kept set changed) or because weights differ inside a stable kept set? If
boundary-dominated, the filter-edge / tie-break rules (D5) must be pinned bit-exactly
on all sides.

**Setup:** single RTX 4090, `qwen` env. Per position, kept set + deterministic sample
computed from the ALONE logprobs and the BATCHED logprobs (both via the decimal
pipeline over the same signed top-128). N=96. Script: `e5_filteredge.py`.

## Results

```
                 memb-change | sample-flip | of flips: boundary / interior
gpt2
top_p=0.9        34/96 (35%) |  6/96       | 3 / 3
top_p=0.95       44/96 (46%) | 10/96       | 7 / 3
min_p=0.02       28/96 (29%) |  6/96       | 5 / 1
min_p=0.05       26/96 (27%) |  6/96       | 6 / 0
Qwen2.5-0.5B
top_p=0.9         7/96 ( 7%) |  4/96       | 0 / 4
top_p=0.95        9/96 ( 9%) |  3/96       | 0 / 3
min_p=0.02        5/96 ( 5%) |  1/96       | 0 / 1
min_p=0.05        5/96 ( 5%) |  0/96       | 0 / 0
```

## Findings

1. **Filter-edge crossings are common and model-dependent.** A token sits near the
   top_p/min_p boundary — so batch noise changes the kept set — on **5–9% (Qwen) to
   27–46% (gpt2)** of positions. gpt2 churns more because its batch noise is ~4×
   Qwen's (E3).
2. **Which side dominates the flips is also model-dependent:** gpt2's flips are
   mostly **boundary** (kept set changed), Qwen's are entirely **interior** (kept set
   stable, rounding differs). Neither is negligible.
3. Most membership changes do *not* flip the sample (the crossing token usually
   carries tiny weight), but a meaningful minority do.

## Decision-relevant conclusion (D5)

- **The filter-edge rule is exercised on a large fraction of positions** (up to ~46%
  have a token in the boundary zone). Because Stage-1 replays the *signed* logprobs,
  both sides see identical numbers — so they agree **only if the boundary rule is
  identical**. A subtle mismatch (`<` vs `<=` at the top_p cutoff, a different
  tie-break, a different `top_k` clamp or `min_p` empty-set fallback) would diverge
  on that same 5–46% of positions → false fraud at scale.
- **So D5's rules are load-bearing and must be pinned bit-exactly** across the
  Python executor, Python validator, and Go validator: comparison direction at the
  cutoff, residual tie-break (ties to token order, D7), `top_k` clamp, `min_p`
  empty-set fallback. This is not an edge case — it fires constantly.

## Caveats / open

- The membership-change rate here is measured under batch noise (alone vs batched)
  as a proxy for "how often a token sits in the boundary zone." In Stage-1 both sides
  use identical signed logprobs, so the risk is a rule *mismatch*, not the noise
  itself.
- Two model families, single card, HF. Boundary density depends on distribution
  shape and temperature (fixed at 1.0 here).
