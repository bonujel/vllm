# Plan — CPU-only work landable in this repo (gonka #1199)

Scope: work that can be done **in this vLLM repo, pure CPU, no GPU**, and that
depends on **no unpinned decision**. Derived from a grill session that classified
each candidate as "can-do now" vs "must-wait". Result: all four candidates are
can-do — the one previously thought blocked (full three-valued verdict) is
unblocked because the gonka repo with the authoritative `verify.go` is on this
machine (`~/dev/web3/gonka`, branch `bonujel/deterministic_sampling_merged`).

Out of scope here (needs GPU or a team decision — tracked in HANDOFF, not this
plan): E1 executor→decimal weights, S1 seed wiring into `gpu_model_runner`,
serving_chat orchestration end-to-end, Stage-2 threshold calibration, inference_id
transport.

Authoritative sources this plan leans on (verify before coding — do not trust the
notes' paraphrase):
- Contract: `docs/DETERMINISTIC_SAMPLING_CONTRACT.md` (v1.0.0) — §0 versioning,
  §3 token order, §4 pipeline, §7 greedy/temp0.
- Go verdict state machine: `~/dev/web3/gonka/.../detsample/verify.go`
  (`VerifyPosition`, three-valued).
- Golden vectors: `tests/v1/validation/conformance_vectors.json`.

---

## Item 0 — Establish a test baseline (no code change)

**Do:** run the CPU-only validation + conformance suites and record what is green
now, before touching anything.

```
pytest tests/v1/validation/ -v
python scripts/gen_conformance_vectors.py   # regenerate; must be a no-op diff
```

- **Done when:** the current pass/fail set is captured (a short note of which
  tests pass on this machine). Any pre-existing failures are noted, not fixed
  here — they become the baseline the later items must not regress.
- **Touches:** nothing (read/run only).
- **Why first:** everything below needs a known-green starting point to prove it
  didn't break anything.

## Item 1 — Converge token order (U12 / contract §3)

**Problem:** two Python validators sort the weight list differently, so they map
the same RNG draw to different tokens and disagree on the same artifact.
- `validation_sampling.py:76` — `sorted(weights.keys())` → lexicographic string
  order (**contract-aligned**, correct).
- `validation_logic.py:228` — `sorted(..., key=lambda x: int(x[0]))` → numeric
  order (the one to fix; `"10" < "2"` string vs `2 < 10` numeric).

**Do:** change the `validation_logic.py` sort to match §3 (lexicographic string).

- **Done when:** a test constructs an artifact whose token ids include both `"2"`
  and `"10"` and asserts the two validators now agree on the same artifact
  (before: they disagree). Item 0 baseline still green.
- **Touches:** `validation_logic.py` sort key + one new/extended test.
- **Not:** do not change `validation_sampling.py` (already correct); do not touch
  the contract (§3 is already pinned).
- **Decision status:** pinned. Contract §3 + its "Open items" name this file and
  direction explicitly — this is pending *implementation*, not pending *decision*.

## Item 2 — Three-valued verdict + fix silent-passthrough (do together)

These are two ends of one thing, so they land together.

**Problem A (silent passthrough):** `serving_chat.py:1846` calls
`validate_full(enforced_tokens=...)` but the param is named `artifact`
(`validation_logic.py:385`) → `TypeError` at runtime → swallowed by the
`except Exception` at :1852 → returns `fraud=False` (silent pass). Root issue is
**the failure direction**: a validator that errors must not translate that into
"honest". The keyword typo is just the symptom that kept the path from ever
running.

**Problem B (binary verdict):** `verify_sampling_from_logprobs -> bool`
(`validation_sampling.py:38`) collapses honest/fraud with no way to say
"inconclusive". Go's `VerifyPosition` is already three-valued.

**Do:**
1. Port `verify.go`'s `VerifyPosition` state machine to Python — mirror its order
   and semantics exactly:
   - contract-version mismatch → inconclusive (§0)
   - seed-domain mismatch → inconclusive
   - greedy / temperature==0 → inconclusive (§7; branch on
     `temperature is not None and temperature > 0`, never falsy-zero)
   - non-positive/unparseable temperature → inconclusive
   - replay error → inconclusive
   - replay ok, token differs → fraud (zero tolerance)
   - token matches → honest
2. Fix the `serving_chat.py` call (`artifact=`) **and** change the `except` so a
   validator-side error yields **inconclusive**, not `fraud=False`. Never
   translate "validator failed" into "executor honest".

- **Done when:**
  - the Python verdict enum + order match `verify.go` clause-for-clause;
  - the same `conformance_vectors.json` cases produce the same verdicts on
    Python and Go (cross-language check);
  - a test proves a validator-side error / unsupported version / greedy position
    returns inconclusive (not fraud, not silent-honest);
  - Item 0 baseline still green.
- **Touches:** `validation_sampling.py` (bool→verdict), `serving_chat.py` (call +
  except direction), tests.
- **Not:** do not invent verdict semantics — copy `verify.go`. Do not wire this
  into the GPU/executor path (that's E1/S1, out of scope). Do not change the
  contract.
- **Decision status:** pinned. `verify.go` is the authoritative state machine and
  is readable locally; every branch maps to a pinned contract clause. This is a
  translation with an executable oracle (the golden vectors), not a design choice.

---

## Suggested order

`Item 0 (baseline)` → `Item 1 (U12, independent, low-risk)` → `Item 2 (verdict +
passthrough)`. Item 2 last because it is the largest and benefits from Item 1's
order fix already being in place.

## Explicitly deferred (do not start here)

| Deferred | Why | Where tracked |
|---|---|---|
| Executor → decimal weights (E1) | needs GPU; biggest remaining block | HANDOFF |
| Chain-bound seed into generation (S1) | needs GPU + inference_id transport | HANDOFF, decision #1/#2 |
| serving_chat full orchestration end-to-end | needs GPU + emitted artifacts | HANDOFF |
| Stage-2 distance threshold calibration | needs real data | HANDOFF |
| Sequence RNG semantics (one stream vs per-position) | genuinely unpinned decision | contract "Open items", HANDOFF decision #4 |
