# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Check 2 -- Sampling Replay (pure CPU, no torch).

The validator replays the executor's per-position sampling from the reported
logprobs and compares against the reported token. Zero tolerance: any mismatch
is fraud. See docs/DETERMINISTIC_SAMPLING_VALIDATION.md and ADR 0001.

Scope (MERGE-PLAN §1, §8):
- This module implements Check 2 only. Full honesty also requires Check 1
  (logprob distance), which re-runs the model on GPU and is out of scope here
  (§8 U8).
- ``verify_sampling_from_logprobs`` verifies a *single* position and returns a
  bool. The sequence-level loop belongs to the serving-layer orchestrator, not
  here (ADR 0001, §8 U6).
- The seed is passed in already-composed as ``seed_str``; this module does not
  derive it. Seed hardening is a separate concern (§8 U1).
"""

from typing import Dict, Optional

from vllm.v1.sample.deterministic_utils import (
    Sha256CounterRNG,
    logprobs_to_weights,
    sample_categorical_weights,
)


def verify_sampling_from_logprobs(
    logprobs: Dict[str, float],
    seed_str: str,
    temperature: str,
    top_p: Optional[str],
    top_k: Optional[int],
    min_p: Optional[str],
    reported_token: str,
) -> bool:
    """Replay Check 2 for a single token position.

    Args:
        logprobs: {token_id_str: float} -- the executor's post-penalty logprobs
            for this position (matches ``EnforcedToken.logprobs``). Converted to
            canonical decimal strings via ``repr(f)`` before entering the
            pipeline (the single float->string conversion point; MERGE-PLAN §4,
            §8 U9). ``Decimal(repr(f))`` is the documented canonicalization.
        seed_str: The already-composed RNG seed string (this function does not
            derive it; §8 U1).
        temperature: Temperature as string (e.g. "0.7"). Must be > 0.
        top_p: Optional nucleus sampling threshold as string.
        top_k: Optional top-k filter count.
        min_p: Optional min-p threshold as string.
        reported_token: The token ID string the executor claims it sampled.

    Returns:
        True if the replayed token matches ``reported_token`` (honest), else
        False (fraud). Zero tolerance.

    Note:
        A True result means the sampling step is consistent with *these*
        logprobs -- it does not prove the logprobs themselves are what the model
        produced (that is Check 1, §8 U8), nor that the executor's production
        weight path is reproducible (§8 U11/U12).
    """
    # Single float->string conversion point (§4). repr() is CPython's
    # shortest round-tripping representation.
    logprob_strings = {tid: repr(f) for tid, f in logprobs.items()}

    weights = logprobs_to_weights(
        logprob_strings, temperature,
        top_p=top_p, top_k=top_k, min_p=min_p,
    )

    # Weight list built in lexicographic token-ID-string order; the returned
    # index maps back through the same order (§8 U12).
    sorted_tids = sorted(weights.keys())
    weight_list = [weights[tid] for tid in sorted_tids]

    rng = Sha256CounterRNG.from_seed_string(seed_str)
    idx = sample_categorical_weights(weight_list, rng)
    replayed_token = sorted_tids[idx]

    return replayed_token == reported_token
