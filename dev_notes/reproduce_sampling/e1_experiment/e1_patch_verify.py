"""E1 patch + verification (item a).

Contains the proposed replacement for
vllm/v1/sample/ops/topk_topp_sampler.py::deterministic_sample and verifies it on
real model output:

  deterministic_sample_float  = the CURRENT function (float (probs*2^16).round())
  deterministic_sample_e1     = the E1 PATCH (decimal pipeline from logprobs over
                                a provisional top-`support_k` support set)
  validator_replay            = what the chain validator does from the artifact

Result expected: float executor mismatches the validator (false rejects);
the E1 executor matches the validator on every position (0 false rejects).
"""
import sys
import torch

sys.path.insert(0, "/home/ps/szy/vwork/vllm/v1/sample")
import deterministic_utils as du
from transformers import AutoModelForCausalLM, AutoTokenizer

SUPPORT_K = 64  # PROVISIONAL support-set size (pending the signed support-set contract, D2)


# ---- CURRENT function (verbatim logic from topk_topp_sampler.py) ----
def deterministic_sample_float(probs, rngs):
    weights = (probs * (2 ** 16)).round().to(torch.int64).cpu().numpy()
    out = []
    for i in range(probs.size(0)):
        if i in rngs:
            out.append(du.sample_categorical_weights(weights[i].tolist(), rngs[i]))
        else:
            out.append(int(probs[i].argmax()))
    return out


# ---- E1 PATCH: decimal weights from logprobs over the top-k support set ----
def deterministic_sample_e1(logits, temperatures, top_ps, top_ks, min_ps, rngs,
                            support_k=SUPPORT_K):
    """Bit-reproducible weights via the decimal pipeline instead of GPU float
    rounding. Signature changes from (probs) to (logits + sampling params): the
    decimal pipeline consumes canonical logprob strings, and we sign/replay only
    the top-`support_k` logprobs (provisional; the final support set is D2)."""
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    out, artifacts = [], []
    for i in range(logits.size(0)):
        if i not in rngs:
            out.append(int(logits[i].argmax()))
            artifacts.append(None)
            continue
        topv, topi = torch.topk(logprobs[i], support_k)
        lp = {str(int(t)): repr(float(v)) for t, v in zip(topi.tolist(), topv.tolist())}
        w = du.logprobs_to_weights(
            lp, str(float(temperatures[i])),
            top_p=None if top_ps is None else str(float(top_ps[i])),
            top_k=None if top_ks is None else int(top_ks[i]),
            min_p=None if min_ps is None else str(float(min_ps[i])),
        )
        tids = sorted(w.keys())
        wl = [w[t] for t in tids]
        out.append(int(tids[du.sample_categorical_weights(wl, rngs[i])]))
        artifacts.append((lp, str(float(temperatures[i])),
                          None if top_ps is None else str(float(top_ps[i])),
                          None if top_ks is None else int(top_ks[i])))
    return out, artifacts


# ---- validator: replay from the artifact's logprobs ----
def validator_replay(artifact, seed):
    lp, temp, top_p, top_k = artifact
    w = du.logprobs_to_weights(lp, temp, top_p=top_p, top_k=top_k)
    tids = sorted(w.keys())
    wl = [w[t] for t in tids]
    return int(tids[du.sample_categorical_weights(wl, du.Sha256CounterRNG.from_seed_string(seed))])


def apply_top_k_mask(logits, k):
    if k is None:
        return logits
    v, _ = torch.topk(logits, k)
    return torch.where(logits < v[..., -1, None], torch.full_like(logits, float("-inf")), logits)


def main(model_id, temp, top_k, N=128):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).cuda().eval()
    ids = tok("The history of artificial intelligence began", return_tensors="pt").input_ids.cuda()
    mism_float = mism_e1 = 0
    for step in range(N):
        with torch.no_grad():
            logits = model(ids).logits[0, -1, :].float().unsqueeze(0)  # [1, vocab]
        seed = "e1|%d" % step
        rngs = {0: du.Sha256CounterRNG.from_seed_string(seed)}

        # float executor (params applied to logits, then softmax -> probs)
        masked = apply_top_k_mask(logits, top_k) / temp
        probs = torch.softmax(masked, dim=-1)
        s_float = deterministic_sample_float(probs, {0: du.Sha256CounterRNG.from_seed_string(seed)})[0]

        # E1 executor
        out, arts = deterministic_sample_e1(logits, [temp], None, [top_k], None,
                                            {0: du.Sha256CounterRNG.from_seed_string(seed)})
        s_e1 = out[0]

        # validator replays from the E1 artifact
        s_val = validator_replay(arts[0], seed)

        if str(s_float) != str(s_val):
            mism_float += 1
        if str(s_e1) != str(s_val):
            mism_e1 += 1
        ids = torch.cat([ids, torch.tensor([[int(s_e1)]], device=ids.device)], dim=1)

    print("%s temp=%s top_k=%s | FLOAT exec vs validator: %d/%d (%.1f%%) | "
          "E1 exec vs validator: %d/%d"
          % (model_id, temp, top_k, mism_float, N, 100.0 * mism_float / N, mism_e1, N))


if __name__ == "__main__":
    model_id = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    main(model_id, 1.0, 40)
    main(model_id, 0.7, 40)
    print("E1_PATCH_DONE")
