"""E1 shadow experiment, generalized: model via argv, top_k and top_p variants.

Per position, false-reject = a decimal validator sampling a different token than
the executor under the SAME RNG. Compares two executors:
  FLOAT   (current):  softmax(logits/T) float32 -> nucleus -> renorm -> round
  DECIMAL (E1 fix):   du.logprobs_to_weights (same pipeline as the validator)
"""
import sys
import torch

sys.path.insert(0, "/home/ps/szy/vwork/vllm/v1/sample")
import deterministic_utils as du
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
N = 128
PROMPT = "The history of artificial intelligence began"

print("MODEL:", MODEL, "loading...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()


def sample(w, seed):
    tids = sorted(w.keys())
    wl = [int(w[t]) for t in tids]
    return tids[du.sample_categorical_weights(wl, du.Sha256CounterRNG.from_seed_string(seed))]


def float_weights(logits, idx, temp, top_p):
    """Float executor: softmax over the top-k, optional top_p nucleus, renorm,
    round to 2^16 -- mirrors du.logprobs_to_weights but in GPU float."""
    sub = logits[idx] / temp
    p = torch.softmax(sub, dim=-1)
    keep = list(range(len(idx)))
    if top_p is not None:
        order = torch.argsort(p, descending=True).tolist()
        cum, kept = 0.0, []
        for j in order:
            cum += float(p[j])
            kept.append(j)
            if cum >= top_p:
                break
        keep = kept
    psub = p[keep]
    psub = psub / psub.sum()
    return {str(idx[j]): int(round(float(psub[n]) * 65536)) for n, j in enumerate(keep)}


def run(temp, topk, top_p):
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    mf = md = 0
    sfmin, sfmax = 10 ** 9, 0
    for step in range(N):
        with torch.no_grad():
            logits = model(ids).logits[0, -1, :].float()
        topv, topi = torch.topk(logits, topk)
        idx = [int(x) for x in topi]

        w_float = float_weights(logits, idx, temp, top_p)
        sf = sum(w_float.values())
        sfmin, sfmax = min(sfmin, sf), max(sfmax, sf)

        logprobs = torch.log_softmax(logits, dim=-1)
        lp = {str(t): repr(float(logprobs[t])) for t in idx}
        w_dec = du.logprobs_to_weights(
            lp, str(temp), top_p=(str(top_p) if top_p is not None else None))

        seed = "e1|%d" % step
        s_float = sample(w_float, seed)
        s_val = sample(w_dec, seed)
        s_dec = sample(w_dec, seed)
        if s_float != s_val:
            mf += 1
        if s_dec != s_val:
            md += 1
        ids = torch.cat([ids, torch.tensor([[int(s_float)]], device=ids.device)], dim=1)

    tag = "temp=%s topk=%s top_p=%s" % (temp, topk, top_p)
    print("%-30s | FLOAT vs decimal: %3d/%d (%.1f%%) | DECIMAL(E1) vs decimal: %d/%d | float sum[%d,%d]"
          % (tag, mf, N, 100.0 * mf / N, md, N, sfmin, sfmax))


run(0.7, 20, None)
run(1.0, 20, None)
run(1.0, 40, 0.9)
run(0.7, 40, 0.95)
print("E1_DONE")
