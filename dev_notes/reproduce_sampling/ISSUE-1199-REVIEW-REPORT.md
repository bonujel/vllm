<!--
============================================================================
  POST-READY SECTION (English) — paste everything between the >>>8 markers
  into the GitHub comment box on gonka-ai/gonka#1199.
  The Chinese section at the bottom is for your own reference — DO NOT post it.
  Each Review Point is quoted verbatim, answered, and grounded in the actual
  source (real code with file:line) or in checks I ran locally.
============================================================================
-->

<!-- >>>8 ---------------- BEGIN GitHub comment ---------------- 8<<< -->

## Review: reproducible sampling for inference validation

I read the proposal and the deterministic-sampling code, re-ran the reproducible parts on a clean machine (Python 3.10.12, CPU only, no torch/GPU), and looked specifically for ways the scheme could be bypassed before it is enforced. Each Review Point below is answered against the actual source.

**TL;DR — final recommendation: not ready.** The two-stage design is sound and the RNG core is correct, but the work sits on two incompatible, unmerged branches; the validator-side replay does not exist; the seed deviates from the proposal and can be ground offline; the float→string artifact format is unspecified; and there is no Go reference, so the cross-machine claim is untested.

**Scope.** The issue names one branch: `gonka-ai/vllm @ tg/detemrinistic_sampling_dump` (commit `4ce45cf5e`, @tamazgadaev). It holds the design doc, the RNG core, and a **decimal / integer-weight** pipeline, but no integration. This is the subject of the review.

While reviewing it, I found that its tests import a module and symbols that are absent from the branch (Point 6). Those live on a separate branch by the same author, `tg/deterministic_sampling_v011`, which carries an earlier **float** sampling path wired into the V1 sampler (`DeterministicSampler`, the `VLLM_DETERMINISTIC_SAMPLING` flag, `SamplingMetadata` fields). The two branches share a common ancestor but are otherwise independent and unreconciled (`v011` is not an ancestor of `dump`, and sits on a much newer vLLM base). I reference `v011` only where it explains the `dump` branch's state; it is not itself in the issue's scope.

---

### Review Point 1 — Confirm alignment with the proposed fix

> Check whether the current implementation matches the two-stage validation direction described in the proposal. Please focus on whether the implementation correctly supports: sequence / sampling replay validation; existing distribution validation; the intended order of checks; the intended protection against the known validation weakness.

The two-stage structure, ordering, and existing distribution check all match the proposal. The two weak spots are the **seed derivation**, which deviates from the proposal, and the **replay**, which is not implemented.

The seed derivation diverges from the proposal:

* **Proposal:** `run_seed = SHA256(user_seed ‖ inference_id_from_chain)` — bound to `inference_id_from_chain`, which the chain assigns and the executor cannot choose.
* **Code:** `f"{seed}|{prompt_token_ids}"` (`docs/DETERMINISTIC_SAMPLING_VALIDATION.md:498`, `:176`) — bound to `prompt_token_ids` instead.
* **Consequence:** this swap is the root of the offline-grinding weakness (Point 2), and it adds a tokenizer-determinism dependency — the validator must reconstruct identical token ids to rebuild the seed, so any tokenizer/engine drift changes the seed and forces a false reject.

| Sub-point | Status |
|---|---|
| sequence / sampling replay validation | Primitives exist; validator replay not built (Point 2); seed deviates |
| existing distribution validation | Supported — unchanged Stage 2 in gonka (Go) |
| intended order of checks | Matches in design |
| protection against the known weakness | Incomplete — only holds if the seed can't be ground (Point 2) and `temperature > 0` (Point 6) |

### Review Point 2 — Review seed, RNG, and replay logic

> Check whether the seed generation, RNG initialization, and replay logic are implemented consistently with the inference validation proposal and existing deterministic sampling document. Please verify whether the validator can reliably replay the sampling step from the provided artifact and seed data.

The RNG initialization and draw sequence are correct and reproducible. Running the core module against fixed inputs:

```
iter_u64("reference_seed_v1", 5)[0]   = 4286832458236889005
decimal_sample(seed="42|[1,2,3]") x2  = 791 791
sum(logprobs_to_weights(...))         = 65536
```

The same seed yields the same token across runs, and the decimal pipeline's integer weights sum to exactly `2^16`. These are stable reference values — the first u64 above is what a Go port would have to match bit-for-bit.

**The seed generation is inconsistent with the proposal and can be ground offline (blocker).**

* **Why it is exploitable:** every seed input is request-controlled — `{seed}` is request-supplied, and production uses `rand.Int31()` while preserving a user-provided seed.
* **The attack:** a self-dealing executor generates a sequence with a cheap model, tries seeds offline until one replays that sequence under the sequence check, and submits it.
* **The proposal is also only partly safe:** `user_seed` is developer-supplied, so `SHA256(user_seed ‖ inference_id)` is still grindable — the seed needs at least one component the executor cannot pick, e.g. `executor_addr` and a `start_block_hash`.

**The validator cannot reliably replay today, because the replay code does not exist.** There is no `validation_sampling.py` on either branch; the replay logic appears only in standalone scripts. And even once written, it is exposed to the float→string issue in Point 3: if executor and validator canonicalize probabilities differently, the replayed draw lands on a different token at a distribution boundary and false-rejects.

### Review Point 3 — Review artifact contents

> Check whether the artifact contains everything required for replay and validation. Please compare the artifact requirements in the proposal with the artifact format described in the deterministic sampling document.

The proposal requires `run_seed` + per-position top-k tokens **and their probabilities**. The carrier type only holds the tokens, and the probability encoding has no canonicalization rule.

`EnforcedToken` carries no probabilities (`vllm/validation.py:13`):

```python
class EnforcedToken(BaseModel):
    token: str
    top_tokens: List[str] = Field(default_factory=list)   # token strings only — no logprobs
    token_id: Optional[int] = Field(default=None, exclude=True)
    top_token_ids: List[int] = Field(default_factory=list, exclude=True)
```

**Blocker — the float→string canonicalization is undefined.** The decimal pipeline ingests probabilities as strings, but nothing specifies which float the string comes from, and that choice changes the value:

* **Same number, two strings:** `repr(-0.05)` is `"-0.05"`, but the same value as a float32 widened to float64 is `"-0.05000000074505806"` — the canonical string depends entirely on the exact float64 bits.
* **Cross-language divergence:** Python (`repr`) and Go (Ryu) are known to disagree on shortest-float formatting (golang/go#17997), so executor and validator can produce different strings and false-reject.
* **Fix:** store the model's actual float32-widened-to-float64 value under one agreed canonicalization, do float→string only on the vLLM side, and never let Go format a float.

### Review Point 4 — Review implementation status

> Using provided documents, identify: what already exists; what still needs to be modified; what still needs to be created; whether the implementation plan in the document is still accurate; whether the existing code is ready to be included in MLNode versions softly.

There are two parallel architectural bets, and that difference is the crux. `v011` samples on GPU floats, in `deterministic_sampler.py::deterministic_sample`:

```python
# Compute softmax probabilities
probs = logits.softmax(dim=-1, dtype=torch.float32)   # GPU float32 — varies across machines
...
token_id = sample_categorical(probs_cpu[i], rng)      # categorical sample over floats
```

`dump` replaces that with the decimal integer-weight pipeline (`deterministic_utils.py` defines `logprobs_to_weights` + `sample_categorical_weights`). Notably, v011's own integer function is documented as being *"for cross-language reproducibility when probs may differ in float rounding"* — the author already saw the float path is not reproducible across machines and pre-staged the integer fix; `dump` promoted it to a full pipeline but dropped the integration. The branches were never reconciled.

| | `v011` | `dump` |
|---|---|---|
| sampling math | float (`softmax(float32)` + categorical) | decimal integer-weight pipeline |
| `DeterministicSampler` (708-line module) | present | absent |
| `VLLM_DETERMINISTIC_SAMPLING` flag + metadata fields | present | absent |
| sampler actually invoked by the worker | **no** | no |

On neither branch does the worker invoke the sampler. `DeterministicSampler` is referenced only by its own module and a single test — `gpu_model_runner` never instantiates it:

```
$ git grep -l DeterministicSampler origin/tg/deterministic_sampling_v011
vllm/v1/sample/deterministic_sampler.py
tests/v1/sample/test_deterministic_sampler.py
```

* **Exists:** RNG + decimal core (`dump`); integration skeleton + float sampler (`v011`); the design doc.
* **Needs modification:** swap v011's float sampling for the decimal pipeline; repoint the tests; change the seed derivation.
* **Needs creation:** worker-level invocation; the validator replay (`validation_sampling.py`); a protocol-forced seed; the canonicalization contract; a Go reference.
* **Is the document's plan still accurate?** Partially — it describes wiring the RNG in `gpu_model_runner.py`, but that wiring is absent on `dump`, and the plan predates the float→decimal split.
* **Ready to include softly?** No (Point 5).

### Review Point 5 — Review safe MLNode rollout

> Please identify: what can be added behind a flag; what can run in non-enforcing mode; what data should be collected; what should block strict enforcement; which parts are safe to include now; which parts should wait.

* **Behind a flag now:** only the standalone, side-effect-free pieces — the RNG and the decimal core, plus offline conformance/reference-vector tooling. They never touch the live request path.
* **Non-enforcing (log-only) mode:** nothing yet — log-only needs the sampler running in the generation loop and emitting artifacts, but per Point 4 the sampler is never invoked by the worker. Once wired, log-only (compute the verdict, record it, never reject) is the right first step.
* **Data to collect:** see the Expected Output list.
* **Should block strict enforcement:** offline seed grinding; the undefined canonicalization; the missing Go reference and unmeasured cross-machine false-reject rate; greedy traffic having no Stage 1 coverage.
* **Safe now:** the offline math core and reference-vector tooling.
* **Should wait:** anything that gates/rejects/penalizes; any seed-dependent acceptance; the float path.

### Review Point 6 — Review vulnerabilities and edge cases

> Carefully check the approach for possible vulnerabilities or edge cases before it becomes enforcing validation logic. Please focus on the actual proposed mechanism and the limitations already documented in the linked materials.

* **Seed grinding (blocker)** — Point 2.
* **Float→string mismatch (blocker)** — Point 3.
* **No cross-language reference.** Both branches are Python-only; with no Go implementation, cross-language reproducibility is an assertion with zero empirical test on either side — the second-largest risk after the seed.
* **Greedy decoding bypasses the RNG.** In `deterministic_sampler.py`, for `temperature < _SAMPLING_EPS` (1e-5) the sampler takes `greedy_sample`, which is `logits.argmax(dim=-1)` and never consults the RNG. So at `temperature == 0` the sequence check adds no protection, and the pre-fill defense only holds for `temperature > 0`. The code should also branch on `temperature is not None` rather than a falsy-zero check.
* **The committed tests do not run.** `tests/v1/sample/test_deterministic_standalone.py` imports `sample_categorical` and `WeightedPrefixSampler` from `deterministic_utils`, but the `dump` core defines neither (it has `sample_categorical_weights`, `logprobs_to_weights`, `decimal_sample_from_logprobs` instead), so the module fails at import/collection. Those symbols exist on `v011`, not `dump` — the tests belong to the other branch's API and can't count as passing evidence until the branches are reconciled.
* **Binary verdict — no fault attribution, no version gating.** `verify_sampling_from_logprobs(...) -> bool` returns only honest/fraud (`return replayed_token == reported_token`); there is no version field on the artifact (`EnforcedToken`) and no fault-attribution anywhere. A mismatch caused by the *validator's* own side — a different tokenizer/engine, divergent float behavior, or an unsupported model version — is therefore indistinguishable from executor fraud, and an honest executor is accused. Before enforcing, the artifact needs a version/build tag (model + engine + tokenizer + weight-pipeline) and the validator should gate on it: an unsupported or mismatched version should be classified as *inconclusive / version-unsupported*, not fraud. This means widening the verdict from a bool to a classified result (honest / fraud / inconclusive), and it is a prerequisite for safe enforcement — it compounds the false-reject risks in Points 1, 3, and 5.
* **Documented limitations to lock down before enforcing:** `logit_bias` disabled in deterministic mode (diverges from stock vLLM); `top_k` clamped to `max_num_logprobs`; CPython/libmpdec required (diverges on PyPy/Jython); any proxy/logger that truncates JSON float precision breaks the distribution comparison.

---

### Expected Output

* **Matches the two-stage proposal?** Partially — structure and RNG match; seed derivation diverges; validator replay not built (Point 1).
* **Already implemented?** RNG + decimal core (`dump`); integration skeleton + float sampler (`v011`); design doc — neither sampler runs in the worker (Point 4).
* **Needs modification / creation?** Modify: float→decimal sampling, tests, seed. Create: worker invocation, validator replay, protocol-forced seed, canonicalization contract, Go reference (Point 4).
* **Ready for soft MLNode integration?** No (Point 5).
* **What should remain non-enforcing?** Everything that gates/rejects/penalizes, all seed-dependent acceptance, the float path (Point 5).
* **Data to collect before strict validation?** (1) cross-machine replay agreement / false-reject rate for the **decimal** pipeline across real GPU/driver/library variation; (2) a Go reference + Python↔Go bit-for-bit conformance vectors; (3) the share of `temperature == 0` (greedy) traffic; (4) measured JSON float-precision loss through the real proxy/logging path.
* **Vulnerabilities / mismatches found:** offline seed grinding, undefined canonicalization, no Go reference, greedy bypass, non-running tests, binary verdict with no fault attribution / version gating, documented limitations (Point 6).
* **Recommended next steps:** (1) adopt the **decimal** path over float; (2) reconcile the branches — keep v011's skeleton, swap in the decimal pipeline, repoint the tests; (3) close the two blockers — protocol-forced seed, and a single vLLM-owned float→string canonicalization; (4) add a Go reference + conformance vectors; (5) then log-only rollout → measure false-reject rate → consider enforcing.
* **Final recommendation: not ready.** The mechanism is promising and the path is clear, but soft integration is blocked by code that does not run in the worker, a security-relevant seed weakness, an unspecified artifact canonicalization, and no cross-language reference. Risk is low today only because nothing is wired in; I would not include even a non-enforcing path until the sampler runs end-to-end and the two blockers are closed.

---

### Addendum — branch reconciliation (beyond the requested scope)

The issue asks for a review, not for code changes. As optional follow-up we reconciled the two branches into `tg/deterministic_sampling_merged` (merge `bd07b130b`) to de-risk the path and validate the findings in practice. This is extra context, not part of the deliverable.

**What the merge resolves:**

* *Validator-side replay now exists (Point 2 / Point 4).* `vllm/validation_sampling.py` provides `verify_sampling_from_logprobs` — a pure-CPU, single-position Check 2 replay over the decimal pipeline, zero tolerance. Its smoke test passes: it accepts a self-consistent token and rejects a tampered one.
* *Probability canonicalization is now defined (Point 3, blocker).* The single float→string point is `Decimal(repr(f))`, performed only on the vLLM side — exactly the recommended design; Go never formats a float.
* *Branches reconciled.* The decimal core is merged onto the v011 skeleton, and the previously non-collectable tests (Point 6) now resolve their symbols.
* *Latent bug fixed (not in the original review).* A module-level `getcontext()` that would have changed the whole process's default `Decimal` precision on import is now scoped via `localcontext()` (ADR 0002); verified — default `prec` stays 28 after import.

**What remains open:**

* *Executor is still on the float path — the new top blocker.* The executor and validator now compute the sampling weights by different arithmetic, so they cannot match bit-for-bit.
  * **Executor** (`topk_topp_sampler.py:95,317`) — GPU float, then round:
    ```python
    probs = logits.softmax(dim=-1, dtype=torch.float32)   # GPU float32, drifts across machines
    weights = (probs * 2**16).round().to(torch.int64)      # float multiply + float round; int only at the end
    ```
  * **Validator** (`validation_sampling.py`) — CPU decimal, exact:
    ```python
    weights = logprobs_to_weights({tid: repr(f) for ...}, ...)   # decimal arithmetic; bit-identical on any machine
    ```
  Both target the same `2^16` integer weights, but the arithmetic differs (GPU float + `.round()` vs decimal quantization), so under zero-tolerance Check 2 every real executor artifact false-rejects until the executor moves onto the decimal path. The fix is not "run the executor on CPU" — it must stay on GPU to run the model — but "compute the weights through the same decimal pipeline". Documented honestly as ADR 0003 (status: proposed).
* *The smoke test uses a self-consistent artifact.* It proves the replay logic is correct; it does not prove a real executor artifact passes. "Tests pass" must not be read as "works end to end."
* *Still not wired end to end.* Only the single-position pure function exists; the sequence-level orchestration in the serving path is deliberately not connected (ADR 0001), and the executor does not yet emit decimal-based artifacts.
* *Untouched:* seed hardening (Point 2), the Go reference (Point 6), and greedy handling (Point 6).

**Effect on the recommendation:** unchanged — still not ready — but the remaining gate list is now precise. Two pre-conditions, both requiring end-to-end GPU verification, stand between here and a meaningful log-only rollout: (1) move the executor weight computation onto the decimal pipeline (ADR 0003), and (2) protocol-force the seed.

Happy to help with any follow-up step, or to go deeper on a specific point.

<!-- >>>8 ----------------- END GitHub comment ----------------- 8<<< -->

---
---

<!--
============================================================================
  以下为中文自读版，方便你核对，请勿贴到 issue。完整翻译。
============================================================================
-->

# （自用，勿贴）中文对照 — 完整翻译

> 下面是上面英文评论的完整中文版，按 issue 的 6 个 Review Point 逐题作答，每条结论都落在真实源码上。代码标识符、`seed`、`artifact`、`blocker` 等保留英文。

## 审查：用于推理验证的可复现采样

我读了提案和这套确定性采样代码，在一台干净机器上（Python 3.10.12，纯 CPU，无 torch/GPU）重跑了"可复现"的部分，并重点排查"正式 enforcing 之前哪些口子能被绕过"。下面每个 Review Point 都对照真实源码作答。

**一句话结论 —— 最终建议：not ready。** 两阶段设计本身没问题、RNG 核心也正确；但东西躺在两条互不兼容、没合并的分支上；验证端重放不存在；seed 偏离提案且可被离线枚举；artifact 的浮点→字符串格式没定规范；又没有 Go 参考，所以"跨机器算出同一结果"这个核心主张连一侧都没实测。

**审查范围。** issue 只点名一条分支：`gonka-ai/vllm @ tg/detemrinistic_sampling_dump`（commit `4ce45cf5e`，@tamazgadaev）。它装着设计文档、RNG 核、以及 **decimal 整数权重** 管线，但没有集成。这是审查的正主。

审查中发现：它的测试 import 了本分支不存在的模块与符号（见 Point 6）。这些位于同一作者的另一条分支 `tg/deterministic_sampling_v011`，后者带着更早的**浮点**采样路线、已接入 V1 sampler（`DeterministicSampler`、`VLLM_DETERMINISTIC_SAMPLING` 开关、`SamplingMetadata` 字段）。两条分支有共同祖先，但彼此独立、未合并（`v011` 不是 `dump` 的祖先，且基于更新的 vLLM 版本）。本报告只在"解释 dump 分支状态"时引用 `v011`，它本身不在 issue 范围内。

---

### Review Point 1 — 与提案修复方案的一致性

> 核对当前实现是否与提案的两阶段方向一致；重点看是否正确支持：序列/采样重放验证、既有分布验证、既定检查顺序、对已知弱点的既定防护。

两阶段形态对（先序列检查、不符即拒、再分布检查）。顺序和既有分布检查没问题。两个薄弱点：**seed 推导偏离提案**、**验证端重放没实现**。

seed 推导偏离提案：

* **提案：** `run_seed = SHA256(user_seed ‖ inference_id_from_chain)`——绑定 `inference_id_from_chain`（链上分配、executor 选不了）。
* **代码：** `f"{seed}|{prompt_token_ids}"`（`docs/DETERMINISTIC_SAMPLING_VALIDATION.md:498`、`:176`）——改绑 `prompt_token_ids`。
* **后果：** 这处偏离既是 Point 2 离线枚举弱点的根，又引入 tokenizer 确定性依赖——验证端必须重建出一模一样的 token ids 才能复原 seed，tokenizer/引擎版本一漂就 false reject。

| 子项 | 状态 |
|---|---|
| 序列/采样重放验证 | 原语在；验证端重放没建（Point 2）；seed 偏离 |
| 既有分布验证 | 支持——gonka（Go）的 Stage 2，没动 |
| 检查顺序 | 设计一致 |
| 对已知弱点的防护 | 不完整——仅当 seed 不可被离线枚举（Point 2）且 `temperature > 0`（Point 6）时成立 |

### Review Point 2 — seed / RNG / 重放逻辑

> 核对 seed 生成、RNG 初始化、重放逻辑是否与提案和文档一致；验证：验证端能否从 artifact 和 seed 可靠地重放采样。

RNG 初始化和出数序列正确且可复现。拿固定输入跑核心模块：

```
iter_u64("reference_seed_v1", 5)[0]   = 4286832458236889005
decimal_sample(seed="42|[1,2,3]") x2  = 791 791
sum(logprobs_to_weights(...))         = 65536
```

同一个 seed 多次运行得到相同的 token，decimal 管线整数权重之和精确等于 `2^16`。这些是稳定的参考值——上面第一个 u64 就是将来 Go 移植版必须 bit 级对上的值。

**seed 生成与提案不一致、且可被离线枚举（blocker）。**

* **为何可利用：** 所有 seed 输入都由请求方控制——`{seed}` 请求方填，生产用 `rand.Int31()` 且保留用户自带 seed。
* **攻击方式：** 自营 executor 用便宜模型生成序列，离线换 seed 试到某个能通过序列检查的，提交它。
* **提案也只部分安全：** `user_seed` 开发者可填，`SHA256(user_seed ‖ inference_id)` 仍可枚举；seed 至少要有一个 executor 真选不了的成分，如 `executor_addr` + `start_block_hash`。

**验证端现在无法可靠重放，因为重放代码不存在。** 两条分支上都没有 `validation_sampling.py`，重放逻辑只在一次性脚本里出现过；就算写了，也会被 Point 3 的浮点→字符串问题坑（两端规范化方式不同 → 边界处抽到不同 token → 误拒）。

### Review Point 3 — artifact 内容

> 核对 artifact 是否含重放和验证所需的全部内容；把提案的 artifact 要求与文档的 artifact 格式对比。

提案要 `run_seed` + 每位 top-k token **及其概率**。但承载类型只装了 token，概率编码又没规范化规则。

`EnforcedToken` 不带概率（`vllm/validation.py:13`）：

```python
class EnforcedToken(BaseModel):
    token: str
    top_tokens: List[str] = Field(default_factory=list)   # 只有 token 字符串——没有 logprobs
    token_id: Optional[int] = Field(default=None, exclude=True)
    top_token_ids: List[int] = Field(default_factory=list, exclude=True)
```

**Blocker —— 浮点→字符串 canonicalization 未定义。** 管线把概率当字符串吃进去，但没规定字符串从哪个浮点来，而这直接改变取值：

* **同一个数、两种字符串：** `repr(-0.05)` 是 `"-0.05"`，而同一个数作为 float32 加宽到 float64 后是 `"-0.05000000074505806"`——canonical 字符串完全取决于那串 float64 比特。
* **跨语言分歧：** Python(`repr`) 与 Go(Ryu) 在最短浮点格式化上已知不一致（golang/go#17997），执行端与验证端可能产生不同字符串而误拒。
* **修法：** 存模型实际 float32→f64 的值、唯一约定化；浮点→字符串只在 vLLM 侧做，绝不让 Go 格式化浮点。

### Review Point 4 — 实现状态

> 识别：哪些已存在、哪些要改、哪些要新建、文档计划是否仍准确、现有代码能否软性纳入 MLNode。

两条平行架构下注，它们的差别就是整个故事的核心。`v011` 在 GPU 浮点上采样（`deterministic_sampler.py::deterministic_sample`）：

```python
# 计算 softmax 概率
probs = logits.softmax(dim=-1, dtype=torch.float32)   # GPU float32——跨机器会变
...
token_id = sample_categorical(probs_cpu[i], rng)      # 在浮点上做分类采样
```

`dump` 把它换成 decimal 整数权重管线（`deterministic_utils.py` 定义 `logprobs_to_weights` + `sample_categorical_weights`）。值得注意：v011 自己那个整数函数文档写明*"为跨语言可复现，因为概率在浮点舍入下可能不同"*——作者早看出浮点不可复现、预埋了整数方案；`dump` 扩成完整管线却丢了集成。两分支从未 reconcile。

| | `v011` | `dump` |
|---|---|---|
| 采样数学 | 浮点（`softmax(float32)` + 分类） | decimal 整数权重管线 |
| `DeterministicSampler`（708 行） | 有 | 无 |
| `VLLM_DETERMINISTIC_SAMPLING` + metadata 字段 | 有 | 无 |
| sampler 真正被 worker 调用 | **没有** | 没有 |

两条分支上 worker 都没调用 sampler。`DeterministicSampler` 只被自身模块和一个测试引用，`gpu_model_runner` 从不实例化它：

```
$ git grep -l DeterministicSampler origin/tg/deterministic_sampling_v011
vllm/v1/sample/deterministic_sampler.py
tests/v1/sample/test_deterministic_sampler.py
```

* **已存在：** RNG + decimal 核（`dump`）；集成骨架 + 浮点采样器（`v011`）；设计文档。
* **需修改：** 把 v011 浮点采样换成 decimal 管线；测试改指向；改 seed 推导。
* **需新建：** worker 层调用；验证端重放（`validation_sampling.py`）；协议强制 seed；canonicalization 契约；Go 参考。
* **文档计划是否准确？** 部分——它写了在 `gpu_model_runner.py` 接 RNG，但 `dump` 上没有；且计划早于 float→decimal 分裂。
* **能否软性纳入？** 不能（Point 5）。

### Review Point 5 — 安全的 MLNode 上线

> 识别：哪些可放开关后、哪些可非强制运行、收什么数据、什么应阻断强制、现在哪些安全、哪些该等。

* **现在可放开关后：** 只有独立无副作用的部分——RNG、decimal 核、离线对拍/参考向量工具。不碰线上请求路径。
* **可 log-only 运行：** 暂无——log-only 需要 sampler 在生成循环里跑并吐 artifact，但据 Point 4，sampler 从没被 worker 调用。接好后 log-only（算判定、记录、不拒）是第一步。
* **收什么数据：** 见 Expected Output。
* **应阻断强制：** 可被离线枚举的 seed；未定义的 canonicalization；缺 Go 参考 + 未测跨机器误拒率；greedy 流量无 Stage 1 覆盖。
* **现在安全可纳入：** 离线数学核 + 对拍工具。
* **该等：** 一切 gate/拒绝/惩罚、依赖 seed 的接受判定、浮点路线。

### Review Point 6 — 漏洞与边界

> 在它成为 enforcing 逻辑前仔细排查漏洞/边界；聚焦实际机制和已记录限制。

* **seed 离线枚举（blocker）** —— Point 2。
* **浮点→字符串不匹配（blocker）** —— Point 3。
* **无跨语言参考。** 两分支都只有 Python；没有 Go 实现，跨语言可复现性只是断言、两侧零实测——仅次于 seed 的第二大风险。
* **greedy 解码绕过 RNG。** 在 `deterministic_sampler.py` 里，当 `temperature < _SAMPLING_EPS`（1e-5）时 sampler 走 `greedy_sample`，也就是 `logits.argmax(dim=-1)`，完全不碰 RNG。所以 `temperature == 0` 时序列检查毫无防护，pre-fill 防护只在 `temperature > 0` 成立。代码还应用 `temperature is not None` 判断，而非依赖 falsy-zero。
* **提交的测试跑不起来。** `tests/v1/sample/test_deterministic_standalone.py` 从 `deterministic_utils` import 了 `sample_categorical` 和 `WeightedPrefixSampler`，但 `dump` 核这两个都没定义（它有的是 `sample_categorical_weights`、`logprobs_to_weights`、`decimal_sample_from_logprobs`），于是模块在 import/collect 阶段就失败。这些符号在 `v011` 上有、`dump` 上没有——测试属于另一分支的 API，两分支 reconcile 前不能算"通过"证据。
* **二值结论——无错误归因、无版本门控。** `verify_sampling_from_logprobs(...) -> bool` 只返回 honest/fraud（`return replayed_token == reported_token`）；artifact（`EnforcedToken`）上没有版本字段，代码里也没有任何错误归因。因此由**验证器自身**造成的不匹配——tokenizer/引擎不同、浮点行为有别、或模型版本不被支持——与 executor 作弊无法区分，诚实的 executor 会被误判。enforcing 前，artifact 需带版本/构建标记（模型 + 引擎 + tokenizer + 权重管线），且验证器应据此门控：版本不支持/不匹配应归类为 *inconclusive / 版本不支持*，而非 fraud；这意味着把结论从 bool 扩展为分类结果（honest / fraud / inconclusive）。这是安全 enforcing 的前提，并会放大 Point 1、3、5 的误拒风险。
* **enforcing 前要锁死的已记录限制：** 确定性模式下 `logit_bias` 被禁（和原版 vLLM 不一致）；`top_k` 被截到 `max_num_logprobs`；要求 CPython/libmpdec（PyPy/Jython 会不一致）；途中任何截断 JSON 浮点精度的 proxy/logger 都会破坏分布比较。

---

### Expected Output（issue 要的清单）

* **是否符合两阶段提案？** 部分——结构和 RNG 一致；seed 偏离；验证端重放没建（Point 1）。
* **已实现哪些？** RNG + decimal 核（`dump`）；集成骨架 + 浮点采样器（`v011`）；设计文档——两 sampler 都没在 worker 里跑（Point 4）。
* **要改/要建哪些？** 改：浮点→decimal 采样、测试、seed。建：worker 调用、验证端重放、协议强制 seed、canonicalization 契约、Go 参考（Point 4）。
* **能否软接入？** 不能（Point 5）。
* **哪些保持非强制？** 一切 gate/拒绝/惩罚、依赖 seed 的接受判定、浮点路线（Point 5）。
* **严格验证前收什么数据？**（1）decimal 管线在真实 GPU/驱动/库差异下的跨机器重放一致率/误拒率；（2）Go 参考 + Python↔Go bit 级一致性向量；（3）`temperature == 0`（greedy）流量占比；（4）真实 proxy/日志路径上实测的 JSON 浮点精度损失。
* **发现的漏洞/不匹配：** seed 离线枚举、未定义 canonicalization、无 Go 参考、greedy 绕过、测试跑不起来、二值结论无错误归因/版本门控、已记录限制（Point 6）。
* **建议后续步骤：**（1）采用 **decimal** 路线而非浮点；（2）合并分支——保留 v011 骨架、换上 decimal 管线、测试改指向；（3）关掉两个 blocker——协议强制 seed + 单一由 vLLM 独占的浮点→字符串 canonicalization；（4）补 Go 参考 + 一致性向量；（5）然后 log-only 上线 → 测误拒率 → 再考虑 enforcing。
* **最终建议：not ready。** 机制有前景、路径清晰，但软接入被卡在：代码没在 worker 里跑、seed 有安全弱点、artifact canonicalization 未定、无跨语言参考。当前风险低仅因还没接线；sampler 端到端跑通 + 两个 blocker 关闭前，连非强制路径也不建议纳入。

---

## 附录 —— 分支合并（超出 issue 要求范围，锦上添花）

issue 只要求审查、并未要求改代码。作为额外的后续工作，我们把两条分支合并为 `tg/deterministic_sampling_merged`（合并提交 `bd07b130b`），以降低后续风险、并在实践中验证上述发现。此节为附加背景，不属于交付物本身。

**合并解决了哪些问题：**

* *验证端重放现已存在（Point 2 / Point 4）。* `vllm/validation_sampling.py` 提供 `verify_sampling_from_logprobs`——纯 CPU、单 token 位置、走十进制管线、零容差的 Check 2 重放。其冒烟测试通过：接受自洽 token、拒绝被篡改的 token。
* *概率规范化现已定义（Point 3，原 blocker）。* 唯一的浮点→字符串转换点为 `Decimal(repr(f))`，且只在 vLLM 侧执行——正是建议的设计；Go 侧不格式化任何浮点。
* *两分支已合并。* 十进制核心并入 v011 骨架；此前无法收集的测试（Point 6）现已能解析到符号。
* *修复了一个潜在缺陷（原审查未涉及）。* 原本模块级的 `getcontext()` 会在 import 时改变整个进程的默认 `Decimal` 精度，现改用 `localcontext()` 局部化（ADR 0002）；已验证——import 后默认 `prec` 仍为 28。

**仍然遗留哪些问题：**

* *executor 仍走浮点路径——这是新的头号阻断项。* executor 与 validator 现在用**不同的运算方式**算采样权重，无法 bit 级对齐：
  * **executor**（`topk_topp_sampler.py:95,317`）—— GPU 浮点、再 round：
    ```python
    probs = logits.softmax(dim=-1, dtype=torch.float32)   # GPU float32，跨机器漂移
    weights = (probs * 2**16).round().to(torch.int64)      # 浮点乘 + 浮点 round，最后才转 int
    ```
  * **validator**（`validation_sampling.py`）—— CPU decimal、精确：
    ```python
    weights = logprobs_to_weights({tid: repr(f) for ...}, ...)   # decimal 运算，任何机器 bit 一致
    ```
  两边目标相同（`2^16` 整数权重），但运算方式不同（GPU 浮点 + `.round()` vs decimal 量化），因此零容差的 Check 2 会对**每一个真实 executor artifact 假拒**，直到 executor 迁到 decimal 路径。修法不是"把 executor 搬到 CPU"（它必须留在 GPU 跑模型），而是"改用同一套 decimal 管线算权重"。已诚实记录为 ADR 0003（状态：proposed）。
* *冒烟测试使用的是自洽 artifact。* 它证明重放逻辑正确，但不证明真实 executor artifact 能通过。"测试通过"不可解读为"端到端可用"。
* *仍未端到端接线。* 目前仅有单位置纯函数；serving 路径中的序列级编排被有意暂不连接（ADR 0001），且 executor 尚未产出基于十进制的 artifact。
* *未触及：* seed 加固（Point 2）、Go 参考实现（Point 6）、greedy 处理（Point 6）。

**对最终建议的影响：** 不变——仍为 not ready——但剩余闸门已清晰。距离有意义的 log-only 上线，尚有两个必须在 GPU 上端到端验证的前置条件：（1）将 executor 的权重计算迁移到十进制管线（ADR 0003）；（2）对 seed 施加协议强制。
