<!--
  English review is the post-ready section (between the >>>8 markers).
  The Chinese section below it is an internal translation — do not post it.
  Structure: Summary → Current state → Findings → Review Points → Expected Output.
-->

<!-- >>>8 ---------------- BEGIN GitHub comment ---------------- 8<<< -->

## Review: reproducible sampling for inference validation

**Final recommendation: not ready.** The two-stage design is sound and the RNG core is correct and reproducible, but the validation loop is not wired into the generation path, two hard blockers remain (an executor-controllable seed and an unspecified probability encoding), and there is no cross-language reference implementation, so the "same result on executor and validator" property is untested. The immediate risk is low only because nothing is enforced yet.

This review is organised in three layers: the current state of the code, a ranked list of findings, and direct answers to the six Review Points that reference those findings. Claims backed by a local run were executed on Python 3.10.12, CPU only, no torch/GPU.

---

## Current state

**Intended design (from the proposal).** A cheap Stage 1 "sequence check" runs before the existing Stage 2 "distribution check". Stage 1 replaces random sampling with a seeded, fully specified RNG so the validator can replay each token: initialise the RNG from `run_seed`, draw a token from the recorded top-k distribution at each position, and reject if the replayed token differs from the one the executor claims.

**Implementation.** The work exists on two divergent, unreconciled branches:

| | `tg/deterministic_sampling_v011` | `tg/detemrinistic_sampling_dump` (`4ce45cf5e`) |
|---|---|---|
| Sampling arithmetic | float — GPU `softmax(float32)` then categorical draw | decimal — integer-weight pipeline (`logprobs_to_weights`) |
| `DeterministicSampler` (V1 sampler module) | present (708 lines) | absent |
| `VLLM_DETERMINISTIC_SAMPLING` flag, metadata fields | present | absent |
| Design document | absent | present |

**What is wired vs. missing.** RNG core and the decimal pipeline exist (`dump`); the float sampler and integration skeleton exist (`v011`). Not present on either branch: worker-level invocation of the sampler, validator-side replay, a Go reference implementation.

**Verified working.** The RNG and decimal pipeline are deterministic. Running the core module against fixed inputs:

```
iter_u64("reference_seed_v1", 5)[0]   = 4286832458236889005
decimal_sample(seed="42|[1,2,3]") x2  = 791 791
sum(logprobs_to_weights(...))         = 65536
```

The same seed produces the same token across runs, and the integer weights sum to exactly `2^16`. The first value is a reference vector a Go port must reproduce bit-for-bit.

---

## Findings

**Finding 1 — Seed is executor-controllable (offline grinding). Severity: blocker.**
The proposal derives the seed as `SHA256(user_seed ‖ inference_id_from_chain)`; the code uses `f"{seed}|{prompt_token_ids}"` (`docs/DETERMINISTIC_SAMPLING_VALIDATION.md:498`, `:176`), and production supplies the seed via `rand.Int31()` while preserving a user-provided seed. Because every seed input is request-controlled, a self-dealing executor can search offline for a seed under which a cheap-model sequence replays correctly, then submit it — defeating Stage 1 regardless of RNG correctness. (Even the proposal's formula is only partly safe, since `user_seed` is caller-supplied.) The seed must be protocol-forced from values the executor cannot choose, e.g. `H(prompt_hash ‖ inference_id ‖ executor_addr ‖ start_block_hash)`. Binding to `prompt_token_ids` also creates a tokenizer-determinism dependency: any tokenizer or engine drift changes the seed and forces a false reject.

**Finding 2 — Probability encoding (float→string) is unspecified. Severity: blocker.**
The decimal pipeline consumes probabilities as strings, but the canonical string is undefined and depends on the exact float64 bits: `repr(-0.05)` is `"-0.05"`, whereas the same value as float32 widened to float64 is `"-0.05000000074505806"`. Python (`repr`) and Go (Ryu) are known to differ on shortest-float formatting (golang/go#17997), so executor and validator can produce different strings for the same probability and false-reject. The artifact must store the model's actual float32-widened-to-float64 value under a single canonicalization owned by vLLM; Go should never format a float.

**Finding 3 — The validation loop is not wired up. Severity: high.**
Neither side runs end to end. On the executor side, `DeterministicSampler` is referenced only by its own module and one test — `gpu_model_runner` never instantiates it:

```
$ git grep -l DeterministicSampler origin/tg/deterministic_sampling_v011
vllm/v1/sample/deterministic_sampler.py
tests/v1/sample/test_deterministic_sampler.py
```

On the validator side, the replay logic (expected in `validation_sampling.py`) does not exist on either branch; it appears only in standalone scripts. Nothing can run in log-only mode until this is closed.

**Finding 4 — The float sampling path is not reproducible across machines. Severity: high.**
`v011` samples on GPU-computed `softmax(float32)` output, which varies with GPU, driver, and library versions. Its own integer-weight helper is documented as being *"for cross-language reproducibility when probs may differ in float rounding"* — i.e. the author already identified this and pre-staged the decimal fix that `dump` later developed. Shipping the float path would produce false rejects across heterogeneous nodes; the decimal path is the one to adopt.

**Finding 5 — No cross-language reference implementation. Severity: high.**
Both branches are Python-only. The cross-language reproducibility the scheme depends on has zero empirical test, because the Go side does not exist. Given Findings 2 and 4, this is the largest unquantified risk after the seed.

**Finding 6 — The artifact does not carry probabilities. Severity: medium.**
The proposal requires per-position top-k tokens **and their probabilities**, but `EnforcedToken` carries only the tokens (`vllm/validation.py:13`):

```python
class EnforcedToken(BaseModel):
    token: str
    top_tokens: List[str] = Field(default_factory=list)   # token strings only — no logprobs
```

The probabilities travel on a separate channel, so the Stage 1 and Stage 2 artifacts need to be reconciled into a single replay record.

**Finding 7 — Greedy decoding bypasses the RNG. Severity: medium.**
For `temperature < _SAMPLING_EPS` (1e-5) the sampler takes `logits.argmax(dim=-1)` and never consults the RNG (`deterministic_sampler.py`). Stage 1 therefore provides no protection at temperature 0; the pre-fill defense holds only for `temperature > 0`. This should be documented, and the branch condition should test `temperature is not None` rather than a falsy zero.

**Finding 8 — Committed tests reference a removed API. Severity: low.**
`tests/v1/sample/test_deterministic_standalone.py` imports `sample_categorical` and `WeightedPrefixSampler`, which exist on `v011` but were removed from the `dump` core (which defines `sample_categorical_weights`, `logprobs_to_weights`, `decimal_sample_from_logprobs`). The module therefore fails at collection. The tests belong to the other branch's API and cannot serve as passing evidence until the branches are reconciled.

---

## Review Points

**1 — Alignment with the proposed fix.** The two-stage structure, ordering, and existing distribution check match the proposal. The RNG matches. Two elements do not: seed derivation (Finding 1) and validator replay, which is unbuilt (Finding 3).

| Proposal element | Status |
|---|---|
| Two-stage ordering (sequence → distribution) | Matches |
| RNG (SHA256 counter mode) | Matches |
| Seed derivation | Diverges — Finding 1 |
| Artifact (tokens + probabilities) | Partial — Finding 6 |
| Validator replay | Not built — Finding 3 |
| Distribution check | Matches (existing Go) |

**2 — Seed, RNG, and replay logic.** RNG initialisation and draw sequence are correct and reproducible (see Current state). Seed generation diverges from the proposal and is exploitable via offline grinding (Finding 1). The validator cannot replay today because the replay implementation does not exist (Finding 3), and once written it is exposed to the encoding issue in Finding 2.

**3 — Artifact contents.** The seed is present but bound to the wrong input (Finding 1); the top-k tokens are present; the probabilities are not in the enforced-token type (Finding 6); and the probability encoding has no canonicalization rule (Finding 2).

**4 — Implementation status.** See Current state for the two-branch map.
- *Exists:* RNG + decimal core (`dump`); float sampler + integration skeleton (`v011`); design document.
- *Needs modification:* replace the float sampler with the decimal pipeline; change the seed derivation; repoint the tests.
- *Needs creation:* worker-level sampler invocation; validator replay; a protocol-forced seed; the canonicalization contract; a Go reference.
- *Plan accuracy:* partial — the document's plan predates the float→decimal split and describes wiring that is absent on `dump`.
- *Ready to include softly:* no (Review Point 5).

**5 — Safe MLNode rollout.**
- *Behind a flag now:* only the standalone RNG and decimal core, plus offline reference-vector tooling; these do not touch the request path.
- *Log-only mode:* nothing yet — the sampler is not invoked by the worker (Finding 3).
- *Blocks strict enforcement:* Findings 1, 2, 5, and the lack of a measured cross-machine false-reject rate; greedy traffic (Finding 7).
- *Safe now / should wait:* safe — the offline core; wait — anything that gates, rejects, or penalizes, and the float path.

**6 — Vulnerabilities and edge cases.** Offline seed grinding (Finding 1, blocker); undefined probability encoding (Finding 2, blocker); non-reproducible float path (Finding 4); no Go reference (Finding 5); greedy bypass (Finding 7); non-running tests (Finding 8). Documented limitations to lock down before enforcing: `logit_bias` disabled in deterministic mode, `top_k` clamped to `max_num_logprobs`, CPython/libmpdec required, and JSON float-precision truncation by any intermediary.

---

## Expected Output

* **Matches the two-stage proposal?** Partially — structure and RNG match; seed diverges; validator replay unbuilt.
* **Already implemented?** RNG + decimal core (`dump`); float sampler + skeleton (`v011`); design doc — neither runs in the worker.
* **Needs modification / creation?** Modify: sampler arithmetic, seed, tests. Create: worker invocation, validator replay, protocol-forced seed, canonicalization contract, Go reference.
* **Ready for soft MLNode integration?** No.
* **Should remain non-enforcing?** Everything that gates, rejects, or penalizes; all seed-dependent acceptance; the float path.
* **Data to collect before strict validation?** Cross-machine false-reject rate for the decimal pipeline across real GPU/driver/library variation; Python↔Go conformance vectors; the share of `temperature == 0` traffic; JSON float-precision loss through the real proxy/logging path.
* **Findings:** Findings 1–8 above.
* **Recommended next steps:** adopt the decimal path; reconcile the branches onto the `v011` integration skeleton; close Findings 1 and 2; add a Go reference and conformance vectors; then run log-only, measure the false-reject rate, and only then consider enforcement.
* **Final recommendation: not ready.**

---

## Addendum — branch reconciliation (beyond the requested scope)

The issue asks for a review, not for code changes. As optional follow-up we reconciled the two branches into `tg/deterministic_sampling_merged` (merge `bd07b130b`) to de-risk the path and validate the findings in practice. This is extra context, not part of the deliverable.

**What the merge resolves:**

* *Validator-side replay now exists (Finding 3, validator half).* `vllm/validation_sampling.py` provides `verify_sampling_from_logprobs` — a pure-CPU, single-position Check 2 replay over the decimal pipeline, zero tolerance. Its smoke test passes: it accepts a self-consistent token and rejects a tampered one.
* *Probability canonicalization is now defined (Finding 2, blocker).* The single float→string point is `Decimal(repr(f))`, performed only on the vLLM side — exactly the recommended design; Go never formats a float.
* *Branches reconciled.* The decimal core is merged onto the v011 skeleton, and the previously non-collectable tests (Finding 8) now resolve their symbols.
* *Latent bug fixed (not in the original review).* A module-level `getcontext()` that would have changed the whole process's default `Decimal` precision on import is now scoped via `localcontext()` (ADR 0002); verified — default `prec` stays 28 after import.

**What remains open:**

* *Executor is still on the float path — the new top blocker.* `topk_topp_sampler.py:317` still computes `weights = (probs * 2**16).round().to(torch.int64)` from GPU float32. This is not bit-compatible with the validator's decimal weights, so the zero-tolerance Check 2 will false-reject every real executor artifact until the executor adopts the decimal pipeline. The merge documents this honestly (ADR 0003, status: proposed) and claims nothing more.
* *The smoke test uses a self-consistent artifact.* It proves the replay logic is correct; it does not prove a real executor artifact passes. "Tests pass" must not be read as "works end to end."
* *Still not wired end to end.* Only the single-position pure function exists; the sequence-level orchestration in the serving path is deliberately not connected (ADR 0001), and the executor does not yet emit decimal-based artifacts.
* *Untouched:* seed hardening (Finding 1), the Go reference (Finding 5), and greedy handling (Finding 7).

**Effect on the recommendation:** unchanged — still not ready — but the remaining gate list is now precise. Two pre-conditions, both requiring end-to-end GPU verification, stand between here and a meaningful log-only rollout: (1) move the executor weight computation onto the decimal pipeline (ADR 0003), and (2) protocol-force the seed (Finding 1).

Happy to help with any follow-up, or to go deeper on a specific finding.

<!-- >>>8 ----------------- END GitHub comment ----------------- 8<<< -->

---
---

<!-- 中文版：内部参考，请勿贴到 issue。 -->

# 审查：用于推理验证的可复现采样（中文，内部参考）

**最终建议：not ready（未就绪）。** 两阶段设计合理、RNG 核心正确且可复现，但验证流程尚未接入生成路径，两个硬性阻断项（执行者可控的 seed、未定义的概率编码）未解决，且没有跨语言参考实现，因此"执行端与验证端结果一致"这一性质尚未经过验证。当前风险之所以低，仅因为尚未启用强制。

本审查分三层：代码现状、按严重度排序的发现（Findings）、以及对 6 个 Review Point 的作答（引用相应 Finding）。凡标注本地运行的结论，均在 Python 3.10.12、纯 CPU、无 torch/GPU 环境下执行。

---

## 代码现状

**既定设计（来自提案）。** 在既有的 Stage 2"分布检查"之前，先运行一个低成本的 Stage 1"序列检查"。Stage 1 用一个带种子、完全确定的 RNG 取代随机采样，使验证端可以逐 token 重放：用 `run_seed` 初始化 RNG，在每个位置从记录的 top-k 分布中抽取一个 token，若重放结果与执行者声称的 token 不一致则拒绝。

**实现。** 工作分散在两条互相偏离、尚未合并的分支上：

| | `tg/deterministic_sampling_v011` | `tg/detemrinistic_sampling_dump`（`4ce45cf5e`） |
|---|---|---|
| 采样算术 | 浮点 —— GPU `softmax(float32)` 后做分类抽样 | 十进制 —— 整数权重管线（`logprobs_to_weights`） |
| `DeterministicSampler`（V1 sampler 模块） | 有（708 行） | 无 |
| `VLLM_DETERMINISTIC_SAMPLING` 开关、metadata 字段 | 有 | 无 |
| 设计文档 | 无 | 有 |

**已接入 vs. 缺失。** RNG 核心与十进制管线已存在（`dump`）；浮点采样器与集成骨架已存在（`v011`）。两条分支均缺失：worker 层对采样器的调用、验证端重放、Go 参考实现。

**已验证可用。** RNG 与十进制管线是确定性的。以固定输入运行核心模块：

```
iter_u64("reference_seed_v1", 5)[0]   = 4286832458236889005
decimal_sample(seed="42|[1,2,3]") x2  = 791 791
sum(logprobs_to_weights(...))         = 65536
```

同一 seed 多次运行得到相同 token，整数权重之和精确等于 `2^16`。第一个值是 Go 移植版必须 bit 级复现的参考向量。

---

## 发现（Findings）

**Finding 1 —— seed 由执行者可控（离线枚举攻击）。严重度：阻断项。**
提案将 seed 推导为 `SHA256(user_seed ‖ inference_id_from_chain)`；代码使用 `f"{seed}|{prompt_token_ids}"`（`docs/DETERMINISTIC_SAMPLING_VALIDATION.md:498`、`:176`），且生产环境通过 `rand.Int31()` 提供 seed 并保留用户自带 seed。由于所有 seed 输入均由请求方控制，自营自利的执行者可离线搜索一个能让廉价模型序列正确重放的 seed 再提交，从而绕过 Stage 1（与 RNG 是否正确无关）。即便提案的公式也仅部分安全，因为 `user_seed` 由调用方提供。seed 必须由协议强制、取自执行者无法选择的值，例如 `H(prompt_hash ‖ inference_id ‖ executor_addr ‖ start_block_hash)`。绑定 `prompt_token_ids` 还引入 tokenizer 确定性依赖：tokenizer 或引擎版本一旦漂移，seed 即变化并导致误拒。

**Finding 2 —— 概率编码（浮点→字符串）未定义。严重度：阻断项。**
十进制管线以字符串形式接收概率，但 canonical 字符串未定义，且取决于具体的 float64 比特：`repr(-0.05)` 为 `"-0.05"`，而同一数值以 float32 加宽到 float64 时为 `"-0.05000000074505806"`。Python（`repr`）与 Go（Ryu）在最短浮点格式化上已知存在差异（golang/go#17997），因此执行端与验证端可能为同一概率生成不同字符串而误拒。artifact 必须存储模型实际输出的 float32→float64 数值，并采用由 vLLM 独占的单一规范化；Go 侧不应格式化任何浮点。

**Finding 3 —— 验证流程尚未接入。严重度：高。**
两端均未端到端运行。执行端侧，`DeterministicSampler` 仅被其自身模块和一个测试引用，`gpu_model_runner` 从不实例化它：

```
$ git grep -l DeterministicSampler origin/tg/deterministic_sampling_v011
vllm/v1/sample/deterministic_sampler.py
tests/v1/sample/test_deterministic_sampler.py
```

验证端侧，重放逻辑（应位于 `validation_sampling.py`）在两条分支上均不存在，仅出现在独立脚本中。在此闭合之前，无法以 log-only 模式运行。

**Finding 4 —— 浮点采样路径不具备跨机器可复现性。严重度：高。**
`v011` 在 GPU 计算的 `softmax(float32)` 结果上采样，而该结果随 GPU、驱动、库版本变化。其自身的整数权重辅助函数被文档标注为*"用于跨语言可复现，因为概率在浮点舍入下可能不同"*——即作者已识别此问题并预置了 `dump` 后续发展出的十进制方案。若上线浮点路径，将在异构节点间产生误拒；应采用十进制路径。

**Finding 5 —— 缺少跨语言参考实现。严重度：高。**
两条分支均仅有 Python。方案所依赖的跨语言可复现性零实测，因为 Go 侧并不存在。结合 Finding 2 与 Finding 4，这是除 seed 外最大的未量化风险。

**Finding 6 —— artifact 不携带概率。严重度：中。**
提案要求每个位置的 top-k tokens **及其概率**，但 `EnforcedToken` 仅携带 tokens（`vllm/validation.py:13`）：

```python
class EnforcedToken(BaseModel):
    token: str
    top_tokens: List[str] = Field(default_factory=list)   # 仅 token 字符串——无 logprobs
```

概率走独立通道，因此 Stage 1 与 Stage 2 的 artifact 需要整合为单一重放记录。

**Finding 7 —— greedy 解码绕过 RNG。严重度：中。**
当 `temperature < _SAMPLING_EPS`（1e-5）时，采样器执行 `logits.argmax(dim=-1)` 而从不调用 RNG（`deterministic_sampler.py`）。因此 Stage 1 在 temperature 0 时不提供任何保护；pre-fill 防护仅在 `temperature > 0` 时成立。应予以说明，且分支条件应判断 `temperature is not None`，而非依赖 falsy zero。

**Finding 8 —— 提交的测试引用了已移除的 API。严重度：低。**
`tests/v1/sample/test_deterministic_standalone.py` 导入了 `sample_categorical` 与 `WeightedPrefixSampler`，二者存在于 `v011`，但已从 `dump` 核心移除（后者定义 `sample_categorical_weights`、`logprobs_to_weights`、`decimal_sample_from_logprobs`）。因此该模块在收集阶段失败。这些测试属于另一分支的 API，在两分支合并前不能作为"通过"证据。

---

## Review Points 作答

**1 —— 与提案修复方案的一致性。** 两阶段结构、顺序、既有分布检查均与提案一致，RNG 亦一致。两处不一致：seed 推导（Finding 1）与尚未构建的验证端重放（Finding 3）。

| 提案要素 | 状态 |
|---|---|
| 两阶段顺序（序列 → 分布） | 一致 |
| RNG（SHA256 计数器模式） | 一致 |
| seed 推导 | 偏离 —— Finding 1 |
| artifact（tokens + 概率） | 部分 —— Finding 6 |
| 验证端重放 | 未构建 —— Finding 3 |
| 分布检查 | 一致（既有 Go） |

**2 —— seed、RNG 与重放逻辑。** RNG 初始化与出数序列正确且可复现（见代码现状）。seed 生成偏离提案，且可通过离线枚举利用（Finding 1）。验证端目前无法重放，因为重放实现不存在（Finding 3）；即便实现，也暴露于 Finding 2 的编码问题。

**3 —— artifact 内容。** seed 存在但绑定了错误的输入（Finding 1）；top-k tokens 存在；概率不在 enforced-token 类型中（Finding 6）；概率编码无规范化规则（Finding 2）。

**4 —— 实现状态。** 两分支概览见代码现状。
- *已存在：* RNG + 十进制核心（`dump`）；浮点采样器 + 集成骨架（`v011`）；设计文档。
- *需修改：* 将浮点采样器替换为十进制管线；修改 seed 推导；调整测试导入。
- *需新建：* worker 层采样器调用；验证端重放；协议强制的 seed；规范化契约；Go 参考实现。
- *计划准确性：* 部分——文档计划早于 float→decimal 分裂，且描述了 `dump` 上并不存在的接线。
- *能否软性纳入：* 否（见 Review Point 5）。

**5 —— 安全的 MLNode 上线。**
- *现在可置于开关后：* 仅独立的 RNG 与十进制核心，以及离线参考向量工具；均不触及请求路径。
- *log-only 模式：* 暂无——采样器未被 worker 调用（Finding 3）。
- *阻断强制启用：* Finding 1、2、5，以及缺少实测的跨机器误拒率；greedy 流量（Finding 7）。
- *现在安全 / 应等待：* 安全——离线核心；应等待——任何 gate/拒绝/惩罚逻辑，以及浮点路径。

**6 —— 漏洞与边界。** 离线 seed 枚举（Finding 1，阻断项）；未定义的概率编码（Finding 2，阻断项）；不可复现的浮点路径（Finding 4）；缺少 Go 参考（Finding 5）；greedy 绕过（Finding 7）；测试无法运行（Finding 8）。启用强制前需锁定的已记录限制：确定性模式下禁用 `logit_bias`、`top_k` 被截断至 `max_num_logprobs`、要求 CPython/libmpdec、以及任何中间环节对 JSON 浮点精度的截断。

---

## Expected Output

* **是否符合两阶段提案？** 部分——结构与 RNG 一致；seed 偏离；验证端重放未构建。
* **已实现哪些？** RNG + 十进制核心（`dump`）；浮点采样器 + 骨架（`v011`）；设计文档——两者均未在 worker 中运行。
* **需修改 / 新建哪些？** 修改：采样算术、seed、测试。新建：worker 调用、验证端重放、协议强制 seed、规范化契约、Go 参考。
* **能否软性接入 MLNode？** 不能。
* **哪些应保持非强制？** 一切 gate/拒绝/惩罚逻辑；所有依赖 seed 的接受判定；浮点路径。
* **严格验证前应收集哪些数据？** 十进制管线在真实 GPU/驱动/库差异下的跨机器误拒率；Python↔Go 一致性向量；`temperature == 0` 流量占比；真实 proxy/日志路径上的 JSON 浮点精度损失。
* **发现：** 上述 Finding 1–8。
* **建议后续步骤：** 采用十进制路径；将工作合并到 `v011` 集成骨架上；闭合 Finding 1 与 2；补充 Go 参考与一致性向量；随后运行 log-only、测量误拒率，之后再考虑强制。
* **最终建议：not ready（未就绪）。**

---

## 附录 —— 分支合并（超出 issue 要求范围，锦上添花）

issue 只要求审查、并未要求改代码。作为额外的后续工作，我们把两条分支合并为 `tg/deterministic_sampling_merged`（合并提交 `bd07b130b`），以降低后续风险、并在实践中验证上述发现。此节为附加背景，不属于交付物本身。

**合并解决了哪些问题：**

* *验证端重放现已存在（Finding 3 的验证端一半）。* `vllm/validation_sampling.py` 提供 `verify_sampling_from_logprobs`——纯 CPU、单 token 位置、走十进制管线、零容差的 Check 2 重放。其冒烟测试通过：接受自洽 token、拒绝被篡改的 token。
* *概率规范化现已定义（Finding 2，原阻断项）。* 唯一的浮点→字符串转换点为 `Decimal(repr(f))`，且只在 vLLM 侧执行——正是建议的设计；Go 侧不格式化任何浮点。
* *两分支已合并。* 十进制核心并入 v011 骨架；此前无法收集的测试（Finding 8）现已能解析到符号。
* *修复了一个潜在缺陷（原审查未涉及）。* 原本模块级的 `getcontext()` 会在 import 时改变整个进程的默认 `Decimal` 精度，现改用 `localcontext()` 局部化（ADR 0002）；已验证——import 后默认 `prec` 仍为 28。

**仍然遗留哪些问题：**

* *executor 仍走浮点路径——这是新的头号阻断项。* `topk_topp_sampler.py:317` 仍以 GPU float32 计算 `weights = (probs * 2**16).round().to(torch.int64)`。它与验证端的十进制权重不 bit 兼容，因此在 executor 改用十进制管线之前，零容差的 Check 2 会对**每一个真实 executor artifact 假拒**。合并方对此诚实记录（ADR 0003，状态：proposed），未作过度声称。
* *冒烟测试使用的是自洽 artifact。* 它证明重放逻辑正确，但不证明真实 executor artifact 能通过。"测试通过"不可解读为"端到端可用"。
* *仍未端到端接线。* 目前仅有单位置纯函数；serving 路径中的序列级编排被有意暂不连接（ADR 0001），且 executor 尚未产出基于十进制的 artifact。
* *未触及：* seed 加固（Finding 1）、Go 参考实现（Finding 5）、greedy 处理（Finding 7）。

**对最终建议的影响：** 不变——仍为 not ready——但剩余闸门已清晰。距离有意义的 log-only 上线，尚有两个必须在 GPU 上端到端验证的前置条件：（1）将 executor 的权重计算迁移到十进制管线（ADR 0003）；（2）对 seed 施加协议强制（Finding 1）。
