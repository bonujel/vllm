# 拼装方案：合并 v011 + dump 两条确定性采样分支

> 目标：以 **`origin/tg/deterministic_sampling_v011`** 为底，把 **`origin/tg/detemrinistic_sampling_dump`** 的 decimal 重放管线挑进来，产出一条**可在纯 CPU 机器上验证**的合并分支。
>
> 这不是 `git merge`（dump 相对 main 改了 244 个文件，且两分支 `deterministic_utils.py` 的 API 不兼容）。这是一次**精挑细选的移植**。
>
> 状态：待 review，尚未动代码。

---

## 0. 为什么以 v011 为底

调研核实后，两分支的实际分工和原 review 报告的转述有出入，这里以**实测**为准：

| | `v011`（33 文件，6495 行） | `dump`（244 文件） |
|---|---|---|
| `Sha256CounterRNG` / `iter_u64` / `uint64_below` / `sample_categorical_weights` | ✅ | ✅（**同源，实现逐行一致**） |
| 集成骨架：`VLLM_DETERMINISTIC_SAMPLING` flag、`SamplingMetadata.deterministic_rngs`、`gpu_model_runner`/`gpu_input_batch`/`topk_topp_sampler` 接线 | ✅ | ❌ |
| 生产采样路径用整数权重抽样（`topk_topp_sampler.deterministic_sample` → `sample_categorical_weights`） | ✅ 但**权重来源不可复现** ⚠️ | ❌ |
| `EnforcedToken` 已带 `logprobs` + `sampling_weights` 字段、`EnforcedTokens.from_content()` 序列化 | ✅ | 仅 token 字段（更弱） |
| `WeightedPrefixSampler`、浮点 `sample_categorical` | ✅ | ❌ |
| **`logprobs_to_weights`（Decimal 从 logprob 字符串重建整数权重）** | ❌ | ✅ |
| **`decimal_sample_from_logprobs`（端到端重放：字符串→token）** | ❌ | ✅ |
| 903 行设计文档 `docs/DETERMINISTIC_SAMPLING_VALIDATION.md` | ❌ | ✅ |

**结论**：以 v011 为底（它有集成骨架、metadata 接线、带 `logprobs`/`sampling_weights` 的 `EnforcedToken`），补入 dump 独有的三样——`logprobs_to_weights`、`decimal_sample_from_logprobs`、设计文档。dump 那 244 个文件的其余部分（poc、benchmark 脚本等）与本次「拼到能验证」无关，**不带入**。

**⚠️ 重要更正（grill Q6，推翻了先前"v011 生产路径已可复现"的表述）**：v011 的 `topk_topp_sampler.deterministic_sample` 里，喂给 `sample_categorical_weights` 的整数权重是 **`(GPU_float32_probs × 2^16).round()`**（`probs` 是 torch.Tensor，源自 GPU 浮点 softmax，`.round()` 也非 decimal HALF_EVEN）——**不可复现**。这与 validator 侧 dump `logprobs_to_weights`（从 logprob 字符串走纯 decimal）算出的权重是**两套不兼容的语义**。后果：**只要 executor 仍用 v011 生产路径产 artifact，validator 的 decimal 重放就对不上，Check 2 零容差比对必然假拒。**

→ 所以"executor 生产路径改用 decimal 权重"**不是可选优化，是本方案成立的必要条件**（第一性原则：executor 与 validator 必须 bit 级同源）。本次**不做**（属生产热路径 + 需 GPU 端到端验证，违背"本次不碰 GPU"前提），但作为 **blocker 级头号遗留 §8 U11** 如实登记，见 [ADR 0003](docs/adr/0003-executor-path-must-adopt-decimal-weights.md)。

**移植性质（更正）**：把 dump 两个 decimal 函数 append 进 v011 的 `deterministic_utils.py` 本身是纯增量、零符号冲突（RNG 家族逐行同源）。但**整体方案不是"纯 append 就完整"**——validator 侧可复现，executor 侧仍不可复现，两端语义分叉未闭合（U11）。本次交付的是**可复现的 validator 侧 + 诚实标注的分叉**，不是端到端可复现的完整系统。

---

## 1. Check 2（重放）为什么纯 CPU —— 以及它的边界（grill Q3）

> **范围收窄（重要，勿再过度承诺）**：纯 CPU 能验证的**只是 Check 2 = sampling replay**——"给定这份 logprobs，抽样有没有被篡改"。它**默认 logprobs 本身是真的**。"logprobs 是不是这个模型真跑出来的"要靠 **Check 1 = logprob distance，那必须 GPU 重跑模型**（文档 §114）。
>
> 所以本次纯 CPU 冒烟测试证明的是**"Check 2 重放逻辑正确"**，**不是**"这套方案能在纯 CPU 上判定 executor 诚不诚实"——一个伪造者编一份**自洽的假 logprobs**，Check 2 会全绿放行，拦它的是 Check 1（GPU）。**Check 1 明确不在本次范围**，记入 §8 U8。

两阶段方案，GPU 在 executor 生成、以及 validator 的 Check 1 出现；Check 2 纯 CPU：

```
执行端（需 GPU）                          验证端（纯 CPU）
────────────                             ────────────
模型前向 → 每位置 logprobs (float32)  ─┐
用 seed 抽 token 序列                   ├─→ artifact 存:
                                        │   - 每位置 {token_id_str: logprob_str}
                                        │   - user_seed, prompt_token_ids
                                        │   - 抽出的 token 序列
                                        │   (载体 = EnforcedTokens，v011 已有)
                                        ▼
                            validator 只重跑这段（无 torch/GPU）:
                            1. logprobs_to_weights: Decimal(logprob_str) → 整数权重(和=65536)
                            2. Sha256CounterRNG.from_seed_string(f"{seed}|{prompt_token_ids}")
                            3. sample_categorical_weights → token
                               (1~3 = decimal_sample_from_logprobs)
                            4. 逐位比对 artifact 里的 token 序列 → pass / reject
```

GPU 只负责产出 logprob 字符串；验证判定全在 CPU 用 `decimal` 标准库跑。这就是为什么 decimal 路线可被任何 CPU bit 级复现，而浮点 softmax 不行。

---

## 2. 逐文件移植清单

### 2.1 `vllm/v1/sample/deterministic_utils.py` —— 增量合并（改）

**做法**：在 v011 版本基础上 **append** dump 独有的部分。逐项：

| 移植项 | 来源（dump 行号） | 落点（v011） | 冲突? |
|---|---|---|---|
| `from decimal import Decimal, localcontext, ROUND_HALF_EVEN` | dump import 区（**改 `getcontext`→`localcontext`**） | 加到 v011 import 区 | 无（v011 未 import decimal） |
| ~~模块级 `getcontext().prec=10` / `rounding=ROUND_HALF_EVEN`~~ | dump 30-34 | **不照搬**——见下「⚠️ decimal 上下文改写」 | **行为改写**（grill Q5 / ADR 0002） |
| `WEIGHT_SCALE = 2**16` | dump 37 | 加到常量区 | ⚠️ **须核对**：v011 若在别处已定义同名常量则复用，不重复定义 |
| `logprobs_to_weights(...)` | dump 143-234 | append 到文件末尾 | 无（v011 无此符号） |
| `decimal_sample_from_logprobs(...)` | dump 240-276 | append 到文件末尾 | 无 |

**保留 v011 独有、dump 没有的**：`WeightedPrefixSampler`(102)、`sample_categorical`(浮点, 139)、`sample_sequence`(167)。这些被 v011 的 `DeterministicSampler` 和测试依赖，**不能删**。

**⚠️ decimal 上下文改写（grill Q5，唯一对 dump 核心代码的行为改写）**：
dump 用**模块级** `getcontext().prec=10; rounding=ROUND_HALF_EVEN`——这会在 `import deterministic_utils` 时**改整个进程的默认 `Decimal` 上下文**，静默污染进程内其它 Decimal 用户（vLLM 是大进程），且反过来会被别处 `getcontext()` 覆盖，破坏 bit 级复现前提。**改法**：删掉模块级全局配置，在 `logprobs_to_weights` / `decimal_sample_from_logprobs` **内部**用 `with localcontext() as ctx: ctx.prec=10; ctx.rounding=ROUND_HALF_EVEN` 包住全部 Decimal 计算。语义等价（同 prec=10 / HALF_EVEN），但 bit 级复现由管线自证、不依赖"没人动过全局"、无 import 副作用。见 [ADR 0002](docs/adr/0002-localize-decimal-context.md)。

**核对项（写代码前必做）**：
- v011 的 `deterministic_utils.py` 里 `import` 区是否已有 `Optional`/`Dict`/`List`/`Sequence`（`logprobs_to_weights` 签名需要）。若缺则补。
- 确认 v011 里没有已存在的 `WEIGHT_SCALE` 或 decimal 上下文配置，避免重复/冲突。
- `logprobs_to_weights`/`decimal_sample_from_logprobs` 全部 Decimal 运算是否已在 `with localcontext()` 内（含 `.exp()`、`.to_integral_value()`、比较），漏在外面就用到默认精度、复现失效。

### 2.2 `vllm/validation_sampling.py` —— 新建（Check 2 重放入口）

dump 文档点名（§415）但两分支都没有的文件。**纯 CPU，绝不 import torch。**

> **决定（grill Q1，选项 A）**：严格照文档 §416 钉死的**签名、名字、粒度**实现，**不**发明新接口。见 [ADR 0001](docs/adr/0001-follow-documented-validation-sampling-contract.md)。
> - 函数名 = `verify_sampling_from_logprobs`（不是我先前草拟的 `replay_and_check`）。
> - 粒度 = **单 token 位置**（返回 `bool`），**不是**整条序列。序列级的逐位循环比对属于**调用方**（文档指定的 `serving_chat.py` 编排层）——本次不接 serving_chat（见下「本次范围」）。
> - **seed 参数 = 已拼好的 `seed_str`**。本函数**只管重放，不管 seed 怎么拼**（`f"{user_seed}|{prompt_token_ids}"` 的推导留在函数外，属 §5 遗留的加固点，不进这个纯函数）。

文档钉死的签名（§416）：

```python
# vllm/validation_sampling.py  —— 纯 Python，无 torch
from typing import Optional, Dict
from vllm.v1.sample.deterministic_utils import (
    Sha256CounterRNG,
    logprobs_to_weights,
    sample_categorical_weights,
)

def verify_sampling_from_logprobs(
    logprobs: Dict[str, float], # {token_id_str: float}，与 artifact / v011 代码一致（grill Q4）
    seed_str: str,              # 已拼好的 seed 字符串；本函数不推导
    temperature: str,
    top_p: Optional[str],
    top_k: Optional[int],
    min_p: Optional[str],
    reported_token: str,        # Executor 报告的、该位置的 token id 字符串
) -> bool:
    """Check 2 单位置重放：
    0. logprob_strs = {tid: repr(f) for tid, f in logprobs.items()}  # §4 唯一转换点
    1. weights = logprobs_to_weights(logprob_strs, temperature, top_p, top_k, min_p)
    2. rng = Sha256CounterRNG.from_seed_string(seed_str)
    3. replayed = sample_categorical_weights(<按 sorted tid 排的 weights>, rng)
    4. return replayed_token_id_str == reported_token   # 零容差
    """
    ...
```

> **入参类型定案（grill Q4）**：`logprobs` 收 **`Dict[str, float]`**，不是 `str`。依据：(1) **现有实践**——`vllm/validation.py:30` 是 `Dict[str, float]`，`from_content` 直接取 OpenAI 响应的 `x["logprob"]`（原生 float），文档 §291/§303 也明说 artifact 是 JSON float；(2) **第一性原则**——单一数据源 = OpenAI 原生 float，单一转换点 = validator 内**第一步** `Decimal(repr(f))`（文档 §303「Decimal(repr(float))」）；存 float + 双方同一 `repr` 重建，与"直接存 str"在 CPython 内等价，且不必在生产热路径提前字符串化。`logprobs_to_weights` 仍吃 `str`，转换收敛在本函数第 0 步。记入 §8 U9。

> 注：文档另有 `decimal_sample_from_logprobs`（dump 已实现，端到端"字符串→token"）可直接复用为步骤 1–3 的合体；`verify_sampling_from_logprobs` 就是它 + 一个 `== reported_token` 比较。实现时二选一即可，但**对外暴露的名字/签名必须是文档钉死的 `verify_sampling_from_logprobs`**。

**本次范围（明确排除 serving_chat 接线）**：文档说 `verify_sampling_from_logprobs` 由 `serving_chat.py` 编排调用。**本次拼装不接 `serving_chat.py`**——接了会把 torch/serving 依赖拖进验证路径，违背"纯 CPU 可验证"的核心诉求。冒烟测试**直调** `verify_sampling_from_logprobs`；序列级循环比对与 serving 编排留作遗留（§5 新增第 6 项）。

**依赖检查**：`verify_sampling_from_logprobs` 入参 `logprobs` = `Dict[str, float]`（与 `EnforcedToken.logprobs` 同类型，caller 不用先转）；函数内**第 0 步** `repr(f)` 转 str 再喂 `logprobs_to_weights`。转换规则见 §4（`Decimal(repr(f))`），类型定案见上（Q4 / §8 U9）。

### 2.3 `docs/DETERMINISTIC_SAMPLING_VALIDATION.md` —— 移植（新增到 v011）

直接从 dump 取整份（903 行）。移植后**加一段勘误**：文档 §498/§176 描述 seed = `{user_seed}|{prompt_token_ids}`，与本合并分支实现一致，但标注为已知弱点（§5）。

### 2.4 测试 —— repoint + 新增

| 文件 | 动作 |
|---|---|
| `vllm/v1/sample/test_deterministic_standalone.py`（v011 版，import `sample_categorical`+`WeightedPrefixSampler`，v011 有这俩） | 保留；**追加** decimal 管线用例：断言 `iter_u64("reference_seed_v1",5)[0] == 4286832458236889005`、`sum(logprobs_to_weights(...).values()) == 65536`、`decimal_sample_from_logprobs` 同 seed 两次同结果 |
| `vllm/v1/sample/test_deterministic_sampler.py`（v011 版，import `DeterministicSampler` 等，v011 有） | 保留，确认 import 通 |
| **新增** `tests/v1/validation/test_replay_smoke.py`（纯 CPU） | 造自洽位置 → `verify_sampling_from_logprobs` 断言 `True`；篡改 `reported_token` → 断言 `False`。**断言语义 = "Check 2 重放逻辑正确"**，不是"executor 诚实"（需 Check 1/GPU，§8 U8），**也不是"对真实 v011 artifact 能通过"**——用的是自洽 artifact（validator 侧自算权重），因为 executor 生产路径权重仍不可复现（§8 U11）。这是「纯 CPU 能验证 Check 2 逻辑」的落地 |

> 不采用 dump 版测试文件（它们 import dump 没有的 `sample_categorical`/`WeightedPrefixSampler`，本来就 import 失败）。以 v011 版为准 + 追加 decimal 用例。

**追加断言（grill Q5 / ADR 0002）**：`test_deterministic_standalone.py` 加一条「import 无全局副作用」用例——`from decimal import getcontext; before=getcontext().prec; import vllm.v1.sample.deterministic_utils; assert getcontext().prec == before`，锁死 `import` 不再改进程默认 decimal 精度。

---

## 3. 不带入的东西（明确排除）

- dump 的 `vllm/poc/**`、`scripts/{benchmark_decimal_vs_native,test_integer_weight_reproducibility,get_deterministic_inferences}.py`、`tests/poc/**` —— 与「拼到能验证」无关。
- dump 版 `vllm/validation.py`（比 v011 弱，缺 `logprobs`/`sampling_weights`）—— 用 v011 版。
- dump 版两个测试文件 —— 用 v011 版。

---

## 4. canonical float→string：文档已钉死为 `Decimal(repr(f))`

`EnforcedToken.logprobs` 存的是 `Dict[str, float]`（JSON float），而 decimal 管线吃的是字符串。float→str 的规则**不是悬案，文档已记录唯一约定**（grill Q2）：

> 文档 §260：Validator takes executor's reported logprobs (JSON floats, converted to Decimal via **`Decimal(repr(f))`**)

**本方案据此定案**：
- validator 侧 `verify_sampling_from_logprobs` 收到 float logprobs 时，就地用 **`Decimal(repr(f))`**（即 Python 最短往返表示）转换。**不**再写"自洽即可 / 规则待定"——Python 侧规则**已定**。
- 数据流以文档为准：artifact 里 logprobs 就是 JSON float（`Dict[str, float]`），转换点在 validator 内部、规则唯一（`repr(f)`）。CONTEXT.md 的 `canonical float→string` 词条已写成 `Decimal(repr(f))`，一致。
- 冒烟测试用自产自洽 artifact（同一进程走同一 `repr(f)` 路径），验证重放逻辑本身正确。

**⚠️ 已知矛盾，如实保留（不在本次解决，见 §5 第 2 项 + §8 登记表）**：review 报告（ISSUE-1199）专门警告 `repr(f)` 有跨语言隐患——Python `repr` 与 Go 的 Ryu 在最短浮点格式化上已知不一致（golang/go#17997）。**文档选了 `repr(f)`，review 报告说 `repr` 危险，二者矛盾。** 本次照文档走（Python↔Python 同为 CPython 时 `repr` 自洽），把矛盾原样记入遗留，**不**替 Executor 侧 / Go 侧擅自改约定。

---

## 5. 拼完仍遗留（本次不做，明确标注）

这些是 review 报告的 blocker，属于「拼起来」之后的独立工作，方案里**如实标注、不悄悄跳过**：

1. **seed 可碾**：`{user_seed}|{prompt_token_ids}` 由请求方可控输入拼成。加固需引入 executor 选不了的成分（如 `executor_addr`+`start_block_hash`），且要改协议——独立工单。
2. **跨语言 canonicalization**：Python 侧规则**已定** = `Decimal(repr(f))`（§4）。**遗留的是**：Go 侧能否 bit 级复现 CPython 的 `repr(f)`（review 报告指 golang/go#17997 已知不一致），以及是否需要一致性向量 + 可能改用更保守约定。**本次不动 Python 约定，只把这条矛盾登记待议。**
3. **Go 参考实现**：本仓库不含，跨语言可复现仍是断言。
4. **greedy 绕过 RNG**：`temperature < 1e-5` 走 argmax，Stage-1 无覆盖。
5. **worker 端到端**：本次只保证 offline 数学核 + validator 重放可跑；不验证 GPU 生成路径真被 worker 调用（那是 v011 骨架自身的既有状态）。
6. **serving_chat 编排 + 序列级比对**（grill Q1，选项 A 的取舍）：文档说 `verify_sampling_from_logprobs` 由 `serving_chat.py` 逐位循环调用来核验整条序列。本次**只交付单位置纯函数 + 直调它的纯 CPU 冒烟测试**，不接 `serving_chat.py`、不实现序列级编排——避免把 torch/serving 依赖拖进验证路径。见 [ADR 0001](docs/adr/0001-follow-documented-validation-sampling-contract.md)。

---

## 6. 落地步骤（review 通过后执行）

1. `git switch -c tg/deterministic_sampling_merged origin/tg/deterministic_sampling_v011`
2. 改 `deterministic_utils.py`：append decimal 上下文 + `WEIGHT_SCALE` + `logprobs_to_weights` + `decimal_sample_from_logprobs`（§2.1，先核对 import 与常量不重复）
3. 新建 `vllm/validation_sampling.py`（§2.2）
4. 移植 `docs/DETERMINISTIC_SAMPLING_VALIDATION.md` + 勘误段（§2.3）
5. 追加 decimal 用例到 `test_deterministic_standalone.py` + 新建 `test_replay_smoke.py`（§2.4）
6. **纯 CPU 验证**（无需 torch/GPU）：
   ```
   .venv/bin/python -m pytest vllm/v1/sample/test_deterministic_standalone.py \
                              tests/v1/validation/test_replay_smoke.py -v
   ```
   若环境无 torch，standalone 里触及 torch 的用例需 `pytest.importorskip("torch")` 跳过，decimal/replay 用例必须跑通。
7. `pre-commit run --files <改动文件>`（88 列行宽、ruff、mypy）

---

## 7. 验收标准（怎么算「拼成功了」）

- [ ] `deterministic_utils.py` 同时含 v011 的 `sample_categorical`/`WeightedPrefixSampler` **和** dump 的 `logprobs_to_weights`/`decimal_sample_from_logprobs`，import 无错。
- [ ] 参考值对上：`iter_u64("reference_seed_v1",5)[0] == 4286832458236889005`；`logprobs_to_weights` 权重和 `== 65536`。
- [ ] `validation_sampling.verify_sampling_from_logprobs`（文档钉死的单位置签名）对自洽位置返回 `True`、对篡改位置返回 `False`；冒烟测试直调它、不经 serving_chat。
- [ ] 上述验证**全程无 GPU/torch 依赖**（这是核心诉求）。
- [ ] `pre-commit` 通过。
- [ ] §5 遗留 + §8 登记表在文档里明确标注，未被伪装成已解决。

---

## 8. 未决 / 待讨论登记表（本次不解决，逐条保留）

> 贯穿性要求：**没改的、需要讨论的都要保留**。grill 至今浮现、但超出「拼起来 + 纯 CPU 可验证」范围的悬案集中记于此，避免散落遗忘。每条注明：现状、为何本次不动、将来怎么定。

| # | 悬案 | 现状 / 本次怎么处理 | 为何本次不解决 | 将来怎么定 |
|---|---|---|---|---|
| U1 | **seed 可碾** | 复现现有 `f"{user_seed}\|{prompt_token_ids}"`；`verify_sampling_from_logprobs` 只吃拼好的 `seed_str`，不碰推导 | 加固要引入 executor 选不了的成分（`executor_addr`+`start_block_hash`）并**改协议**，属跨仓库/跨端工作 | 独立工单：定 seed 组成 → 改 executor 与链上协议 → 更新文档 |
| U2 | **`repr(f)` 跨语言矛盾** | 照文档 `Decimal(repr(f))`；Python↔Python 自洽 | review 报告指 CPython `repr` 与 Go Ryu 已知不一致（golang/go#17997）；改约定等于替 Go 侧做未实测决策 | 补 Go 参考后跑一致性向量；若不一致，议是否改用 `f"{f:.17g}"` / hex 等更保守约定 |
| U3 | **无 Go 参考实现** | 本仓库不含；跨语言可复现目前是断言 | 超出本仓库范围 | 另起 Go 移植 + Python↔Go bit 级 conformance vectors |
| U4 | **greedy 绕过 RNG**（定位已核实修正） | review 报告说的 `temperature<1e-5→argmax` 绕过在 **`deterministic_sampler.py` 类里**（非生产路径）；生产路径 `topk_topp_sampler` 是 `if deterministic_rngs: → deterministic_sample`，无该前置分支。两处对 greedy 的处理不同，需在闭环时统一 | 是既有行为，非本次引入；且涉及 GPU 生产路径 | 议：生产路径 greedy（`deterministic_rngs` 存在但 temperature≈0）如何覆盖 Check 2；两处语义统一 |
| U5 | **worker 端到端未验证** | 只保证 offline 数学核 + validator 重放可跑 | GPU 生成路径真被 worker 调用属 v011 骨架自身状态，需 GPU 环境 | 有 GPU 环境后端到端跑一遍生成→artifact→重放 |
| U6 | **serving_chat 编排 + 序列级比对未接**（ADR 0001） | 只交付单位置纯函数 + 直调冒烟测试 | 接 serving_chat 会把 torch/serving 依赖拖进验证路径，违背纯 CPU 诉求 | 后续在 serving 层实现逐位循环调用 `verify_sampling_from_logprobs` |
| U7 | **`WEIGHT_SCALE` 常量是否已在 v011 别处定义** | §2.1 标注「写代码前须核对，避免重复定义」 | 需实际改代码时读 v011 文件才能定 | 落地步骤 2 执行时核对；若已存在则复用不重定义 |
| U8 | **Check 1（logprob distance）需 GPU、本次不做**（grill Q3） | 本次只交付 Check 2 纯 CPU 重放；冒烟测试语义 = "重放逻辑正确"，非"executor 诚实" | Check 1 要 validator 重跑模型（文档 §114 `GPU (validator re-runs model)`），本次前提是不碰 GPU | 有 GPU 环境后实现 `validation_distance.py`（文档 §421）+ 距离阈值编排；完整诚实性判定 = Check 1 ∧ Check 2 |
| U9 | **`verify_sampling_from_logprobs` 入参类型 = `Dict[str, float]`**（grill Q4，已定案，非遗留） | 收 float、内部第 0 步 `repr(f)` 转 str | — （已定案，登记备查：若将来 artifact 格式改存 str，此签名与转换点需同步调整） | 保持单一转换点；artifact 格式若变（U2 相关）则回看此处 |
| U10 | **decimal 上下文局部化**（grill Q5，已定案改写，非遗留） | 用 `localcontext()` 替 dump 的模块级 `getcontext()`；加 import-无副作用测试 | — （已定案，登记备查：这是唯一对 dump 核心代码的行为改写；ADR 0002） | 若将来引入 Go 参考，需确认 Go 侧 decimal 上下文也等价（prec=10 / HALF_EVEN） |
| **U11** 🔴 | **executor 生产路径权重不可复现（blocker，方案成立的必要条件）**（grill Q6） | v011 `topk_topp_sampler.deterministic_sample` 用 `(GPU_float_probs×2^16).round()`，与 validator 的 decimal 权重语义不同 | 改它属生产热路径 + 需 GPU 端到端验证，违背"本次不碰 GPU"；本次只交付可复现的 validator 侧 | **必做后续**：把该路径改走 dump decimal 管线算权重（从 logprobs 而非 GPU probs），使 executor↔validator bit 级同源；需 GPU 环境端到端验证。ADR 0003 |
| **U12** 🔴 | **抽样索引顺序分叉（blocker，独立于 U11）**（grill 细节1/2） | `sample_categorical_weights` 返回 `weight_list` 的**下标**，其含义取决于列表排列顺序。**dump validator 用字符串字典序 `sorted(tid)`；v011 生产路径用 vocab 整数下标序**——两者不一致（且 `"10"<"2"` 字符串序 vs `2<10` 整数序直接冲突）。即便 U11 对齐权重数值，顺序不统一 Check 2 仍假拒。dump 残余 tiebreak `max(...,key=lambda t:(w[t],t))` 的 `t` 也是字符串序，同根 | 与 U11 同属"两端同源"闭环，需改生产路径 + 定规范顺序 | **必做后续**：选定唯一规范顺序（第一性原则倾向 **vocab 整数序**，无歧义；字典序 `"10"<"2"` 反直觉且脆），让 executor 与 validator 建 `weight_list`、残余 tiebreak 都用它。与 U11 一起在 GPU 环境验证 |

> 新增悬案随 grill 继续追加到本表。落地实现时，凡触及某条悬案的代码处，注释里引用对应编号（如 `# see MERGE-PLAN §8 U2`）。
