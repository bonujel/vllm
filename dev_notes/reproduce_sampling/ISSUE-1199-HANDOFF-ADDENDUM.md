# Issue #1199 调研补遗 (Addendum to handoff) — 基于实际 checkout 验证

> 本文是对 `issue-1199-vllm-handoff.md` 的**校正与推进**。所有结论均在 `bonujel/vllm`
> (= `gonka-ai/vllm` 镜像) 的真实分支上实测得到，非纸面推断。
> 验证日期: 2026-06-29。Python 3.10.12，纯 CPU，无 torch/GPU。

---

## 0. 一句话：最重要的新发现

原 handoff 只看了 `tg/detemrinistic_sampling_dump`(4月的 WIP dump)一个分支，
据此得出"集成全缺、测试坏掉"。实测发现**存在两条平行的架构路线**，handoff 漏掉了
更完整的那条：

| 分支 | 日期 | 核心采样 | 集成深度 | 跨语言可复现 |
|---|---|---|---|---|
| `tg/deterministic_sampling_v011` | 2025-12 ~ **2026-02** | **float**: GPU `logits.softmax(float32)` → numpy → `sample_categorical(probs)` | **较深**: 有 `deterministic_sampler.py`(708行 nn.Module)、`VLLM_DETERMINISTIC_SAMPLING` env、metadata 的 `deterministic_rngs`/`enforced_*` 字段 | ❌ 几乎必坏(见下) |
| `tg/detemrinistic_sampling_dump` | **2026-04** (WIP dump) | **decimal**: `logprobs_to_weights`(整数权重) → `sample_categorical_weights(int)` | **极浅**: 只有 standalone 数学核 + 903行设计文档 + 脚本；**无任何生成路径接线** | ✅ 这正是它存在的理由 |

**它们是同一个问题的两次不同下注，而且都没完成。** v011 接进了 vLLM 但用 GPU float
概率采样(跨机器不可复现)；dump 把概率换成 decimal 整数权重以求跨语言 bit 一致，
但还没接进采样器、且把 v011 的集成模块整个删了。

---

## 1. 实测复现的事实(全部通过)

在 `tg/detemrinistic_sampling_dump` 的 `deterministic_utils.py` 上：

- ✅ `iter_u64("reference_seed_v1", 5)[0] == 4286832458236889005`(与 handoff §7 一致)
- ✅ `decimal_sample_from_logprobs({...}, seed="42|[1,2,3]", T="1.0") == "791"`
- ✅ **F2 根因确认**：`sample_categorical` / `WeightedPrefixSampler` /
  `next_uniform01` 在 dump 分支的 `deterministic_utils.py` 里**全部不存在**，
  `deterministic_sampler.py` 模块也不存在 → 两个测试文件 collect 即 ImportError。
- ✅ **F3 画面级确认(关键)**：
  - `repr(-0.05)               == '-0.05'`
  - `repr(float32(-0.05)→f64)  == '-0.05000000074505806'`
  - 即 canonical 字符串完全取决于"从哪个 float 出发"。artifact **必须**存模型
    实际吐出的那个 float32 加宽到 float64 后的 repr，否则执行端/验证端串不上。

---

## 2. F2 的真正根因(对 handoff 的修正)

handoff 说"测试 stale / 坏掉、要修"。**更准确的说法**：
dump 分支的两个测试 import 的是 **v011 的 API**(`sample_categorical`、
`WeightedPrefixSampler`、`next_uniform01`、`deterministic_sampler.py`)。
v011 上这些符号**全部存在，测试本应通过**。dump 分支重写了核(`float→decimal`)、
却没更新测试、也没把集成模块搬过来 —— 所以才 ImportError。

含义：这不是"忘了维护测试"，而是**两条分支 API 不兼容、尚未 reconcile**。
Phase 1 修测试时必须先决定走哪条路线(见 §4)，否则修了也是修错对象。

---

## 3. 为什么 v011 的 float 路线跨机器几乎必坏(新증据，强化 F3/F7)

v011 `deterministic_sampler.py::deterministic_sample` 实测逻辑：

```python
probs = logits.softmax(dim=-1, dtype=torch.float32)   # GPU 上 float32 softmax
probs_cpu = probs.cpu().numpy()                        # float32
token_id = sample_categorical(probs_cpu[i], rng)       # 在 float 概率上做分类采样
```

- RNG(SHA256)确实可复现，但**被采样的概率**是 GPU float32 softmax 的结果。
- 验证端若在不同 GPU/驱动/库版本(甚至 CPU)上重算 softmax，float 末位会差 →
  累积分布边界处的采样落点会翻 → **误拒(false reject)**。
- v011 自己的 `deterministic_utils.py` 里 `sample_categorical_weights` 的 docstring
  就写着"cross-language reproducibility when probs may differ in float rounding" ——
  **作者本人已经意识到 float 路线不行**，整数权重函数是为此准备的后手。

dump 分支的 decimal 管线(`logprobs_to_weights`)正是把这个后手扶正：
softmax/top-k/top-p/min-p 全程 `Decimal` + 量化到 2^16 整数权重，
保证任意 CPython 3.3+(libmpdec)bit 一致。**这条路线对，但还没接线。**

---

## 4. 集成现状(对 handoff §3 / F1 的精确化)

逐项实测(`grep` 真实文件)：

| 接线点 | dump 分支 | v011 分支 |
|---|---|---|
| `VLLM_DETERMINISTIC_SAMPLING`(envs.py) | ❌ 无 | ✅ 有 |
| metadata `deterministic_rngs`/`enforced_*` | ❌ 无 | ✅ 有 |
| `DeterministicSampler` nn.Module | ❌ 无(测试 import 的模块不存在) | ✅ 有(708行) |
| **worker 里真正实例化/调用 DeterministicSampler** | ❌ 无 | ❌ **也无**(只在自身文件+测试里被引用) |
| 跨语言正确的 decimal 概率核 | ✅ 有(未接线) | ✅ 有 `sample_categorical_weights`(但 sampler 走的是 float `sample_categorical`) |

结论：**即便是更完整的 v011，DeterministicSampler 也没被 gpu_model_runner 真正调用** ——
metadata 数据结构铺好了，采样器没在生成循环里被触发。所以 F1("不能软上线产品")
依然成立，但 handoff 低估了已完成度：env flag + metadata + enforced-token 管线在 v011 已存在。

---

## 5. 据此更新的解题顺序(替换 handoff §10 的顺序)

1. **先做架构决断(阻塞一切)**：Stage 1 走 **decimal 整数权重**(dump 路线，跨语言可证)
   还是 float(v011 路线，跨机器不可证)。**强烈建议 decimal** —— v011 作者自己留的
   后手已经指向这里。这等价于 handoff 的 F-arch，但现在有了"v011 已试 float 并预埋
   整数权重函数"的实证支撑。
2. **把两分支 reconcile 成一条**：取 v011 的集成骨架(env/metadata/enforced/sampler 壳),
   把其中 `deterministic_sample` 的 `softmax(float32)+sample_categorical(float)` 换成
   dump 的 `logprobs_to_weights + sample_categorical_weights`。测试也随之指向 decimal API。
3. 然后才是 handoff 原 Phase 1：补 token/边界用例、reference vectors、跨语言(Go)对拍。
4. seed contract(F4)、float→string canonicalization(F3)两个 blocker 不变，仍须先于
   enforcing 落定。

---

## 6. 仍未解决 / 仍为 blocker(沿用 handoff，未被本次推翻)

- **F3** float→canonical-string 的跨语言 contract 未定义 —— 本次实测进一步坐实
  (repr 依赖具体 float64 bits)。架构建议:转换只放 vLLM 侧，Go 永不碰 float→string。
- **F4** seed 可被 executor 离线 grind —— 未变，仍需 protocol-forced seed。
- **F7** 无 Go 实现可对拍 —— 未变；两分支都只有 Python 侧，跨语言一致性零实测。
- greedy(F5)、5 条已知限制(F6)、tokenizer 确定性 —— 未变。

---

## 7. 给维护者的新增确认问题(补充 handoff §9)

5. **v011 和 dump 哪个是 canonical line?** 两者 API 不兼容、未 reconcile。
   是否应正式以 dump 的 decimal 核 + v011 的集成骨架合并为单一分支？
6. **DeterministicSampler 为何从未在 worker 实例化?** 是有意停在 metadata 层，
   还是集成 PR 尚未提交？这决定 Phase 3 还要补多少接线。
