# Deterministic Sampling Validation

用于去中心化 LLM 网络的**可验证推理**：一个 Executor 节点跑推理，一个 Validator 节点独立核验其诚实性，而无需重跑整个推理。术语以 `docs/DETERMINISTIC_SAMPLING_VALIDATION.md`（dump 分支）为准。

## Language

**Executor**：
运行推理、采样出 token 序列、并产出 artifact 的节点。需要 GPU。
_Avoid_: producer, worker（worker 指 vLLM 内部的 GPU worker，是另一层概念）

**Validator**：
接收 artifact、独立核验 Executor 诚实性的节点。Check 2 纯 CPU；Check 1 需要 GPU 重跑模型。
_Avoid_: verifier, checker

**Artifact**：
Executor 交给 Validator 的核验载体。含 enforced token 序列、每位置的 post-penalty logprobs（`Dict[str, float]`）、seed、采样参数。载体类型是 `EnforcedTokens`（`vllm/validation.py`）。
_Avoid_: proof, payload, dump

**Check 1（logprob distance）**：
Validator 用**相同 prompt+penalties 重跑模型**，比对自己的 top-K logprobs 与 Executor 的距离。需要 GPU，有容差（tolerance）。
_Avoid_: distribution check, Stage 2, 分布验证

**Check 2（sampling replay）**：
Validator 从 Executor 报告的 logprobs（`Decimal(repr(f))` 转换）重跑 decimal 管线 → 整数权重 → SHA256 RNG 抽样 → 与报告 token 逐位精确比对。**纯 CPU、零容差**、在推理之前执行。
_Avoid_: sequence check, Stage 1, 序列验证

**decimal 管线**：
`logprob 字符串 → Decimal 温度缩放 → Decimal softmax → 过滤(top_k/top_p/min_p) → 量化为整数权重(2^16 scale)`。Executor 和 Validator 跑同一管线，保证任何 CPython 机器上 bit 级一致。核心实现 `logprobs_to_weights()`（dump 分支 `deterministic_utils.py`）。

**enforced token**：
Executor 采样出、被"强制"要求模型复现的 token。已由 `vllm/validation.py` 的 enforced-token 特性支持。

**canonical float→string**：
把 logprob 从 float 转成 decimal 输入的唯一约定规则。文档已钉死为 **`Decimal(repr(f))`**。
