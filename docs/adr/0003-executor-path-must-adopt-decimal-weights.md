# executor 生产路径必须改用 decimal 权重（blocker，本次不做）

---
Status: proposed
---

v011 的生产采样路径（`vllm/v1/sample/ops/topk_topp_sampler.py::deterministic_sample`）把整数权重算成 `(probs * 2^16).round().to(int64)`，其中 `probs` 是 GPU float32 softmax 的 `torch.Tensor`。validator 侧（本次合并引入的 dump `logprobs_to_weights`）则从 logprob 字符串走纯 decimal 管线算权重。**两套权重语义不兼容**：GPU 浮点跨机器会飘、`.round()` 非 decimal HALF_EVEN，与 decimal 结果不 bit 一致。

因此：只要 executor 仍用现生产路径产 artifact，validator 的 Check 2（零容差重放比对）对真实 artifact **必然假拒**。让 executor 生产路径改用 decimal 管线（从 logprobs 而非 GPU probs 算权重）**是本验证方案成立的必要条件，不是可选优化**——第一性原则：executor 与 validator 必须算出 bit 级相同的权重。

## 为什么本次不做

- 属**生产热路径**改动，且必须在 **GPU 环境端到端验证**（生成→artifact→重放），违背本次合并"不碰 GPU、纯 CPU 可验证"的前提。
- 硬塞会把一次范围清晰的移植变成半拉子的生产路径改造，且无法在纯 CPU 下验证——违背"小步、可验证、每步有测试兜底"的实践。

## 本次实际交付的边界（不自欺）

本次交付**可复现的 validator 侧**（decimal 管线 + `verify_sampling_from_logprobs` + 纯 CPU 冒烟测试）。冒烟测试用**自洽 artifact**（validator 侧自算权重），证明的是"Check 2 重放逻辑正确"，**不**证明"对真实 v011 executor artifact 能通过"——在本 ADR 落地前，对真实 artifact 会假拒。此边界在拼装方案 §0/§2.4/§8 U11 明确标注。

## 将来怎么定

把 `deterministic_sample` 里的 `(probs*2^16).round()` 替换为：从该位置的 post-penalty logprobs 字符串走 dump `logprobs_to_weights`，使 executor 与 validator 同源；再在 GPU 环境端到端验证 executor 产出的 artifact 能被 validator 的 Check 2 通过。
