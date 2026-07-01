# decimal 上下文局部化：用 localcontext() 替代模块级 getcontext()

合并 v011+dump 时，dump 的 `deterministic_utils.py` 在**模块级**设置全局 decimal 上下文（`getcontext().prec = 10; getcontext().rounding = ROUND_HALF_EVEN`）。移植时**改写**为在 decimal 管线函数（`logprobs_to_weights` / `decimal_sample_from_logprobs`）**内部**用 `with localcontext() as ctx: ctx.prec = 10; ctx.rounding = ROUND_HALF_EVEN` 包住计算。这是本次合并**唯一一处对 dump 核心代码的行为改写**（其余均为纯增量 append）。

## 为什么

- **消除 import 副作用**：模块级 `getcontext()` 会在 `import deterministic_utils` 时改**整个进程**的默认 `Decimal` 上下文。vLLM 是大进程，别处任何依赖默认 28 位精度的 `Decimal` 用户会被静默砍到 10 位——一个跨模块隐藏耦合。
- **复现前提自证**：反过来，若别的模块也调 `getcontext()`，谁后 import 谁赢，decimal 管线的 bit 级复现前提会被外部破坏。局部化后，prec=10/HALF_EVEN 由管线自身在每次计算时保证，不依赖"全局没被人动过"。
- **语义等价**：`localcontext()` 与模块级设置的算术语义完全相同（同 prec、同 rounding），bit 级结果不变。

## 取舍

偏离了"忠实照搬 dump"的基调，需要改写移植来的核心代码。接受此代价，换取无 import 副作用 + 复现前提不受外部干扰。用一条断言锁死：`import deterministic_utils` 后进程默认 `getcontext().prec` 不变（见拼装方案 §2.4）。

## 注意

管线内**所有** Decimal 运算（含 `.exp()`、`.to_integral_value()`、归一化除法、比较）必须落在 `with localcontext()` 块内；任何漏在块外的运算会用到默认精度，导致复现失效。
