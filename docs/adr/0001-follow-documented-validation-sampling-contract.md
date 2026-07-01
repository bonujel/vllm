# 沿用文档钉死的 validation_sampling 契约，不发明新接口

合并 v011+dump 两条确定性采样分支时，新建的 `vllm/validation_sampling.py` 严格照 `docs/DETERMINISTIC_SAMPLING_VALIDATION.md`（§415-416）已记录的契约实现：函数名 `verify_sampling_from_logprobs`、**单 token 位置**粒度、返回 `bool`、直接吃已拼好的 `seed_str`。拼装草案里先前发明的 `replay_and_check(artifact, user_seed, prompt_token_ids, ...) -> ReplayResult`（整条序列粒度、自己推导 seed）被放弃。

## 为什么

- **文档是既有的领域决策**，Executor/Validator 双方要跑同一套契约才能 bit 级一致；validator 端擅自换名字/换粒度/换 seed 语义，会和文档描述、以及未来 Go 参考实现产生分歧。
- **单一职责**：`verify_sampling_from_logprobs` 只做"重放并比对单个 token"，不承担 seed 推导（`f"{user_seed}|{prompt_token_ids}"` 留在函数外，本就是 §5 的加固遗留点）。序列级逐位循环属于调用方。

## 取舍与范围

文档指定该函数由 `serving_chat.py` 编排逐位调用。**本次合并不接 `serving_chat.py`、不实现序列级编排**——那会把 torch/serving 依赖拖进验证路径，违背本次"纯 CPU 可验证"的核心诉求。本次只交付：单位置纯函数 + 直调它的纯 CPU 冒烟测试。serving 编排与序列级比对留作后续遗留（拼装方案 §5 第 6 项）。
