# 项目进展

> 最近更新：2026-11-03
>
> 当前阶段：暂停扩大 GraphGPO baseline 复现，优先验证 BottleneckFlow 的核心 intuition。

## 近期工作

### 1. 项目与复现基础

- 官方 `verl-agent` 已作为 submodule 固定到 commit `234a913609ef820de826f2276b7b912e4b8b8410`。
- GraphGPO arXiv v2 论文已保存到 `references/papers/graphgpo/`。
- Qwen2.5-1.5B-Instruct 位于 `/cephfs/wenhaozhang/models/Qwen2.5-1.5B-Instruct`。
- ALFWorld 原始数据和 rollout 输入已经准备在 `/cephfs/wenhaozhang/datasets/agentic-credit-assignment/`。
- micromamba 环境 `agentic-credit` 已包含 GraphGPO 运行所需的 PyTorch、Ray、vLLM、Hydra 和 ALFWorld 依赖。

### 2. GraphGPO 理解与静态审查

- 已完成论文问题定义、方法链路、baseline、核心 claim、关键实验和假设整理。
- 已完成论文公式到官方代码入口、rollout、状态图、最短路、advantage 和 PPO update 的 mapping。
- 已记录当前可见的 paper–code 差异：
  - 不可达状态从 $+\infty$ 替换为 `far_distance + 1`；
  - episode advantage 默认按 step 重复纳入统计，实际受 trajectory length 加权；
  - graph return 的指数相对论文公式偏一位，并可能与固定 invalid-action penalty 发生尺度耦合。
- 已确认更关键的组件问题是 graph state identity 和 unit edge cost 是否足以表达不同环境中的真实任务结构。

详细记录见 [`reproduction/graphgpo.md`](reproduction/graphgpo.md)。

### 3. GraphGPO 工程 smoke

已在 2×A100 上使用 Qwen2.5-1.5B-Instruct + ALFWorld 完成一次轻量单步 update：

- 2 个 task groups，每组 8 条 rollout；
- 最多 5 个环境步；
- 完成真实环境 rollout、经验图构建、最短路、joint advantage、old/ref log-prob 和 FSDP actor update；
- step group 平均大小约 2.759；
- advantage 范围 `[-3.453, 1.512]`；
- gradient norm 22.831；
- 单步训练耗时约 62.4 s，不含初始化；
- 进程 exit code 为 0。

该 smoke 只验证工程链路，不支持 GraphGPO 的最终性能或核心机制结论。由于 max steps 仅为 5，本次 rollout success rate 为 0，不能用于评估正常主实验中的成功路径 credit。

### 4. 尚未完成的 baseline 工作

- GraphGPO 150-epoch ALFWorld 主实验尚未启动；预计 2×A100 需要约 18–30 小时。
- GRPO、GiGPO 和 GraphGPO 的 matched comparison 尚未运行。
- 多 seed、最终指标、训练曲线和机制消融尚未复现。
- checkpoint 恢复链路尚未验证。

当前决定不是否定 GraphGPO，而是暂缓投入长时间 baseline 训练，先判断新的关键节点与路径责任 intuition 是否具备基本可行性。

### 5. 新技术路线

已形成 BottleneckFlow 雏形：

$$
\text{Rollout Graph}
\rightarrow
\text{Bottleneck Importance}
\rightarrow
\text{Ancestor Path Responsibility}
\rightarrow
\text{Conservative Step Credit}.
$$

当前所有方法 claim 均未验证。完整技术路线、候选公式、风险、P0/P1/P2 假设与停止条件见 [`TECHNICAL_ROUTE.md`](TECHNICAL_ROUTE.md)。

## 即将进行的工作

优先做不依赖长时间在线训练的 P0 可行性验证：

1. 设计具有已知 bottleneck、替代路径和 ground-truth ancestor responsibility 的小型 DAG 环境；
2. 生成可控的成功/失败 rollout group；
3. 统计共享节点、成功对比、公共前缀和重复路径偏差；
4. 验证 bottleneck importance 能否恢复关键节点；
5. 验证 normalized path flow 能否恢复祖先 edge 排序；
6. 与 shortest distance、success visitation、raw frequency 和 empirical value 对比；
7. 只有离线归因成立后，再决定是否接入 GraphGPO/GRPO 在线训练框架。

尚未确定具体实验实现和日期；开始正式实验后，应在 `docs/experiments/` 建立独立记录。

## 当前阻塞与风险

- 目标 benchmark 中是否存在足够稳定的共享成功瓶颈尚无统计证据。
- 状态合并错误可能制造虚假汇合或拆散真实共享状态。
- 成功路径频次可能主要反映 behavior policy 和公共前缀，而非 action contribution。
- 每 task 仅 8 条 rollout 时，success contrast 和 path responsibility 可能方差过大。
- 复杂结构 attribution 即使更准确，也不一定转化为更好的 policy-gradient update。

## 最近行动

下一步先把 BottleneckFlow 的最小可控环境与离线归因验证定义清楚，不启动 GraphGPO 主实验，不继续优化 baseline 实现。

