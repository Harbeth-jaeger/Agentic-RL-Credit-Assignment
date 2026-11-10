# BottleneckFlow-001 算法与实现设计

> 对应实验设计：[`../experiments/001_bottleneckflow_alfworld.md`](../experiments/001_bottleneckflow_alfworld.md)
>
> 001 固定采用“关键节点方案 B + 关键路径方案 C”。工程上不重写完整 RL 系统，而是复用 `verl-agent` 中 GraphGPO 的 ALFWorld、rollout、构图、veRL/FSDP/vLLM 训练链路，只替换 credit assignment 及必要接线。
>
> 当前探索设定为 ALFWorld 官方 `train` split 在线采样、`Qwen/Qwen2.5-1.5B-Instruct`、8×A100 80GB 和 veRL-agent/GraphGPO 训练链路。001 只跑 BottleneckFlow B+C，与论文报告的 GraphGPO 数值做初步对照，不声称已在当前环境复现 baseline。

## 1. Problem：问题、接口与代码位置

### 1.1 要解决什么问题

普通 GRPO 只知道一条 rollout 最后成功还是失败，因此同一条轨迹里的所有 action 基本共享同一个轨迹级 advantage。BottleneckFlow-001 要在不改变最终 reward 方向的前提下回答两个问题：

1. 哪些状态是真正重要的关键节点？
2. 为了到达这些节点，前面的哪些 segment 和 action 更应该承担 credit？

001 的答案固定为：

- **方案 B** 找节点：先用支持度和成功率置信下界 gate，再按成功覆盖、到达后成功率、早期性和剩余路径效率排序；
- **方案 C** 分路径：把轨迹按关键节点切成 milestone-to-milestone segment，先在不同来路之间守恒分配目标节点 credit，再在 segment 内分给 edge；
- 最终只重加权 GRPO turn advantage，不把失败轨迹中的 action 翻成正奖励。

反事实 oracle 不属于 001。先用真实 rollout 跑出完整结果，再根据实际问题设计诊断实验。

### 1.2 系统边界

```text
GraphGPO/veRL-agent 官方主链
  → ALFWorld + 官方 prompt/harness
  → vLLM 多轮 rollout
  → GraphGPO anchor state + ComplexGraph.from_data()
  → project-owned BottleneckFlow B+C
  → turn-level advantages
  → veRL/FSDP actor update
```

尽量保留官方的环境、prompt、history、state anchor、rollout、数据容器、log-prob、PPO clipped loss、optimizer、validation 和 checkpoint 逻辑。我们的核心代码只负责 B+C 节点与路径 credit；开发分支中的薄 patch 负责把 `DataProto + ComplexGraph` 交给项目模块，再把 turn advantage 送回官方 trainer。

001 不开启官方 invalid-action penalty。每步仍记录协议合法性、是否属于 admissible actions、环境是否返回 `Nothing happens`，但 invalid/no-op transition 也是图中的普通 edge，由 B+C 判断其价值，不人工扣分、不语义修复。

和 baseline 的不同点包括：无效 action 进入图，但不施加 penalty

按照 baseline 和技术方案不同的点包括：保留 GraphGPO 的 step-weighted episode advantage

### 1.3 输入接口

当前基础类型位于 `src/agentic_credit_assignment/credit/types.py`：

```python
Transition(
    state: str,
    action: str,
    next_state: str,
)

Trajectory(
    trajectory_id: str,
    transitions: tuple[Transition, ...],
    reward: float,
    success: bool,
    truncated: bool,
)
```

一次 `assign_credit` 的输入必须满足：

- 所有 trajectory 来自同一个 ALFWorld game/task 的独立 reset；
- `trajectory_id` 在 group 内唯一；
- transition 顺序与训练样本 turn 顺序完全一致；
- `success=True` 与环境最终 win 信号一致；
- 走满 horizon 且未成功按有效失败处理；environment error 单独标记并在构图前剔除；
- state、action 已做确定性的 canonicalization。

veRL `DataProto` 还需保留每个 turn 的 prompt、完整 admissible action set、response token 数、`uid`、`traj_uid`、`anchor_obs`、`next_obs`、`is_action_valid` 和 response mask。按五个相邻 baseline 的共同口径，`<think>...</think>` 与 `<action>...</action>` 的完整 assistant response 都参与 actor loss；prompt 和环境 observation 不参与 loss。

### 1.4 输出接口

对外仍返回项目自己的 `CreditResult`：

```python
CreditResult(
    advantages: tuple[tuple[float, ...], ...],
    diagnostics: dict[str, float | int | str],
)
```

必须保证：

- `advantages[i][t]` 与输入第 `i` 条轨迹第 `t` 个 transition 一一对应；
- shape、顺序和轨迹数不变；
- 每个值有限，不含 NaN/Inf；
- advantage 符号不因结构 credit 翻转；
- fallback 时输出与普通 GRPO 完全一致；
- diagnostics 足以还原 gate、排序、segment 分账、守恒误差和 fallback 原因。

### 1.5 代码位置

代码明确分为两类：我们自己的算法源码，以及 GraphGPO 官方仓库的自有开发分支。

#### 1.5.1 我们自己的 `src/`


| 位置                                                      | 作用                                                                                       |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `src/agentic_credit_assignment/credit/types.py`           | 项目内部的 trajectory/transition/credit 类型                                               |
| `src/agentic_credit_assignment/credit/bottleneck.py`      | 方案 B 关键节点、first-hit segment、方案 C edge credit                                     |
| `src/agentic_credit_assignment/credit/assigners.py`       | sign-preserving weighting、fallback 和统一输出                                             |
| `src/agentic_credit_assignment/graph/graphgpo_adapter.py` | 将`DataProto + ComplexGraph` 转为项目算法输入，再将 turn advantage 回填 veRL batch；待新建 |
| `tests/`                                                  | 手算图、守恒、符号、budget、invalid/no-op edge 与集成测试                                  |

核心方法不写进 `third_party`，保证其可单测、可复用、不被官方 trainer 绑死。旧 `src/.../slime/` 实现属于前一版路线，新链路不继续依赖；确认 veRL 版闭环后再删除或归档，不边改边混用。

#### 1.5.2 GraphGPO 开发分支

`third_party/verl-agent` 的官方基线锁定 `master@20bd331`，保持干净；从该 commit 创建 `aca/001-bottleneckflow` 开发分支。该分支是官方代码的可修改副本，只放框架必须的接线和口径修改：


| 官方文件                                             | 当前职责                                                                 | 001 处理                                                                                         |
| ---------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| `recipe/GraphGPO/run_qwen2.5_1.5b_alfworld_train.sh` | ALFWorld 1.5B full-run 入口                                              | 不直接改官方脚本；项目 wrapper 覆盖 8 卡、本地 placeholder、penalty、checkpoint 和输出路径       |
| `recipe/GraphGPO/main_graphgpo.py`                   | Hydra、Ray、model/tokenizer、worker/trainer 初始化                       | 复用                                                                                             |
| `recipe/GraphGPO/graphgpo_env_manager.py`            | ALFWorld manager、prompt/history、anchor state、validation               | 复用；增加可审计 action-valid 字段，评测仅`valid_seen`                                           |
| `recipe/GraphGPO/alfworld_envs.py`                   | Ray ALFWorld worker                                                      | 复用                                                                                             |
| `recipe/GraphGPO/rollout_loop.py`                    | 多轮 rollout 和`DataProto` 汇总                                          | 复用，保留完整 response loss mask 与逐 step trace                                                |
| `recipe/GraphGPO/core_graph.py`                      | `ComplexGraph.from_data()` 与 GraphGPO 最短路 credit                     | 复用构图；不调用`Calculate_all_shortest_path()` / `compute_graph_path_returns()` 生成训练 credit |
| `recipe/GraphGPO/graphgpo_ray_trainer.py`            | `fit()`、reward/advantage、log-prob、actor update、validation/checkpoint | 增加 BottleneckFlow estimator dispatch；关闭 invalid penalty；保留其他训练流程                   |
| `examples/data_preprocess/prepare.py`                | 官方借 Geometry3K 生成 carrier parquet                                   | 不下载 Geometry3K；项目本地生成最小 placeholder parquet                                          |

官方主链为：

```text
run_qwen2.5_1.5b_alfworld_train.sh
  → recipe.GraphGPO.main_graphgpo
  → TaskRunner.run()
  → make_envs() / AlfWorldEnvironmentManager / AlfworldWorker
  → TrajectoryCollector.multi_turn_loop()
  → vanilla_multi_turn_loop() / gather_rollout_data()
  → RayPPOTrainer.fit()
  → ComplexGraph.from_data()
  → credit / advantage estimator
  → old/ref log-prob
  → actor update
  → valid_seen / checkpoint / logging
```

001 的最小替换边界是：保留 `ComplexGraph.from_data()` 之前的环境与 rollout，保留 advantage 之后的 log-prob 与 actor update；将 GraphGPO 的最短路 return/advantage 替换为 BottleneckFlow B+C。

## 2. Algorithm：算法流程与模块分工

### 2.1 配置

新增 `BottleneckFlowConfig`，至少包含：

```text
min_success_support = 2
min_success_coverage = 0.5
beta_prior = (1.0, 1.0)
lcb_quantile = 0.05
top_k = 3
remaining_path_decay = 0.1
importance_weights = (w_N=0.5, w_E=1.0, w_Q=0.5)
dedup_success_jaccard = 0.9
dedup_max_graph_distance = 2
key_to_key_bonus = 1.0
segment_length_penalty = 0.2
failure_background = 0.5
local_advantage_weight = 0.5
flow_temperature = 0.5
credit_strength = 0.5
turn_weight_bounds = (0.5, 2.0)
min_transition_match = 0.95
```

每次运行把完整配置写入 manifest；禁止只在函数内部保留隐式默认值。

### 2.2 第 0 步：输入校验与普通 GRPO 底座

先检查同 task、唯一 trajectory id、非空 transition、outcome 一致性和合法长度。随后按 group reward 计算：

$$
A_i^{\mathrm{traj}}=
\frac{R_i-\mu_R}{\sigma_R+\epsilon}.
$$

这既是所有方法的共同底座，也是 B+C 无法形成可靠结构信号时的 fallback 输出。全成功或全失败 group 的方差为 0，输出零 advantage，不凭空制造学习信号。

001 为了只替换 credit assignment，episode advantage 先沿用 GraphGPO 当前实现：trajectory outcome 在组内统计时按该 trajectory 的 step 数重复参与均值/方差。这会使长轨迹权重更高，不是标准的“每条 rollout 一票”。本版本明确保留该口径；后续将 trajectory-level 修正作为单独改动/消融，不与 001 主方法混在一起。

### 2.3 第 1 步：构建 coverage graph

对同一 task 的 rollout group 构建：

$$
G=(V,E),\qquad e=(s,a,s').
$$

其中 $s$ 是 adapter 输出的 canonical state signature，而不是模型 prompt：

```text
normalized observation
+ location / holding / object state / object location（可获得时）
+ sorted admissible actions
```

节点身份直接复用 GraphGPO `anchor_obs` 与 `ComplexGraph.from_data()` 的构图结果，通过项目 adapter 转换为 B+C 需要的 coverage 统计。001 默认使用其 exact state merge；先用 pilot 记录节点数、merge ratio、连通分量、重复状态组和可视化图，再决定是否需要更强的 canonicalization。

需要同时保存：

- node/edge 的成功轨迹支持集合；
- node/edge 的明确失败轨迹支持集合；
- 每条轨迹中 node 的 first-hit step；
- edge 的所有 occurrence，以及用于 credit 的 first occurrence；
- 成功轨迹长度与到达节点后的剩余步数。

统计单位是 **trajectory coverage**，同一轨迹循环访问十次仍只贡献一次支持。原始 occurrence 只用于最后映射回 turn。

Invalid/no-op action 也保留为 `(s,a,s')` edge，包括常见的 `s'=s` self-loop。它不接受额外 `-0.1` penalty，也不被 harness 替换成其他动作；其 credit 由成功支持、失败背景和 local action signal 自然决定。为防止循环刷 credit，support 按 trajectory 去重，edge-to-turn 只给 first occurrence 结构 credit，但所有 occurrence 都保留在审计 trace 中。

### 2.4 第 2 步：方案 B 选择关键节点

对节点 $v$ 定义：

$$
N(v)=\frac{N^+(v)}{N^+},
$$

其中 $N^+(v)$ 是包含 $v$ 的成功轨迹数，$N^+$ 是成功轨迹总数。

到达后的成功概率使用 Beta posterior：

$$
p_v\mid\mathcal D\sim
\operatorname{Beta}(N^+(v)+\alpha,N^-(v)+\beta),
$$

并取其下分位数：

$$
L(v)=\operatorname{LCB}[S(v)]
=F^{-1}_{p_v}(q).
$$

早期性和剩余路径效率分别为：

$$
E(v)=\frac{1}{N^+(v)}
\sum_{i:Y_i=1,v\in\tau_i}
\left(1-\frac{T_i(v)}{L_i}\right),
$$

$$
Q(v)=\frac{1}{N^+(v)}
\sum_{i:Y_i=1,v\in\tau_i}
\exp[-\lambda_d(L_i-T_i(v))].
$$

候选先通过 gate：

1. 不是 start 或 terminal；
2. $N^+(v)\ge n_{\min}$；
3. $N(v)\ge\tau_N$；
4. $L(v)$ 高于使用同一 prior 平滑后的 task 基础成功率下界；
5. 没有被连续节点去冗余淘汰。

通过后按下式排序：

$$
I(v)=L(v)N(v)^{w_N}E(v)^{w_E}Q(v)^{w_Q}.
$$

取 top-$K$ 作为关键节点。这里先 gate 再排序，是为了不让一次幸运成功的 `1/1` 节点仅靠高成功率进入候选。

### 2.5 第 3 步：连续节点去冗余

若两个候选满足：

- 图距离不超过阈值；
- 成功支持集合 Jaccard 足够高；
- 失败支持集合近似；
- 中间没有真实 action 分叉；

则把它们视为同一阶段，只保留 $I(v)$ 更高者；相同分数用 canonical node key 决定，保证可复现。001 不使用 embedding 或 LLM 做状态合并，避免不可审计的误合并。

### 2.6 第 4 步：first-hit milestone segmentation

对每条轨迹按 first-hit 时间排列实际命中的关键节点，在 start、相邻关键节点和 terminal 之间切 segment。循环回来不再新增同一个 milestone。

例如：

```text
τ1: start → s10 → s8 → success
    [start,s10] | [s10,s8] | [s8,success]

τ2: start → s6 → s8 → success
    [start,s6]  | [s6,s8]  | [s8,success]

τ3: start → s6 → s7 → fail
    [start,s6]  | [s6,failure]
```

只对真实 rollout 出现过的 segment 计分，不枚举图上理论可达但没有证据的路径。

### 2.7 第 5 步：方案 C 在不同来路间分配目标节点 credit

对目标关键节点 $v$，每一种真实 predecessor milestone $u$ 的 route score 为：

$$
H(u,v)=\operatorname{Coverage}(u\to v)
\left(1+\kappa\sqrt{I(u)I(v)}\right)
\exp[-\xi\ell(u,v)].
$$

其中 $\ell(u,v)$ 是该类成功 segment 的平均长度；对 start 令 $I(start)=0$。然后：

$$
M(u,v)=I(v)\frac{H(u,v)}{\sum_{u'}H(u',v)},
\qquad \sum_uM(u,v)=I(v).
$$

这一步允许 `s10→s8` 比 `s6→s8` 获得更多份额，但不会因为 s8 有两条来路就凭空产生两份 $I(s8)$。

### 2.8 第 6 步：在 segment 内给 edge 分账

同状态局部 action advantage：

$$
A_{\mathrm{local}}(s,a)=
P(Y=1\mid s,a)-P(Y=1\mid s).
$$

对 segment $(u,v)$ 内的 edge：

$$
\rho(e;u,v)=q_{e,u,v}
\left[
P(e\in\tau[u:v]\mid Y=1,u\prec v)
-\eta P(e\in\tau\mid Y=0)
+\zeta A_{\mathrm{local}}(s,a)
\right]_+.
$$

$q_{e,u,v}$ 是支持度可靠性，可用平滑后的有效样本比例实现。再做 segment 内 softmax：

$$
W(e\mid u,v)=
\frac{\exp(\rho(e;u,v)/\tau)}
{\sum_{e'}\exp(\rho(e';u,v)/\tau)},
$$

$$
C(e\mid u,v)=M(u,v)W(e\mid u,v),
\qquad \sum_eC(e\mid u,v)=M(u,v).
$$

若所有 $\rho$ 都为 0，则在该 segment 的合格 edge 间均匀分配，而不是除零或丢失目标节点 credit。公共成功/失败动作会被失败背景压低，`s6→s8` 与 `s6→s7` 这样的局部分叉则由 local advantage 区分。

### 2.9 第 7 步：edge credit 映射回 turn

按完整 `(state, action, next_state)` key 回填到原始 transition。同一轨迹重复经过同一 edge 时，只给 first occurrence 结构 credit，避免循环套利。

计算 transition match coverage：

$$
r_{match}=\frac{\text{成功映射的 occurrence 数}}
{\text{应映射 occurrence 数}}.
$$

若 $r_{match}<0.95$，整个 task group 回退 GRPO，并记录缺失 key 示例；不能只对“碰巧匹配上”的 turn 加权。

### 2.10 第 8 步：生成 sign-preserving turn advantage

对每条轨迹的 turn credit 做标准化，生成原始权重：

$$
u_{i,t}=\exp\left[
\operatorname{sign}(A_i^{\mathrm{traj}})
\lambda_c z(C_{i,t})
\right].
$$

然后把 $u$ 投影到 $[w_{min},w_{max}]$，并按每个 turn 完整 response（think + action）的 token 数做预算归一化：

$$
\sum_t n_{i,t}w_{i,t}=\sum_t n_{i,t}.
$$

最终：

$$
A_{i,t}=A_i^{\mathrm{traj}}w_{i,t}.
$$

因此高 credit turn 在成功轨迹上获得更大的正更新，在失败轨迹上只减轻负更新；符号不会翻转。若轨迹 advantage 为 0 或 credit 无方差，令全部 $w=1$。

这也是 invalid/no-op step 的唯一学习通道：方法若有效，这些 edge 应因成功支持低、失败背景高或 local signal 差而得到较低的 $C_{i,t}$；在成功轨迹中少强化，在失败轨迹中多承担负更新。如果它们没有被压制，应视为算法结果，而不是再用手工 penalty 补救。

### 2.11 第 9 步：fallback 与 diagnostics

以下是允许的、可计数的数据不足 fallback：

- 成功轨迹少于 2；
- 没有节点通过 gate；
- 没有有效 segment 或 edge；
- transition match coverage 不足；
- 所有 trajectory reward 相同。

以下不是 fallback，而是必须抛错的问题：

- 输入 shape 或顺序不一致；
- outcome 字段矛盾；
- NaN/Inf；
- 守恒、符号、预算或权重边界 invariant 失败；
- 未知异常。

每组至少记录 gate 各阶段数量、节点四项得分、segment route score、两层守恒误差、权重分布、changed-turn 比例、fallback 原因和各阶段耗时。

### 2.12 文件级 I/O 分工


| 模块                                                    | 输入                                                       | 输出                                                              |
| ------------------------------------------------------- | ---------------------------------------------------------- | ----------------------------------------------------------------- |
| `recipe/GraphGPO/core_graph.py::ComplexGraph.from_data` | veRL`DataProto`                                            | 按 task 分组的 GraphGPO 经验图                                    |
| `src/.../graph/graphgpo_adapter.py`                     | `DataProto + ComplexGraph`                                 | 项目`Trajectory`、coverage/support、first-hit 与 occurrence map   |
| `src/.../credit/bottleneck.py`                          | graph view + B+C config                                    | key states、segments、route mass、edge/turn credit 和守恒诊断     |
| `src/.../credit/assigners.py`                           | trajectory advantage + turn credit + response token counts | bounded turn advantages 与`CreditResult`                          |
| `src/.../graph/graphgpo_adapter.py`                     | `CreditResult + DataProto`                                 | 对齐完整 response mask 的 veRL actor batch                        |
| `recipe/GraphGPO/graphgpo_ray_trainer.py`               | actor batch + Hydra config                                 | old/ref log-prob、actor update、validation、checkpoint 与 metrics |

## 3. Verification：如何判断算法正确

### 3.1 方案 B 单元测试

构造可手算 group，验证：

1. `N+(v)` 是包含节点的成功轨迹数，不是 occurrence 次数；
2. `7/8` 支持节点的 LCB 高于 `1/1` 幸运节点；
3. 其他量相同时，更早节点的 $E(v)$ 更高；
4. 到达后剩余路径更短的节点 $Q(v)$ 更高；
5. start、terminal、低支持节点被 gate；
6. 连续等价节点只保留一个，真实分叉节点不会误删；
7. 用完整 case fixture 得到预期顺序 `s10 > s8 > s6`。

### 3.2 方案 C 单元测试

验证：

1. first-hit segmentation 不会因循环重复切同一 milestone；
2. 在示例中 `M(s10,s8) > M(s6,s8)`；
3. 对每个目标节点，$\sum_uM(u,v)=I(v)$；
4. 对每个 segment，$\sum_eC(e\mid u,v)=M(u,v)$；
5. `s6→s8` 的 local signal 高于 `s6→s7`；
6. 全部 edge responsibility 为 0 时均匀分配且仍守恒；
7. 重复 edge 只给 first occurrence credit。

数值误差阈值先设为 `1e-6`，并在 CPU float64 的算法测试中执行。

### 3.3 性质测试

随机生成小图并检查：

- 调换 trajectory 输入顺序不改变按 id 对齐后的结果；
- 重命名 node id 不改变数值结果；
- 重复某个 occurrence 不增加 trajectory support；
- 所有权重有限且在边界内；
- turn advantage 与 trajectory advantage 同号或同为 0；
- token-weighted budget 保持；
- task group 之间没有统计泄漏；
- fallback 输出逐元素等于 GRPO。

### 3.4 固定 rollout 集成测试

保存一个很小的、可审计的 rollout fixture，从 `episodes_to_trajectories` 一直跑到 `CreditResult`：

- transition 对齐率必须为 1；
- diagnostics 中节点、segment 和 edge 与手算结果一致；
- 至少有一个成功组产生非 1 turn weight；
- 相同 seed 重跑结果一致；
- 全失败 fixture 明确 fallback，而不是伪造正 credit。

### 3.5 真实训练 smoke

在 8×A100 80GB 上先只跑 1 个 task、1 个 update：

- invalid action 会被检测、记录并作为 edge 进入图，绝不被语义修复，不触发人工 penalty；
- response mask 覆盖完整 think + action token，prompt token 不进 loss；
- mixed-outcome group 时 optimizer step 大于 0；
- loss、KL、gradient norm 有限；
- mixed-outcome smoke 中 actor 参数至少一个发生变化；`use_kl_in_reward=False`，actor KL loss 系数为 `0.01`；
- checkpoint 可从 CephFS 恢复并继续下一 update。

“命令退出码为 0”不等于算法正确。只有节点排序、两层守恒、符号/预算 invariant、非零梯度和恢复都通过，才允许进入主实验。

## 4. Risk：风险点与 fallback


| 风险                         | 表现                               | 处理方式                                                                                 |
| ---------------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------- |
| 基模成功率为 0               | 无 GRPO 方差，B+C 一直 fallback    | 先做 viability；若后续引入 warm-start，必须在 matched baseline 中使用同一初始 checkpoint |
| mixed group 太少             | 关键节点统计和 local signal 不稳定 | 增大 group size 或任务采样量；后续 matched baseline 使用同一预算                         |
| 小样本幸运节点               | `1/1` 节点分数虚高                 | 支持度 gate + Beta LCB，不直接用经验成功率                                               |
| 公共前缀冒充关键点           | start 附近节点覆盖高               | 排除 start、要求 LCB、早期与效率共同排序、结构去冗余                                     |
| observation key 太细         | 同一语义状态无法合并               | 先完善确定性 state signature；001 不直接上 embedding merge                               |
| observation key 太粗         | 不同状态误合并                     | 保留任务关键信息，检查 action set/环境事实冲突                                           |
| 循环重复刷 credit            | 同一 edge 被多次奖励               | coverage 统计 + first-hit segment + first occurrence 回填                                |
| segment 多导致总 credit 膨胀 | 来路越多总量越大                   | 对每个目标节点固定$I(v)$，两层分账都做守恒断言                                           |
| 长度偏置过强                 | 只奖励短路，忽略必要长操作         | 调低或关闭$\xi$，与无长度惩罚消融比较                                                    |
| local action 样本不足        | 分叉概率极不稳定                   | 平滑并加入可靠性$q$；不足时退化为成功/失败 coverage                                      |
| credit 过尖                  | 少数 turn 权重打满，KL/梯度抖动    | 调高$\tau$、降低 $\lambda_c$、收紧权重范围                                               |
| 结构 credit 改变 reward 语义 | 失败 action 被翻成正奖励           | sign-preserving + invariant；失败只允许减轻负更新                                        |
| invalid edge 未被压制        | no-op 与有效路径获得相近 credit    | 保留为主方法诊断；检查失败背景、local signal 和 self-loop 去重，不默认恢复手工 penalty   |
| 实现错位                     | graph edge 找不到训练 turn         | 完整 edge key、match coverage gate、失败时整组 fallback                                  |
| 静默 fallback 掩盖 bug       | 结果看似 GRPO                      | 仅允许枚举的数据不足原因；未知异常直接抛错                                               |
| K8s 重启丢数据               | `/localssd` 消失                   | 本地只放可重建热数据，checkpoint/manifest/metrics 落 CephFS                              |
| CephFS 小文件瓶颈            | rollout 写入拖慢训练               | 本地聚合为 JSONL/分片后批量写入，checkpoint staging 后复制                               |

相关性并不等于因果性，但 001 的目标是先验证这套可解释的经验归因是否带来训练收益，因此不加入反事实 oracle。若结果显示节点稳定但训练无收益，下一版再专门设计诊断。

## 5. Analysis：耗时、显存与优化

### 5.1 已有观测与尚未测量项

前一版 slime/Qwen3.5 链路的 smoke 不能作为当前 veRL-agent/Qwen2.5 方案的性能证据。当前已完成的是 GraphGPO 官方主链、ALFWorld 环境、构图与 advantage 接口的静态定位；B+C 接入后还必须重新跑 8×A100 的代表性 update，实测 rollout、graph/credit、log-prob、actor update、checkpoint 的耗时和峰值显存。

### 5.2 算法复杂度

设一个 group 有 $G$ 条轨迹、总 transition 数为 $T$、候选节点数为 $V_c$、真实 segment occurrence 数为 $J$：

- 构图和支持统计：$O(T)$；
- 节点评分：$O(V_c+T)$；
- 去冗余朴素实现最坏 $O(V_c^2)$，但 top 候选很少，可先按支持签名分桶；
- segmentation 与 edge 分账：$O(T+J)$；
- 映射和权重投影：$O(T)$。

相对模型 rollout、log-prob 和反向传播，CPU credit 计算应是小头；是否成立必须通过分阶段 wall-clock 指标验证。目标是 credit 计算低于单 update 时间的 5%，超过 10% 就优化。

### 5.3 首轮时间预算

首轮配置是每 update `4 tasks × 16 rollouts`、最多 50 environment steps、50 updates。总量为每个 RL 组 3200 episodes；实际耗时高度依赖平均轨迹长度、模型输出长度和成功率。

001 只运行 BottleneckFlow B+C 训练 arm。在没有代表性 profile 前不写死 ETA；先跑 1 个完整 update，实测 rollout、log-prob、actor update、credit 和 checkpoint，再按实际平均轨迹长度估算总时间。

这不是“我们的方案单独固定要这么久”，而是完整 rollout + RL 训练的粗估。B+C 自身新增的是 CPU 图统计；主成本仍是模型 rollout、log-prob 和 actor update。正式命令启动前，以 1 个完整 update 的实测结果重新计算。

### 5.4 显存压力

`Qwen/Qwen2.5-1.5B-Instruct` 是官方 instruction-tuned Dense checkpoint，不是 Base，也不是 MoE。主要显存来自：

- actor 参数、梯度与 optimizer state；
- vLLM rollout worker 的权重和 KV cache；
- 长轨迹 batch 的 forward activations；
- GRPO minibatch 反向传播；
- prompt 与候选 action 的 token 长度。

必须记录 `allocated`、`reserved` 和真实 peak，不能只看 `nvidia-smi`。若显存异常，先检查 batch/token padding、KV cache 与无意保留计算图，再考虑减小 minibatch、rollout concurrency 或开启 activation checkpointing。1.5B Dense 单卡可容纳，rollout 优先 profile `TP=1`，避免不必要的跨卡通信。

### 5.5 加速顺序

1. **vLLM continuous batching**：让 8 卡上的 rollout worker 并发处理 ALFWorld 请求，对 1.5B 模型优先测 `TP=1`，再与官方 `TP=2` 比较端到端吞吐；
2. **veRL/FSDP 八卡训练**：`trainer.n_gpus_per_node=8`，保持 batch/token 口径不变，通过 profile 调整 micro-batch 与 offload；
3. **FlashAttention/SDPA + BF16/TF32**：启动时记录实际 attention backend，不把安装成功或静默回退当成已启用加速；
4. **短历史与显式 action 协议**：只保留最近 2 步交互，控制上下文长度；
5. **全参数训练与 fused optimizer**：默认由 veRL/FSDP 训练 1.5B actor；只有 profile 证明必要时才切换 LoRA；
6. **CPU credit 优化**：canonical node/edge 转整数 id，集合统计改 bitset 或紧凑数组；
7. **三层存储**：HF cache、rollout 热数据和 tokenized batch 在 `/localssd`，正式指标/checkpoint 在 CephFS，最终结果再归档；
8. **减少 I/O**：metrics 追加 JSONL，rollout 分片批量落盘，不逐 turn 写小文件。

完整 runner 只执行 BottleneckFlow B+C 的训练、`valid_seen` 评测与审计产物，不自动运行其他 arm。开启周期 checkpoint，将 trainer、actor/optimizer 和 dataloader 进度保存到 CephFS，`resume_mode=auto`，并限制保留的历史 checkpoint 数量。

若 rollout 是瓶颈，优先增大批量、并行环境和缓存；若反向传播是瓶颈，再调 minibatch/epoch；若 credit 超过 10%，再优化图结构。所有加速改动必须对所有对照组一致，不能让主方法获得额外环境交互或 GPU 时间。

### 5.6 主跑前的性能准入

单 update profile 至少满足：

- 无 OOM 或 NaN；invalid/no-op 动作可被正常记录与构图；
- GPU 确实使用 BF16，attention backend 有记录；
- B+C 非 fallback group 中有非零 changed turns；
- credit 阶段耗时占比可接受；
- 两层 conservation error 小于 `1e-6`；
- checkpoint 可从 CephFS 加载；
- 每个 arm 的交互预算和实际 task/seed manifest 可审计。

达不到准入条件时先修链路或调统一配置，不启动多组长跑。
