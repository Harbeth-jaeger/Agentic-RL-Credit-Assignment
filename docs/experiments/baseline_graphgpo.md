# GraphGPO 评测与复现

## 基本信息

- 论文：*Beyond Trajectory-Level Attribution: Graph-Based Credit Assignment for Agentic Reinforcement Learning*
- 论文版本：arXiv:2605.26684v2
- 论文文件：`references/papers/graphgpo/arxiv-2605.26684v2.pdf`
- 代码仓库：`third_party/verl-agent`
- 固定 commit：`234a913609ef820de826f2276b7b912e4b8b8410`
- 目标：完成工程复现、结果复现和结论复现，并基于发现的问题提出修复或改进。

## 1. 理解 Paper

### 1.1 问题定义

#### 任务与输入输出

对任务实例 $x\sim p(X)$，策略 $\pi_\theta$ 与环境交互，产生同一 task group 下的 $M$ 条轨迹

$$
\tau_m=\{(s_t^m,a_t^m,s_{t+1}^m)\}_{t=1}^{T_m},\qquad m=1,\ldots,M,
$$

环境只在轨迹终局给出可验证结果 $R(\tau_m)$。目标是不训练额外 critic，而是利用同组 rollout 中重复访问的状态和跨轨迹连通关系，为每个 transition 构造比“把终局 advantage 广播给整条轨迹”更细粒度的 credit，最终得到用于 PPO-style policy update 的 token-level advantage。

GraphGPO 的输入是：同一任务的多条 rollout、每一步的环境状态与动作、terminal success/failure，以及边成本 $c(s,a)$；输出是每个已观察 transition 的 graph advantage $A^G(s,a,s')$，再与 episode advantage $A^E(\tau)$ 线性组合。

#### 论文中的方法链路

1. 将同一任务的 $M$ 条轨迹按“相同状态”合并为有向状态转移图 $\mathcal G=(\mathcal S,\mathcal E)$。
2. 以成功 terminal $s_{\rm succ}$ 为目标，在图上计算最小累计边成本：

   $$
   d(s)=\min_{s\leadsto s_{\rm succ}}\sum c(s_i,a_i),
   $$

   无成功路径时令 $d(s)=+\infty$。
3. 对边 $(s,a,s')$ 定义 graph return

   $$
   R^G(s,a,s')=r_{\rm succ}\,\omega^{d(s')+c(s,a)},\quad 0<\omega<1.
   $$
4. 将同一源状态 $s$ 出发的 transition 组成 $G^G(s)$，在组内标准化：

   $$
   A^G(s,a,s')=\frac{R^G(s,a,s')-\mu(G^G(s))}{\sigma(G^G(s))}.
   $$

   若该状态只有一个样本，则 graph advantage 为 0。
5. 与按 task group 标准化的 episode advantage 组合：

   $$
   A(s,a,s')=\beta^G A^G(s,a,s')+\beta^E A^E(\tau),
   $$

   论文实验取 $\beta^G=\beta^E=1$，再代入 clipped PPO/GRPO objective，并加入 KL penalty。

#### 关键假设

- 环境近似确定性；论文的单调性和条件方差命题均显式依赖 deterministic environment。
- 不同轨迹中的“同一状态”能够被可靠识别；否则图会碎片化或错误合并。
- 一个 task group 内至少观察到成功轨迹，才能从 success node 反向得到有信息的距离。
- 当前 rollout batch 的经验图足以近似与 credit 有关的可达性；未观察到的路径会被当作不存在。
- 目标是单一、可验证且 terminal success 可可靠识别。
- 图距离/边成本能代表任务进展；论文主实验统一使用单位成本 $c(s,a)=1$。
- 同一源状态下存在多个被观察到的 transition，组内相对 advantage 才非零。

### 1.2 关键 Baseline


| 方法              | Credit 粒度                  | Credit 来源                                                          | 与 GraphGPO 的关系                                                                                    |
| ----------------- | ---------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| GRPO              | trajectory/episode           | 同 task group 的终局 reward 均值与标准差；整条轨迹共享同一 advantage | 最直接的粗粒度基线；GraphGPO 保留其 critic-free group-relative 框架和 episode advantage               |
| GiGPO             | state-conditioned step group | 将相同状态的 step 分组，但 step return 仍是$\lambda^{T-t}R(\tau)$    | 最重要的直接 baseline；GraphGPO 保留 same-state grouping，将“轨迹终局折扣”替换为跨轨迹图距离 return |
| PPO               | token/step（critic）         | 学习 value function/GAE                                              | 用于比较无 critic 的 group-based 方法与经典 actor–critic 方法                                        |
| RLOO              | trajectory                   | leave-one-out group baseline                                         | 另一 critic-free trajectory-level 对照                                                                |
| ReAct / Reflexion | 无参数更新                   | prompting / verbal reflection                                        | 非 RL agent baseline；数值直接沿用 GiGPO，不是本文同设置重跑的核心对照                                |

GraphGPO 本质上不是新的 rollout 或 policy-loss 框架，而是 GiGPO/GRPO 训练管线中的 advantage estimator 替换：rollout、episode reward、PPO clipping 和 KL regularization 均保留，主要新增经验图、最短路和 graph-return normalization。

### 1.3 方法改动


| 组件                   | 是否修改   | 论文中的行为                                                      |
| ---------------------- | ---------- | ----------------------------------------------------------------- |
| Rollout group          | 保留       | 每个 task 采样$M=8$ 条轨迹                                        |
| Episode reward         | 保留       | 成功 10、失败 0；无效动作另罚$-0.1$                               |
| Episode advantage      | 保留       | 在 task group 内相对标准化，作为$A^E$                             |
| State grouping         | 沿用并扩展 | 与 GiGPO 一样比较同一状态的动作，但先把所有轨迹聚成全局状态转移图 |
| Step return            | 核心修改   | 从$\lambda^{T-t}R(\tau)$ 改为成功节点最短距离产生的 $R^G$         |
| Step advantage         | 核心修改   | 同源状态的 graph returns 组内标准化为$A^G$                        |
| Joint advantage        | 轻微扩展   | $\beta^G A^G+\beta^E A^E$；主实验两项权重均为 1                   |
| Policy objective       | 保留       | clipped importance-ratio objective + KL regularization            |
| 模型、prompt 与 memory | 控制变量   | Qwen2.5/Qwen2.5-VL；只保留最近两步交互历史                        |

### 1.4 核心 Claim


| ID | Claim                                                                                                        | 依赖假设                                                                                          | 可证伪条件                                                                                                                     |
| -- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| C1 | GraphGPO 能识别失败轨迹中的有价值步骤，credit 比 trajectory-outcome attribution 更忠实                       | 状态可正确合并；经验图含成功路径；距离与真实进展一致                                              | 在相同状态/action 的人工标注或反事实评估中，$A^G$ 与真实贡献无关，或打乱 graph return 后性能不降                               |
| C2 | 若一个动作以更低的$d(s')+c(s,a)$ 接近目标，则其 $A^G$ 严格更高（Proposition 4.1）                            | $0<\omega<1$；同源状态内标准化非退化；确定性转移                                                  | 构造满足前提的两条边却得到错误排序；或实现中的 unreachable/normalization 破坏排序                                              |
| C3 | 固定经验图后，graph feedback 对同一 transition 的条件方差不高于 trajectory-style feedback（Proposition 4.2） | 确定性环境；固定 rollout 图；success/failure 都以正概率经过该 transition；transition 与时间步独立 | 在满足前提的受控环境中，同一 transition 的 graph feedback 仍随机或经验方差更高；注意该命题本身不等价于偏差更小或 credit 更正确 |
| C4 | GraphGPO 比 GRPO/GiGPO 更高效地利用 rollout，在稀疏成功的训练早中期收敛更快                                  | 每批至少偶尔有成功轨迹；图覆盖足够；比较训练预算一致                                              | 按相同 environment interactions/tokens/算力比较时不更快，或优势只来自环境、reward、采样设置差异                                |
| C5 | GraphGPO 在 ALFWorld、WebShop、Sokoban 上提升最终性能                                                        | 官方训练和评测设置公平；多 seed 结果可靠                                                          | 严格复现下均值优势消失、方差覆盖差异，或只在某一特殊状态编码下成立                                                             |
| C6 | GraphGPO 额外计算开销可忽略                                                                                  | 图大小受$M,T$ 限制；状态 hash/比较低成本                                                          | 大 group/长 horizon/高维状态下图构造、canonicalization 或最短路成为显著瓶颈                                                    |

### 1.5 关键实验


| Claim | 支持实验                                                  | Benchmark                | 模型                   | 指标                              | 论文结果                                                                                                                    |
| ----- | --------------------------------------------------------- | ------------------------ | ---------------------- | --------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| C1/C5 | 主结果（3 seeds，多数结果）                               | ALFWorld                 | Qwen2.5-1.5B-Instruct  | overall success rate              | GRPO 77.86，GiGPO 90.88，GraphGPO 92.71                                                                                     |
| C1/C5 | 主结果                                                    | WebShop                  | Qwen2.5-1.5B-Instruct  | score / success rate              | GraphGPO 89.29 / 78.65；GiGPO 87.94 / 73.83；GRPO 84.73 / 71.35                                                             |
| C1/C5 | 主结果                                                    | ALFWorld                 | Qwen2.5-7B-Instruct    | overall success rate              | GRPO 83.33，GiGPO 94.27，GraphGPO 95.31                                                                                     |
| C1/C5 | 主结果                                                    | WebShop                  | Qwen2.5-7B-Instruct    | score / success rate              | GraphGPO 86.94 / 80.31；GiGPO 86.72 / 78.38；GRPO 84.31 / 75.00                                                             |
| C1/C5 | 主结果                                                    | Sokoban 6×6             | Qwen2.5-VL-3B-Instruct | success rate                      | GRPO 67.10，GiGPO 76.92，GraphGPO 86.98±0.73                                                                               |
| C4    | training curves                                           | 三个 benchmark           | 对应主模型             | train/validation success vs. step | 论文称 GraphGPO 在早中期领先；验证性能在 ALFWorld 比 GiGPO/GRPO 约早 30/110 steps，在 WebShop 约早 80/90 steps 达到相近水平 |
| C1    | 去掉 episode advantage                                    | 三个 benchmark           | 1.5B 或 3B-VL          | success rate                      | 仅 step advantage 时整体下降，但 GraphGPO 仍高于 GiGPO；Sokoban 83.07 vs. 62.50                                             |
| C4/C5 | dynamic sampling（全成功或全失败 group 最多重采样 10 次） | ALFWorld/WebShop/Sokoban | 1.5B 或 3B-VL          | success rate                      | GraphGPO+DS：ALFWorld 98.43，WebShop 85.68，Sokoban 90.36，均高于对应 GiGPO+DS（Sokoban 未报）                              |
| C5    | $\omega$ sensitivity                                      | ALFWorld/WebShop         | Qwen2.5-1.5B           | average success                   | 论文称敏感性较弱，建议$\omega\in[0.1,0.4]$；正文网页未完整呈现表 4/5 数值，复现时需从 PDF 图表人工核对                      |
| C6    | per-iteration runtime breakdown                           | 未明确（主设置）         | 未明确                 | wall-clock                        | graph construction 0.108 s，graph advantage 0.025 s；rollout 216.9 s，policy update 74.2 s；论文称新增开销约总时间 0.04%    |

### 1.6 初步疑问与假设


| ID | 问题或假设                                                                                                                            | 所需证据                                                                                                   | 当前状态                                                    |
| -- | ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| H1 | GraphGPO 的收益可能主要来自 goal proximity shaping，而非“动作带来的距离减少”                                                        | 对比 absolute$d(s')$、distance delta $d(s')-d(s)$、potential-based shaping；保持 return 分布与训练预算一致 | 官方代码已有`normalize_distance` 开关，但主实验关闭；未检验 |
| H2 | 无成功 rollout 时，图距离没有论文定义下的有效目标信号；实现的 finite fallback 可能制造伪排序                                          | 按 group success count 分层检查距离、advantage 和梯度；全失败合成图单测                                    | 静态代码已确认 finite fallback；影响未检验                  |
| H3 | 状态表示加入 observation、位置/持有物、物体移动史和 admissible actions，可能既减少 aliasing 又造成 graph fragmentation                | 统计唯一状态率、same-state group size、连通率；对不同 state key 做消融                                     | 状态字段已定位；未检验                                      |
| H4 | exact matching 对文本环境过严；SequenceMatcher 相似合并又可能错误合并语义不同状态                                                     | 人工抽样 cluster purity；threshold sweep；环境原生 state ID 对照                                           | 官方主设置 exact matching；未检验                           |
| H5 | 同源状态只出现一次时$A^G=0$，有效 graph supervision 的覆盖率可能远低于全部 step                                                       | 报告 step-group size 分布、非零$A^G$ 比例、按训练阶段变化                                                  | 未检验                                                      |
| H6 | Proposition 4.2 的“零条件方差”来自固定 batch graph 后 feedback 为确定函数，是较弱结论，不能证明 estimator 低偏差或跨 batch 方差更低 | 跨 rollout batch 重建图，测同一真实 transition 的 bias/variance；与 oracle distance 对比                   | 理论前提审查完成；实验未检验                                |
| H7 | ALFWorld/WebShop 的环境增强本身显著提高所有方法，论文主结果不能直接与原始 GiGPO 环境结果横比                                          | 用同一增强环境重跑全部方法；记录环境 patch                                                                 | 论文明确说明统一重跑；代码增强已定位                        |
| H8 | episode advantage 的代码按 step 加权统计，会让长轨迹在均值/方差中权重更高，偏离式 (2) 的 trajectory-level group normalization         | 合成不等长轨迹与手算对照；比较每轨迹一票实现                                                               | 静态代码已确认；未运行                                      |

## 2. 理解代码

### 2.1 Paper–Code 对应表


| 论文组件或公式              | 代码文件                                      | 函数或类                                             | 输入                                                                     | 输出                                              | 配置项                                                                      |
| --------------------------- | --------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------- | --------------------------------------------------------------------------- |
| Algorithm 1：训练入口       | `recipe/GraphGPO/main_graphgpo.py`            | `main` / `run_ppo` / `TaskRunner.run`                | Hydra config、dataset、worker/env classes                                | Ray workers、envs、`RayPPOTrainer.fit()`          | `recipe/GraphGPO/config/graphgpo_trainer.yaml`                              |
| 多轮 rollout                | `recipe/GraphGPO/rollout_loop.py`             | `TrajectoryCollector.vanilla_multi_turn_loop`        | task batch、policy worker、parallel envs                                 | 每个有效 step 的`DataProto`                       | `env.max_steps`, `env.rollout.n`                                            |
| task/trajectory identity    | `rollout_loop.py`                             | `vanilla_multi_turn_loop`                            | batch index、group size                                                  | 同 task 的`uid`；每条轨迹唯一 `traj_uid`          | `env.rollout.n=8`                                                           |
| 状态$s$                     | `recipe/GraphGPO/graphgpo_env_manager.py`     | 各环境 manager 的`reset` / `step`                    | 原始 observation、环境内部信息                                           | `anchor`（图节点）与 `text`（model prompt）       | `env.history_length=2`（继承配置）                                          |
| ALFWorld 状态增强           | `graphgpo_env_manager.py`, `alfworld_envs.py` | `AlfWorldEnvironmentManager.get_location_holding` 等 | observation、location、holding、item state/location、admissible commands | 拼接字符串`anchor`                                | 无单独开关                                                                  |
| 成功 terminal               | `rollout_loop.py`                             | `gather_rollout_data`                                | `episode_rewards>0`、最后 active step                                    | 成功轨迹末边的`next_obs='success'`                | reward 由环境给出                                                           |
| 状态 canonicalization       | `recipe/GraphGPO/core_graph.py`               | `to_hashable`, `StateCanonicalizer`                  | string/array state                                                       | hashable exact state 或首个相似代表               | `enable_similarity=False`, `similarity_thresh=0.95`                         |
| 聚合图$\mathcal G$（§4.2） | `core_graph.py`                               | `ComplexGraph.from_data`                             | `uid`, `anchor_obs`, `next_obs`, action, traj id                         | 每 task 一个`ComplexGraph` multi-edge graph       | unit`weight=1.0`                                                            |
| 目标距离$d(s)$（式 3）      | `core_graph.py`                               | `all_to_target_shortest_paths`, `calculate_path`     | reverse graph、`final_state='success'`                                   | `distances`, `paths`                              | 固定读取 edge`weight`                                                       |
| graph return（式 4）        | `core_graph.py`                               | `compute_graph_path_returns`                         | graph distances、每 step old/new state                                   | shape`[num_steps]` 的 `step_rewards`              | `algorithm.gamma` 实际对应论文 $\omega$；`normalize_distance`               |
| 同状态 group（式 5）        | `gigpo/core_gigpo.py`（被 GraphGPO 复用）     | `build_step_group`                                   | `anchor_obs`, task `uid`                                                 | 每个 step 的 group UUID                           | GraphGPO 调用未把 similarity 传入这里；图 canonicalizer 单独处理 similarity |
| graph advantage（式 6）     | `gigpo/core_gigpo.py`                         | `step_norm_reward`                                   | graph`step_rewards`、response mask、step-group UUID                      | 广播到该 action 所有 response token 的$A^G$       | `mode=mean_std_norm`                                                        |
| episode advantage（式 2）   | `gigpo/core_gigpo.py`                         | `episode_norm_reward`                                | token-level episode rewards、task/trajectory id                          | 广播到 response token 的$A^E$                     | 同一`mode`                                                                  |
| joint advantage（式 8）     | `core_graph.py`                               | `compute_graphgpo_outcome_advantage`                 | $A^G$, $A^E$                                                             | `scores` 同时作为 advantages/returns              | `step_advantage_w=1`, `episode_advantage_w=1`                               |
| estimator dispatch          | `recipe/GraphGPO/graphgpo_ray_trainer.py`     | `compute_advantage`                                  | `adv_estimator=graphgpo` 和 batch                                        | `batch.advantages`, `batch.returns`               | `algorithm.adv_estimator`                                                   |
| invalid-action penalty      | `graphgpo_ray_trainer.py`                     | `apply_invalid_action_penalty`                       | `is_action_valid`、token/step rewards                                    | 无效动作的 episode score 和 graph return 各减 0.1 | `use_invalid_action_penalty`, `invalid_action_penalty_coef=0.1`             |
| PPO policy update（式 9）   | `graphgpo_ray_trainer.py` + verl actor worker | `RayPPOTrainer.fit`, actor `update_actor`            | token advantages、old/ref log-probs                                      | policy update                                     | PPO clip/KL/mini-batch 等继承 verl 配置                                     |

### 2.2 数据与信息流

```text
task dataset
  -> gen_batch；每连续 8 个 env 共享 task uid，每个 env 有独立 traj_uid
  -> env.reset/step
       model 输入: text prompt（任务 + 最近 2 步历史 + 当前 observation/action candidates）
       graph 状态: anchor（不包含 prompt history，但包含环境选择的确定性字段）
  -> 每个有效 step 记录
       anchor_obs, text_actions, next_obs, rewards, is_action_valid,
       uid, traj_uid, active_masks
  -> trajectory 收尾
       episode_rewards/lengths 写回每个 step；成功末边 next_obs 改为 "success"
  -> ComplexGraph.from_data
       按 uid 建图；anchor_obs -> next_obs；边权固定 1；可选字符串相似合并
  -> reverse Dijkstra from "success"
       得到 batch 经验图上的 d(state)；不可达节点被代码改为 max_finite_d + 1
  -> compute_graph_path_returns
       主设置: step_reward = 10 * gamma ** d(next_state)
       可选: 10 * gamma ** (d(next)-d(current)+1)
  -> EpisodeRewardManager
       episode reward 放到每个 action response 的最后一个有效 token
  -> invalid action penalty
       token_level_score 与 graph step_reward 同时减 0.1
  -> same-anchor grouping + normalization
       graph step_reward -> A^G；episode token scores -> A^E
  -> A = 1.0 * A^G + 1.0 * A^E
       对每一步生成 action 的所有 response tokens 广播同一 advantage
  -> recompute old/ref log-probs -> clipped PPO actor loss + KL loss -> update
```

重要边界：图只在**当前 training iteration、当前 task 的 8 条 rollout**内构建，不跨 batch 持久化；validation 不构图训练。`uid` 是临时 UUID，不是数据集稳定 task ID。

各环境实际 `anchor`：

- ALFWorld：原始文本 observation + location/holding + 物体移动/状态信息 + 排序后的 admissible commands。它比论文“只增加当前位置与手持物”的简述更丰富。
- WebShop：格式化后的 text-rich observation，不含 prompt history。
- Sokoban：视觉设置直接以 RGB array 的扁平 tuple 为节点，精确像素相等才合并。

### 2.3 配置与超参数


| 参数                                |                                           论文值 |        代码默认值 |                                  运行脚本值 | 是否一致             | 备注                                                       |
| ----------------------------------- | -----------------------------------------------: | ----------------: | ------------------------------------------: | -------------------- | ---------------------------------------------------------- |
| group size$M$                       |                                                8 |    由上游配置决定 |                                           8 | 是                   | `env.rollout.n`，不是 `actor_rollout_ref.rollout.n`        |
| task groups / rollout               |                                               16 |                — |              train batch 16，因此共 128 env | 是                   | 16×8                                                      |
| training epochs                     |                                              150 |          上游默认 |                                         150 | 是                   | 所有脚本                                                   |
| history length                      |                                                2 |                 2 |                            脚本继承默认值 2 | 是                   | manager 从`config.env.history_length` 读取                 |
| success/failure reward              |                                           10 / 0 |       环境 reward |                              脚本未显式设置 | 静态上基本一致       | `EpisodeRewardManager` 使用累计 `episode_rewards`          |
| invalid action penalty              |                                             -0.1 |      actor config |                             0.1 coefficient | 是                   | 同时作用于 token score 和 graph step return                |
| $\beta^G,\beta^E$                   |                                             1, 1 |              1, 1 |                                        1, 1 | 是                   | `step_advantage_w`, `episode_advantage_w`                  |
| edge cost$c(s,a)$                   |                                                1 |      `weight=1.0` |                                 无 override | 是                   | tool/time cost 未实现为配置                                |
| distance discount$\omega$：ALFWorld |                                             0.10 |      `gamma=0.10` |                                        0.10 | 是                   | 代码沿用`gamma` 名称                                       |
| distance discount$\omega$：WebShop  |                                             0.20 |              0.10 |                                        0.20 | 是（脚本）           | YAML 默认只适合 ALFWorld                                   |
| distance discount$\omega$：Sokoban  |                                             0.80 |              0.10 |                                        0.80 | 是（脚本）           | 超出论文附录一般建议的 0.1–0.4，但与 Sokoban 专项设置一致 |
| normalization                       |                              论文公式为 mean/std |   `mean_std_norm` |                             `mean_std_norm` | 是                   | PyTorch sample std；epsilon=$10^{-6}$                      |
| state similarity                    |                                 未作为主方法描述 |    disabled, 0.95 |                                      未启用 | 主设置为 exact match | 属于代码额外功能                                           |
| distance delta                      | 论文式 (4) 用 absolute successor distance + cost |          disabled |                                    disabled | 部分一致             | 开启后是代码额外 variant                                   |
| actor LR                            |                                        $10^{-6}$ |          上游默认 |                                   $10^{-6}$ | 是                   | 全部 benchmark                                             |
| prompt/response length：ALFWorld    |                                       2048 / 512 |          上游默认 |                                  2048 / 512 | 是                   |                                                            |
| prompt/response length：WebShop     |                                       5120 / 512 |          上游默认 |                                  5120 / 512 | 是                   |                                                            |
| prompt/response length：Sokoban     |                                       1024 / 512 |          上游默认 |                                  1024 / 512 | 是                   |                                                            |
| max env steps                       |                                     50 / 15 / 15 |      env-specific |                                50 / 15 / 15 | 是                   | ALFWorld/WebShop/Sokoban                                   |
| rollout/validation temperature      |                                        1.0 / 0.4 |           1.0 / 0 | 脚本继承 rollout=1.0，并覆盖 validation=0.4 | 是                   | resolved config 仍应随实验保存                             |
| actor mini-batch                    |                 ALFWorld 256；WebShop/Sokoban 64 |          上游默认 |              1.5B ALFWorld 256；其余脚本 64 | 基本一致             | 7B ALFWorld 脚本为 64，论文附录未按模型细分                |
| KL coefficient                      |                                             0.01 | 上游 actor config |                                        0.01 | 是                   | `use_kl_in_reward=False`，但 actor KL loss 开启            |
| GPUs                                |                 1.5B/3B-VL: 2×H100；7B: 4×H100 |                — |                                       2 / 4 | 数量一致             | 本地为 A100，属于等价性而非完全相同硬件                    |

### 2.4 实现一致性检查

- [X]  论文中的关键步骤均能映射到代码。
- [X]  核心配置参数能静态追踪到对应计算；resolved Hydra 值仍需 smoke test 确认。
- [ ]  公式与实现完全一致：graph return 存在指数 off-by-one，episode normalization 的统计粒度也不同。
- [X]  terminal、failure 和 unreachable state 的代码处理已明确并记录。
- [X]  normalization 和 advantage mixing 的作用域已明确。
- [X]  已发现的过滤、fallback 和特殊处理已记录在下表；是否还有运行时差异需 smoke test。

### 2.5 差异与可疑点


| ID  | Paper 描述                                                              | Code 行为                                                                                                                                   | 影响                                                                                                                                        | 验证方案                                                                       |
| --- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| D1  | 式 (4)：$10\omega^{d(s')+c(s,a)}$，单位边成本时指数为 $d(s')+1$         | 主分支计算`10*gamma**(d(new)+weight-1)`，单位权重化简为 $10\gamma^{d(s')}$                                                                  | 指数差 1；在同状态 mean/std normalization 下只是统一正比例缩放，理论上 advantage 不变，但`mean_norm`、epsilon、invalid penalty 后会改变结果 | 合成图手算并分别测试`mean_std_norm` / `mean_norm` / invalid action             |
| D2  | 不可达状态$d(s)=+\infty$，其 graph return 应趋于 0                      | `all_to_target_shortest_paths` 将所有不可达状态设为 `far_distance+1`；若整图无 success path，所有节点被设为 1                               | 为经验上不可达的状态提供正 return；全失败 group 中可能所有 return 相等，也可能与 invalid penalty/分组交互产生信号；与论文语义不一致         | 全失败、部分可达、多个不可达分支三类合成图单测                                 |
| D3  | $A^E$ 按同 task 的 trajectory rewards 计算式 (2)                        | `episode_norm_reward(..., compute_mean_std_cross_steps=True)` 对 batch 中每个 step 都加入 score，且代码每次重复 append 两次；长轨迹权重更高 | episode baseline 被 trajectory length 加权，偏离“一条轨迹一个样本”；可能系统性偏置长/短轨迹                                               | 不等长轨迹合成例与标准 GRPO episode advantage 对比                             |
| D4  | state 是环境确定性状态；附录称 ALFWorld 只增加 location 与 held objects | ALFWorld`anchor` 还包含原 observation、item movement/state history 和 admissible commands                                                   | 状态可能更 Markov，但也更易碎片化；结果可能依赖未充分披露的状态工程                                                                         | 字段逐项消融并统计 group coverage/aliasing                                     |
| D5  | 同一 state 合并                                                         | 主设置精确 hash；可选`SequenceMatcher` 以首次遇到的代表贪心聚类，顺序相关且非传递                                                           | 小文本差异造成断图；相似模式可能误合并，且图 canonicalization 与 advantage grouping 可能不一致                                              | shuffle 输入顺序、cluster purity、阈值 sweep；检查相似模式下 group IDs         |
| D6  | 图 group$G^G(s)$ 是同源状态的候选 transition/edges                      | `build_step_group` 对每个 rollout step 分组，重复出现的同一 edge 以多次样本进入均值/标准差                                                  | 高频 transition 获得更大统计权重；并非对 unique graph edges 等权                                                                            | 构造重复边频次不均的 batch，与 unique-edge 版本对照                            |
| D7  | 失败 terminal 属于$S_{term}$，无成功路径距离为无穷                      | 失败没有专用`fail` 节点；超时/普通失败保留最后 observation，只有成功末边被改为 `'success'`                                                  | 不同失败终止可能与非终止同状态合并；terminal identity 没有显式进入节点                                                                      | 检查 done reason 与末状态碰撞；增加 terminal tag 的消融                        |
| D8  | graph reward 由距离与 edge cost 决定                                    | 无效动作 penalty 在图 return 计算后直接减 0.1，edge cost 本身仍为 1；`is_action_valid` 在 return 函数中读取但未使用                         | graph signal混入额外 heuristic，且惩罚尺度相对不同$\omega$/距离非常大                                                                       | 关闭 penalty、将 invalid 作为 edge cost、当前实现三者对比                      |
| D9  | 论文只提出 absolute-distance graph return                               | 代码提供`normalize_distance=True`：$10\gamma^{d(s')-d(s)+1}$，README 称其为 progress variant                                                | 这是潜在有意义但未在论文主结果验证的扩展；名称“normalize”不直观                                                                           | 作为独立 ablation，不与忠实复现混合                                            |
| D10 | Proposition 4.2 在固定图下论证条件方差降低                              | 代码每个 iteration 从 8 条新 rollout 重建经验图；跨 batch 的同一状态距离会变化                                                              | 定理不覆盖实际训练中最重要的 graph-estimation variance                                                                                      | 固定 policy 重复采 batch，测$d(s)$/$A^G$ 的跨 batch 方差                       |
| D11 | 论文称 graph overhead 由 hash table + Dijkstra 构成                     | exact hash 对数组会 flatten 成大 tuple；相似模式对每个新状态线性扫描全部 cluster 并运行`SequenceMatcher`                                    | 在图像或大文本、大 group、长 horizon 下未必保持论文复杂度/低开销                                                                            | 分环境 profile canonicalization、graph build、Dijkstra，并随$M,T$ 扩展         |
| D12 | 主实验公平比较 GRPO/GiGPO/GraphGPO                                      | 官方 repo 只有 GraphGPO recipe 的启动脚本，未提供论文三方法同一增强环境的统一复现实验清单或 seed orchestration                              | 容易因入口、环境 patch、batch/hardware 设置不同而产生不可比结果                                                                             | 从 GraphGPO trainer 统一切换 estimator，冻结其余 resolved config；保存 3 seeds |

#### 静态审查结论

官方 commit `234a913` 的 GraphGPO 是一个 self-contained recipe，关键链路完整且可读，但当前不能据此认定“代码忠实实现论文公式”。最优先的最小验证应覆盖 D1–D3：它们直接影响 graph return 和 joint advantage 数值；其次量化 D4–D6 的 graph coverage。仓库中未发现 GraphGPO 单元测试，现有 `compare_advantages.py` 和预生成可视化属于分析工具，不构成正确性测试。

## 3. 复现检验

### 3.1 实验原则

- 先小后大：先单元测试和最小 smoke test，再进行完整训练。
- 先 baseline 后新方法：依次验证 GRPO、GiGPO 和 GraphGPO。
- 固定模型、数据、rollout 数量、训练预算和评估协议。
- 保存代码 commit、环境快照、配置、随机种子、日志和 checkpoint。

### 3.2 工程复现

目标：代码能够稳定运行，数据流、graph、advantage、loss 和梯度行为正常。

- [X]  ALFWorld reset/step 正常。
- [X]  双卡 Ray、vLLM 和 FSDP 初始化正常。
- [X]  rollout 和 graph construction 正常。
- [X]  advantage 非全零、无 NaN/Inf。
- [X]  单个 update 能完成。
- [ ]  日志与 checkpoint 可恢复。

#### 3.2.1 单步 smoke 运行观察

- 有效运行环境为 micromamba `agentic-credit`（Python 3.10.22、PyTorch 2.6.0+cu124、vLLM 0.8.3、Ray 2.50.0），本地模型为 Qwen2.5-1.5B-Instruct，使用 2×A100。
- 轻量配置：2 个 task groups、每组 8 条轨迹、最多 5 个环境步、response 上限 128、只做 1 次 actor update；ALFWorld 实际加载 3553 个 train games 和 140 个 in-distribution eval games。
- 完整链路以 exit code 0 结束：rollout、经验图、最短路、joint advantage、old/ref log-prob、FSDP actor update 均执行。step group 平均大小 2.759；advantage 范围为 `[-3.453, 1.512]`；gradient norm 为 22.831；该步 wall time 62.4 s（不含初始化）。
- 本 batch 的 16 条轨迹全部失败（`episode/success_rate=0`、episode reward 全为 0），但 advantage 仍明显非零。结合 D2/D8，这是关键运行证据：全失败图没有论文定义的成功路径信号，非零训练信号来自代码的有限距离 fallback、无效动作惩罚及其归一化交互，不能解释为“基于到成功节点距离的 credit”。

### 3.3 结果复现

目标：在相同或等价实验条件下，复现论文报告的主要数值和训练趋势。


| 方法     | 配置   | Seeds | 论文结果 | 复现结果 | 差异 | 状态   |
| -------- | ------ | ----: | -------: | -------: | ---: | ------ |
| GRPO     | 待填写 |       |          |          |      | 未开始 |
| GiGPO    | 待填写 |       |          |          |      | 未开始 |
| GraphGPO | 待填写 |       |          |          |      | 未开始 |

### 3.4 结论复现

目标：验证结果是否真正支持 graph-based credit assignment 的核心 claim，而不只是最终指标提高。

待设计的机制对照包括：

- 去除 graph advantage，只保留 episode advantage。
- 打乱 graph return，但保持其数值分布。
- 比较 goal proximity 与 distance reduction。
- 改变 group size、成功轨迹比例和 graph coverage。
- 检查失败轨迹中的有效步骤是否获得合理 credit。

### 3.5 实验日志


| 日期       | 实验 ID                                             | 目的                          | 配置                                                                                         | 结果                                                                                                                         | 结论                                                                             | 产物路径                                              |
| ---------- | --------------------------------------------------- | ----------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------- |
| 2026-11-03 | `graphgpo-alfworld-env-smoke`                       | 验证 ALFWorld 环境            | 官方`AlfworldEnvs`，本地 `json_2.1.1`                                                        | 成功加载 3553 个 train games；`reset` 返回 42 个 admissible commands；执行 `go to cabinet 1` 后 observation/reward/done 正常 | 环境与数据层通过                                                                 | 无（终端 smoke）                                      |
| 2026-11-03 | `graphgpo-qwen1.5b-alfworld-update-smoke-preflight` | 验证启动命令                  | 系统`/usr/bin/python3`                                                                       | 缺少`hydra-core`，未进入工程                                                                                                 | 本地解释器选择错误；不属于官方实现问题，切换既有 micromamba 环境后解决           | 无                                                    |
| 2026-11-03 | `graphgpo-qwen1.5b-alfworld-update-smoke`           | 验证 GraphGPO 单次完整 update | Qwen2.5-1.5B；2×A100；train batch 2；每任务 8 rollouts；max steps 5；total training steps 1 | exit code 0；完整 update 完成；success rate 0，但 advantage`[-3.453, 1.512]`、grad norm 22.831                               | 工程链路可运行；同时为“全失败 batch 仍产生非零、非成功路径 credit”提供运行证据 | console/Ray 日志；未保存 checkpoint（`save_freq=-1`） |

## 4. 修复与优化

### 4.1 问题分类


| ID | 类型                   | 问题                                                                                    | 证据                                                                                                                                                        | 严重性 | 状态                                                   |
| -- | ---------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ------------------------------------------------------ |
| E1 | 本地环境（非官方实现） | 启动命令曾解析到缺依赖的`/usr/bin/python3`                                              | micromamba`agentic-credit` 中 hydra/torch/vLLM/Ray 与双卡均正常，完整 smoke 随后通过                                                                        | 已排除 | 已解决，不作为 GraphGPO 缺陷                           |
| I1 | 实现 / 论文一致性      | 全失败 rollout group 仍产生非零 joint advantage，无法归因于到成功节点的图距离           | smoke 中 success/reward 全为 0，但 advantage`[-3.453, 1.512]`；代码将 `inf` 改为 `far_distance+1`，并对 graph step return 直接施加 invalid-action penalty   | 高     | 已复现；需用关闭 penalty、保持全失败的受控实验拆分来源 |
| I2 | 实现 / 统计口径        | episode advantage 默认按 step 而不是按 trajectory 估计 group mean/std，长轨迹被重复计权 | `episode_norm_reward(..., compute_mean_std_cross_steps=True)`；`seen_pairs` 仅在该参数为 False 时更新，因此默认将每个 step 的相同 episode score重复加入统计 | 高     | 静态确认；需合成不等长轨迹单测及修正前后主实验         |

### 4.2 修复

记录使实现忠实于论文或修正明确错误的改动。修复前后的结果必须分别保存，不与方法创新混合。


| Fix ID | 对应问题 | 修改   | 修复前 | 修复后 | 是否解决 |
| ------ | -------- | ------ | ------ | ------ | -------- |
| 待填写 | 待填写   | 待填写 | 待填写 | 待填写 | 待验证   |

### 4.3 优化与 Idea

记录超出原论文实现的新方法假设。每个 idea 需要明确机制、预期现象和可证伪条件。


| Idea ID | 动机   | 方法   | 预期收益 | 可证伪条件 | 验证状态 |
| ------- | ------ | ------ | -------- | ---------- | -------- |
| 待填写  | 待填写 | 待填写 | 待填写   | 待填写     | 未验证   |

### 4.4 扩展验证

- [ ]  消融分析。
- [ ]  超参数敏感性分析。
- [ ]  多随机种子与方差分析。
- [ ]  失败 case 分析。
- [ ]  不同 benchmark 或分布外验证。
- [ ]  在相同训练 token、环境交互数和计算预算下比较。
