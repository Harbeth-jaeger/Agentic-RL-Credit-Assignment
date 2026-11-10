# Agentic Credit Assignment

本项目研究 agentic reinforcement learning 中的长程 credit assignment：已知一组 agent trajectory 的终局结果或 trajectory-level advantage，如何为轨迹内各个 step/turn 构造更可靠的局部 credit (A_{i,t})。

## 新 Session 从这里开始

进入服务器后，Codex 应先完整阅读：

1. 本文件；
2. 项目根目录的 `AGENTS.md`；
3. [`docs/TECHNICAL_ROUTE.md`](docs/TECHNICAL_ROUTE.md)：当前研究问题、暂定技术路线和证据状态；
4. [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md)：最近进展、当前阻塞与下一步；
5. [`docs/diagnostics/README.md`](docs/diagnostics/README.md)：诊断地图、优先级与停止条件。

只有处理具体 baseline、方法、诊断或实验时，再进入对应明细文档。GraphGPO 复现记录位于 [`docs/reproduction/graphgpo.md`](docs/reproduction/graphgpo.md)。

日常进入项目：

```bash
source /root/.bashrc
aca
```

`aca` 会启动或激活 mihomo 代理、激活 `agentic-credit` micromamba 环境，并进入本仓库。

建议先做低成本状态检查：

```bash
pwd
git status --short --branch
git submodule status
nvidia-smi -L
python --version
```

不要在未确认任务范围时启动模型/数据下载、完整训练、全量评测或大规模 sweep。

## K8S 与存储布局

这是 K8S 临时容器实例。实例可能关闭、删除或更换，不能把容器根文件系统视作唯一长期存储。


| 位置                  | 用途                                                | 生命周期与注意事项                                 |
| --------------------- | --------------------------------------------------- | -------------------------------------------------- |
| `/root`               | micromamba、Python 环境、VS Code Server、Codex 配置 | 本地小文件性能好；依赖实例存档`yanjunchi.tar` 恢复 |
| `/cephfs/wenhaozhang` | 代码、数据、模型、checkpoint、实验结果              | 独立持久化；重要研究资产必须放这里                 |
| `/localssd`           | pip/mamba/HF 临时缓存和可重建中间文件               | 高速但临时，不能保存唯一副本                       |

本项目关键路径：

```text
代码        /cephfs/wenhaozhang/projects/agentic-credit-assignment
环境        /root/micromamba/envs/agentic-credit
数据        /cephfs/wenhaozhang/datasets/agentic-credit-assignment
模型        /cephfs/wenhaozhang/models
实验输出    /cephfs/wenhaozhang/runs/agentic-credit-assignment
临时缓存    /localssd/micromamba-pkgs, /localssd/hf-cache, /localssd/pip-cache
```

仓库内链接：

```text
data    -> /cephfs/wenhaozhang/datasets/agentic-credit-assignment
models  -> /cephfs/wenhaozhang/models
outputs -> /cephfs/wenhaozhang/runs/agentic-credit-assignment
```

注意事项：

- Python 环境包含大量小文件，应保留在 `/root`，不要迁移到 CephFS；环境定义和 freeze 文件应保存在仓库。
- 模型、数据、checkpoint 和完整训练日志不提交 Git。
- 新实例首次进入后检查 CephFS 挂载、GPU、代理、micromamba 环境和 submodule。
- 大型下载、编译 CUDA 扩展或长时间训练前先确认存储、资源、耗时和恢复方案。
- 删除实例前更新 `yanjunchi.tar`；apt 系统环境由 `yanjunchi_env.tar` 备份。

## 项目结构

```text
agentic-credit-assignment/
├── AGENTS.md
├── README.md
├── configs/
├── docs/
│   ├── TECHNICAL_ROUTE.md      # 当前技术路线与证据状态
│   ├── PROJECT_STATUS.md       # 当前进展、阻塞与最近行动
│   ├── diagnostics/            # 根因诊断地图与独立诊断记录
│   ├── experiments/            # 正式实验记录
│   ├── decisions/              # 方向性设计决策
│   ├── algos/                  # 确定方法的实现设计
│   ├── method/                 # 方法材料
│   └── reproduction/graphgpo.md
├── references/papers/graphgpo/
├── requirements/
├── scripts/
├── src/
├── tests/
├── third_party/verl-agent/       # 固定 commit 的 Git submodule
├── data -> /cephfs/wenhaozhang/datasets/agentic-credit-assignment
├── models -> /cephfs/wenhaozhang/models
└── outputs -> /cephfs/wenhaozhang/runs/agentic-credit-assignment
```


## 当前研究方向

项目最初采用“复现并优化权威 baseline”的路线，以 GraphGPO 为主要对象理解 graph-based credit assignment。现已完成论文与代码静态映射，并跑通 Qwen2.5-1.5B-Instruct + ALFWorld 的双卡单步 smoke；完整主实验、matched baseline 和多 seed 结果尚未运行。

当前暂缓扩大 baseline 训练，优先按自己的 intuition 验证新的方法雏形 **BottleneckFlow：Success-Contrastive Bottleneck and Path-Flow Credit Assignment**。

核心问题是：

> 如何识别多条 rollout 中反复出现在成功路径上的关键状态，并根据历史 action 对到达这些状态承担的路径责任，将关键节点的信用分配给真正促成它们的历史步骤？

暂定主线为：

$$
\text{Rollout Graph}
\rightarrow
\text{Bottleneck Importance}
\rightarrow
\text{Ancestor Path Responsibility}
\rightarrow
\text{Conservative Step Credit}.
$$

该路线目前只是待验证 idea，不代表已得到实验支持。完整问题定义、公式候选、风险、验证顺序和停止条件见 [`docs/TECHNICAL_ROUTE.md`](docs/TECHNICAL_ROUTE.md)。

## Baseline：GraphGPO

- 论文：*Beyond Trajectory-Level Attribution: Graph-Based Credit Assignment for Agentic Reinforcement Learning*
- 版本：arXiv:2605.26684v2，ICML 2026
- 本地论文：[`references/papers/graphgpo/arxiv-2605.26684v2.pdf`](references/papers/graphgpo/arxiv-2605.26684v2.pdf)
- 官方代码：`third_party/verl-agent` Git submodule
- 固定 commit：`234a913609ef820de826f2276b7b912e4b8b8410`
- 计划对照：GRPO、GiGPO、GraphGPO
- 首选最小实验：Qwen2.5-1.5B-Instruct + ALFWorld

GraphGPO 将同一 task group 的 rollout 聚合为状态转移图，从成功节点反向计算图距离，据此构造 step-level return，再与 episode-level advantage 组合进行策略优化。本项目将重点核查状态合并、图距离、不可达状态、归一化和 advantage mixing 是否支持论文的核心 claim。

当前 baseline 状态：

- 论文理解与 paper–code mapping 已完成；
- ALFWorld 数据、rollout 输入和 Qwen2.5-1.5B-Instruct 已准备；
- 双卡单步 smoke 已完整通过 rollout、建图、advantage 和 actor update；
- 150-epoch 主实验、GRPO/GiGPO matched comparison 和多 seed 复现尚未进行；
- 现阶段不继续优化 baseline，保留它作为方法对照和后续在线训练框架。

最新状态见 [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md)，详细复现证据见 [`docs/reproduction/graphgpo.md`](docs/reproduction/graphgpo.md)。

## 下一步

下一 session 不应立即启动 GraphGPO 主实验或大规模下载。优先任务是：

1. 构造具有 ground-truth bottleneck、替代路径和 ancestor responsibility 的小型 DAG 环境；
2. 验证 success–failure contrast 能否识别关键节点；
3. 验证 normalized path flow 能否恢复祖先 action 排序；
4. 与 shortest distance、success visitation、empirical value 和 raw frequency 比较；
5. 只有离线 attribution 稳定后，再决定是否接入 GRPO/GraphGPO 在线训练。
