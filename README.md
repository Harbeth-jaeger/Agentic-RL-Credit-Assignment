# Agentic Credit Assignment

本项目研究 agentic reinforcement learning 中的长程 credit assignment：已知一组 agent trajectory 的终局结果或 trajectory-level advantage，如何为轨迹内各个 step/turn 构造更可靠的局部 credit (A_{i,t})。

## 新 Session 从这里开始

进入服务器后，Codex 应先完整阅读：

1. 本文件；
2. 项目根目录的 `AGENTS.md`；
3. 当前任务涉及的文档。GraphGPO 复现的主记录是 [`docs/reproduction/graphgpo.md`](docs/reproduction/graphgpo.md)。

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

## 研究对象

### 核心问题

传统 group-based RL 往往把同一 trajectory 的最终 advantage 广播到所有 turn，难以区分不同动作对结果的贡献。本项目关注：

- 如何从多条 rollout 和终局反馈估计 step/turn-level advantage；
- credit estimator 是否真的识别了有贡献的动作，而不只是产生有利的 reward shaping；
- 方法对成功轨迹比例、状态定义、group size、图连通性及超参数是否稳健；
- 论文宣称的机制、公开代码的实际实现和最终性能之间是否一致。

### 当前进展

暂时没有收敛出一套 idea 雏形，准备通过”复现并优化权威 baseline“ 的思路，形成 idea。

当前阶段以 ICML 2026 工作 GraphGPO 为主要对象，采用“理解论文 → 理解代码 → 分层复现 → 修复或优化”的路线，目标不是机械运行作者脚本，而是验证实现、结果和核心机制，找出可修复的问题或可推广的新方法。

### 当前 Baseline：GraphGPO

- 论文：*Beyond Trajectory-Level Attribution: Graph-Based Credit Assignment for Agentic Reinforcement Learning*
- 版本：arXiv:2605.26684v2，ICML 2026
- 本地论文：[`references/papers/graphgpo/arxiv-2605.26684v2.pdf`](references/papers/graphgpo/arxiv-2605.26684v2.pdf)
- 官方代码：`third_party/verl-agent` Git submodule
- 固定 commit：`234a913609ef820de826f2276b7b912e4b8b8410`
- 计划对照：GRPO、GiGPO、GraphGPO
- 首选最小实验：Qwen2.5-1.5B-Instruct + ALFWorld

GraphGPO 将同一 task group 的 rollout 聚合为状态转移图，从成功节点反向计算图距离，据此构造 step-level return，再与 episode-level advantage 组合进行策略优化。本项目将重点核查状态合并、图距离、不可达状态、归一化和 advantage mixing 是否支持论文的核心 claim。

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
├── docs/reproduction/graphgpo.md
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

## 已知问题

截至 2026-11-03，已完成：

- 建立项目目录及外部数据、模型和运行输出链接；
- 将官方 `verl-agent` 作为 submodule 固定到 GraphGPO 合入 commit；
- 保存 GraphGPO arXiv v2 论文；
- 建立 GraphGPO 复现记录框架；
- 配置双卡 GPU 基础环境及 GraphGPO/ALFWorld 所需 Python 包；
- 安装用户级 Codex 调试、验证和 Hugging Face 技能，以及 Context7 MCP。

当前已验证的基础环境：


| 组件           | 当前值                               |
| -------------- | ------------------------------------ |
| GPU            | 2 × NVIDIA A100 80GB PCIe           |
| Driver         | 575.57.08                            |
| Python         | 3.10.22                              |
| PyTorch        | 2.6.0+cu124                          |
| vLLM           | 0.8.3                                |
| Ray            | 2.50.0                               |
| Transformers   | 4.51.1                               |
| TensorDict     | 0.8.3                                |
| FlashAttention | 2.7.4.post1                          |
| verl           | 0.3.1.dev（本地 editable submodule） |

尚未完成：

- 逐节精读论文并填写 claim、实验和可证伪假设；
- 完整 paper–code mapping；
- 下载 ALFWorld 数据和 Qwen 模型；
- ALFWorld reset/step、Ray/vLLM/FSDP 和训练 update smoke test；
- GRPO、GiGPO、GraphGPO 的结果及结论复现；
- 任何正式的新方法设计或训练实验。

### 已知环境问题

`python -m pip check` 当前报告：

```text
textworld 1.7.0 is not supported on this platform
```

已确认该包的 wheel 标记为通用 manylinux，且 `import textworld` 成功；但尚未验证 ALFWorld 实际 reset/step。因此不能宣称 ALFWorld 环境已经可运行，下一步使用它之前应继续定位并执行最小运行测试。

## 下一步

下一 session 不应立即下载数据或启动训练。优先任务是：

1. 细读 GraphGPO 论文；
2. 在 [`docs/reproduction/graphgpo.md`](docs/reproduction/graphgpo.md) 填写“理解 Paper”部分；
3. 提取核心 claim、关键实验、依赖假设和可证伪条件；
4. 再进入代码，建立 paper–code mapping 和最小复现计划。
