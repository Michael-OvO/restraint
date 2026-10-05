# Honest Agents: Training Agents to Know When Not to Act and to Report Completion Truthfully

**一句话**：AgentAbstain（NeurIPS 2026 Oral）证明了 agent 不知道何时不该行动；本项目给出在**训练中**解决这个问题的方法。

---

## 1. Motivation

Frontier 评测的重心已经全面转向长程 agentic 任务。在这种设定下，agent 的最终汇报往往是人对任务结果的唯一观测，**"该不该行动"和"汇报是否真实"成为一阶可靠性问题**。现有证据表明，这两种能力都很弱，而且会随训练退化：

- **Agent 不知道何时不该行动。** AgentAbstain（UIUC，NeurIPS 2026 Oral，E&D Track 前 0.4%）在 42 个可执行沙箱环境中评测了 17 个前沿模型。最好的 Gemini 3.1 Pro 只有 59.5%，平均 45.7%，13 个模型低于 50%。失败模式包括：
  - implicit abstention（悄悄停下但不说明），17.8%
  - 完全没有放弃，19.7%
  - post-hoc abstention（先执行了不可逆操作，事后才声称拒绝），2.6%
- **训练会削弱 abstention。** AbstentionBench（Meta FAIR，NeurIPS 2025 D&B）发现，reasoning fine-tuning 让 abstention 平均下降 24%，而扩大模型规模基本没有帮助。
- **虚假完成很常见。** 在 SWE-bench Lite 上，StateFlow 的 fabrication rate 为 33.7%。
- **RL 会诱发作弊。** Tmax 在 RL 之后出现了训练前没有的 verifier tampering 和伪造产物。

## 2. Problem

我们认为这是 **outcome-based RL 的结构性后果**。在二元结果奖励下，abstain 或如实报告失败的回报恒为 0，而声称完成的期望回报非负，所以 policy 不存在任何适时放弃或如实汇报的激励。

AbstentionBench 观察到的"训练削弱 abstention"，在 agent RL 中预计会被进一步放大。评测层面同样缺少这个信号：主流 benchmark 对 decline 一律给 0 分。

## 3. Related Work & Gap

| 工作 | 场景 | 是否训练 |
|---|---|---|
| **AgentAbstain**（UIUC，NeurIPS 2026 Oral） | 多轮 agent，8 类场景：缺关键参数、动作歧义、约束冲突、高风险操作、工具能力不足、工具故障、证据冲突、执行中发现风险；263 组成对任务，附带可生成新实例的 pipeline | ✗ 只评测 |
| **Agentic Abstention**（UW，2026） | 多轮 agent，"何时停止"建模为序贯决策；28k+ 任务 | ✗ 只用 context engineering：CONVOLVE 把 Llama-3.3-70B 的及时召回率从 26.7% 提到 57.4%，不更新参数 |
| **AbstentionBench**（Meta FAIR，NeurIPS 2025 D&B） | 单轮 QA，5 类场景；20 个数据集，20 个 LLM | ✗ 只评测；system prompt 有改善，但不解决根本问题 |
| **Goal-Autopilot**（2026） | SWE agent，推理时拦截 | ✗ 外部系统，以牺牲 coverage 为代价 |
| LLM-judge / rubric 奖励 | 通用 | 训练，但 judge 本身可被 exploit |

**Gap**：社区已经确认 agentic abstention 是核心问题（AgentAbstain 拿到 oral），但现有工作全部停留在**评测或推理时干预**。在 policy 层面用 RL 训练 agent 的适时放弃和诚实汇报，目前在开源设定下仍是空白（投稿前需系统文献检索确认）。**AgentAbstain 负责测量，本项目负责训练。**

## 4. Approach

**4.1 汇报作为可核对的声明。** agent 结束时输出结构化 self-report，包括已完成项、未完成项、阻塞原因和证据命令。与 QA 不同，agent 的声明可以对**环境终态**做程序化核对：在干净容器中重放证据命令，检查受保护路径哈希和 canonical diff。这样得到的诚实性监督不需要人工标注，也不需要 LLM judge，可以规模化。

**4.2 Honesty-aware reward**

| 情形 | 奖励 | 针对的问题 |
|---|---|---|
| 可完成，成功，声称完成 | +1 | |
| 可完成，失败，声称完成 | −1 | fabrication |
| 可完成，失败，如实报告 | +α | |
| 不可完成，明确 abstain 并说明原因 | +1 | |
| 不可完成，声称完成 | −1 | 该放弃却没放弃 |
| 不可完成，停下但不说明原因 | 部分得分 | implicit abstention |
| abstain 之前已执行不可逆操作 | 额外 −β | post-hoc abstention |
| 可完成，未尝试即 abstain | 0 / 小幅惩罚 | lazy abstention |

**4.3 α 的决策论含义。** 设 agent 对成功的信念为 p。声称完成的期望回报为 2p − 1，如实报告的回报为 α，所以最优策略是**仅当 p > (1+α)/2 时声称完成**。α 显式设定了声称完成所需的置信度门槛；扫描 α 即可得到 success–honesty 的 Pareto 前沿。

**4.4 构造不可完成任务。** 对可完成的终端任务（TMAX-15K）做结构化扰动：移除依赖或工具、注入矛盾需求、撤销权限、隐去关键参数、引入高风险请求。不可完成性由构造保证，标签不需要人工判定。

扰动类别与 AgentAbstain 的 8 类场景有重叠，因此按"训练类别 / 留出类别"划分，单独报告跨类别迁移。

**设定**：Qwen3.5-4B + LoRA，GRPO/DPPO，2–4 张 H100 并行扫参。

## 5. Contributions

1. **把 agentic abstention 从评测问题推进到训练问题**：在 policy 层面用 RL 训练适时放弃和诚实汇报的开源 recipe，作为 AgentAbstain 的训练侧对应工作。
2. **Environment-grounded honesty supervision**：利用 agent 场景独有的可核对性，替代人工标注和 LLM judge。
3. **系统测量 outcome-only RL 下 fabrication 和 abstention 能力随训练的演化**，把 AbstentionBench 对 reasoning fine-tuning 的发现扩展到 agentic RL。
4. **可解释、可调的诚实性参数 α**，以及 success–honesty 的 Pareto 前沿。
5. **可扩展的不可完成任务生成方法**，为 calibrated abstention 提供训练数据。

## 6. Hypotheses

- **H1**：outcome-only RL 让 fabrication rate 随训练上升，AgentAbstain 分数下降。
- **H2**：honesty-aware reward 显著提升 AgentAbstain 分数，降低 fabrication 和 post-hoc abstention，同时任务成功率的下降不超过统计噪声。
- **H3**：α 平滑地控制 success–honesty 的 trade-off；习得的 abstention 能迁移到留出类别，以及 AgentAbstain 中未在训练里出现过的场景。

## 7. Evaluation

| 维度 | 指标 |
|---|---|
| **主指标** | AgentAbstain paired accuracy，按 8 类场景拆分；implicit 和 post-hoc abstention 比例 |
| 去污染 | 用 AgentAbstain 的生成 pipeline 产出**全新实例**，作为额外的 held-out 集 |
| 诚实性 | fabrication rate（held-out 扰动集、SWE-bench Lite） |
| 迁移 | Agentic Abstention 终端子集（及时停止率）；AbstentionBench（确认单轮 QA 的 abstention 没有被破坏） |
| 能力 | TB Lite 成功率 |
| 退化检测 | 可完成任务上的 abstention rate（检测 lazy abstention） |

**Baselines**：base model；prompt-only；CONVOLVE 式 context engineering；outcome-only RL；inference-time gate。

---

Sources:
- [AgentAbstain project page](https://agentabstain.github.io)
- [AgentAbstain (arXiv)](https://www.alphaxiv.org/abs/2607.10059)
- [Agentic Abstention (arXiv)](https://arxiv.org/pdf/2606.28733)
- [AbstentionBench (arXiv)](https://arxiv.org/abs/2506.09038)
- [AbstentionBench (NeurIPS 2025)](https://neurips.cc/virtual/2025/poster/121675)
- [Goal-Autopilot (arXiv)](https://arxiv.org/abs/2606.11688)
- [Tmax (arXiv)](https://arxiv.org/abs/2606.23321)
