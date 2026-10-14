# WAM + VLM 决策、记忆与单步运行时

本目录包含标准库决策内核、真实 WAM WebSocket 服务与技能调度原型。WAM 已训练并通过真实 validation 观测的动作检查；VLM CPU 接口已测试但不可靠。下面的 `core/runtime` 独立物理证据门仍只有合成驱动，尚未与仿真策略接通。具体实验结果见 `../progress.md`，不能据此声称 WAM 已替代 RL。

```bash
python3 scripts/check_harness.py
```

该检查使用明确标记的合成输入与假驱动，在 CPU 上验证门控、23D/数值/时效边界、单步执行、队列清空、询问/超时/预算、有限恢复、隐藏真值拒绝、失败与恢复证据、数据库隔离；不是机器人实验成绩，不加载模型或占用 GPU。

## 真实接口边界

已核对 `third_party/LiLa-WAM/robotwin_infer.py`：`_predict_chunk()` 返回反归一化 action chunk，`step()` 内部排队并返回单步 action；其 `instruction` 参数被忽略，原始 state parser 面向 RoboTwin 的 16D endpose。不能把 VLM 文本直接传给这个参数，就宣称实现了语言条件控制。BEHAVIOR 需要独立的 61D state/23D action adapter、真实条件桥接和新权重验证。

本内核在 **WAM chunk 输出之后、controller 执行之前**工作：

```text
外部语义候选 + WAM chunk + 新鲜物理证据
  → decide(...)
  → EXECUTE / ASK / OBSERVE / REPLAN / HOLD
执行后完整谓词证据
  → Memory.record(...)
  → 同作用域 Memory.recall(...)
```

`decide()` 必填参数：

| 参数 | 契约 |
|---|---|
| `candidates` | 尚未排除的全部语义解释，每种解释恰好一项；不能靠删除证据不足的候选来掩盖歧义 |
| `action_low`, `action_high` | 23 项控制器实际单位的界限；不假设 `[-1, 1]`，不允许以训练数据分位数替代物理控制边界 |
| `observation_id`, `now` | 当前观测 ID 和与候选时间戳共用的 monotonic 时钟 |
| `config` | 经开发集确定的 calibration ID、风险上界阈值、残差阈值、最大观测年龄、提交步数；提交默认 1，最多 4 |
| `ask_remaining` | 本 episode 还允许几次询问；设为 0 时语义分歧返回 HOLD |

`Candidate` 含 `interpretation`、每步 23D 的 `actions`、每步已落地的预期效果 `action_keys`、`observation_id`、`observed_at` 和 `Evidence`。输入使用 Python list/tuple 和有限的 Python int/float；模型 Tensor/NumPy 输出须在 adapter 中转为普通数值。

`Evidence` 含 `executable: True/False/None`、`checked_steps`、`risk_upper`、`prediction_residual`、`calibration_id`、`source`、`verifier_kind`。外部验证器必须证明这些证据确实覆盖指定前缀，并绑定当前观测、机器人、控制器和模型版本。**缺失或不匹配的校准不能当低风险。** 目前只校验字段和阈值，不产生风险校准，不把 WAM latent loss 冒充碰撞概率。`prediction_residual=None` 表示尚无轨迹与预测时间跨度都匹配的残差，此项不参与门控，但物理可执行性、风险与校准仍必须已知；禁止用 0 伪造未得到的残差。纯函数兼容不提供 `verifier_kind` 的旧调用；运行时强制要求 `perception_predicates` 或 `controller_postconditions`，不能用 `simulator_predicates` 给正式 policy 提供物理真值。

| 返回值 | 条件/责任 |
|---|---|
| `EXECUTE` | 所有语义分支的物理证据已知且合格；至少一步数值动作和效果 key 完全相同；只返回共同前缀，并附观测 ID 与到期时间 |
| `ASK` | 所有分支均有合格动作，但立即下一步的预期效果随用户意图改变；返回候选解释供 UI 构造问题 |
| `OBSERVE` | 观测过时/不匹配，或几何、动力学、风险校准未知；不向用户询问这些物理事实 |
| `REPLAN` | 已知不可执行、风险/残差超阈值；或者效果相同但采样动作不同，需要先求一个共同可验证动作 |
| `HOLD` | 无候选、非法输入或询问预算用尽；无动作输出 |

**所有非 EXECUTE 返回的 `actions` 都是空元组。HOLD 不是 23D 零向量。** R1Pro 的底盘速度与躯干/双臂绝对关节位置单位不同；安全等待命令必须由真实控制器结合当前关节与夹爪状态生成。外部执行器还必须检查到期时间，每个控制周期保留中断能力，清空旧 WAM action queue，并持续执行安全约束。仅限长为 1–4 的前缀不能替代控制器。

## 单步运行时

`harness.runtime.run_episode(adapter, ...)` 复用上述门控和记忆，不直接调用 RoboTwin 的 `step()`。真实接线必须绕开旧队列返回完整提议，再由本运行时验证和提交。无需继承基类；`adapter` 必须提供以下所有方法，无默认安全验证器或默认执行器：

| 方法 | 返回值与责任 |
|---|---|
| `observe(reason)` | `Observation(id, monotonic_time, sensors)`；每次采样的 ID 必须唯一，通道仅 `rgb/depth/proprioception`。物理信息未知时会带原因再次调用，适合刷新深度/视角观测；涉及机器人运动的观察动作必须先经过同样的物理验证，不能在回调里无条件移动 |
| `propose(observation, instruction, clarification, memories)` | 全部仍合理解释的 `Candidate` 列表；负责真实 VLM 条件桥接和 WAM 推理。`clarification` 是用户明确选中的当前解释；`memories` 只含同作用域、允许来源的记录 |
| `verify(observation, candidate)` | 独立物理验证器的 `Evidence`；运行时先清除模型自己声明的安全证据，禁止模型自我授权 |
| `execute(action, observation_id=..., expires_at=...)` | 在驱动侧**原子检查**观测 ID、到期时间和实时控制约束；接受并完成恰好一个控制步返回 `True`，未执行返回 `False`，不能将其继续排成旧动作队列 |
| `hold(reason)` | 根据实测关节/夹爪状态停止并维持安全姿态；不能简单发 23D 零向量 |
| `clear_actions()` | 清空模型与控制器所有未提交动作；不能清掉感知/本体状态历史 |
| `ask(choices, deadline=...)` | 截止前返回一个当前解释，超时返回 `None` 或抛 `TimeoutError`；真正 UI 和交互服务在此接入 |
| `outcome(before, action, after)` | `Outcome(predicates, terminal, source, verifier_kind, observation_id)`；完整技能后置条件必须来自最新的可见观测/控制器，结果绑定 `after` 的观测 ID |

运行时要求 `GateConfig.commit_steps=1`；每轮、提议后、询问前和退出时清空未提交动作。每次执行后先 hold，再读取新观测验证结果；下一步重新采样与提议。ASK 先停止并扣除预算，回答后必须重新观测，不复用回答前的动作。`OBSERVE/REPLAN` 有连续重试上限，另有 episode 总步数与总循环上限。默认值为 `max_steps=100 / max_cycles=200 / max_retries=3 / ask_budget=2 / ask_timeout_s=30`，校准与物理动作界限必须显式提供。

在线模式将已验证失败传给下一次提议；后续成功关联为 recovery。同作用域跨 episode 经验通过 SQLite 保留。Frozen 模式只查询、不写入；两种模式都返回 `RunResult`，包含状态、原因、步数、询问次数和带来源的事件记录。`SUCCESS` 仅表示当前 `Scope` 对应的完整后置条件已满足；若 Scope 是单个技能，不能把它算成整个 BEHAVIOR 任务成功，多技能调度仍由外层规划器负责。调用方应将事件保存到本次 run 的日志。这里实现了**经验读取→提议→验证→执行→结果写回**的调度；经验是否让真实模型表现改善，仍须实测，不能把账本本身当成学习效果。

同步回调必须实现自己的驱动超时；Python 运行时无法抢占一个卡死的推理或 UI 调用。它会拒绝超时才返回的回答/动作，异常退出会尝试 hold 和清队列，但真实机器人仍需独立控制器 watchdog。此版本在推理期间保持 hold，适合先做离散闭环验证；高频、连续全身控制需要后续测量时延后接入相应驱动，不能把当前合成检查当作实时控制认证。

## 记忆契约

`Memory(path, split_id=..., run_id=..., readonly=...)` 将一个 SQLite 文件固定到一个 split/run。不同 split 或 run 打开同一文件会被拒绝。`Scope` 的 task、skill、scene/context fingerprint、robot/controller/WAM/VLM 版本和 split/run 均精确匹配后才可查询。

`context_fingerprint()` 对调用者提供的 JSON 场景/任务上下文签名做 SHA-256。上下文应明确包含场景版本、对象类别/实例、相关空间关系及适用前提；此处不生成签名语义、不执行近邻检索，也不把跨场景泛化当作已实现。当前用途是**同 run、同上下文跨 episode 的证据复用**。

`record()` 只接受完整的外部谓词集合，值为 `True / False / None`，且 `terminal=True` 表示当前技能尝试已经结束（不要求整场环境 episode 结束）。

- 全部谓词为 True：记录 verified success。
- 至少一个 False 且尝试结束：记录 verified failure，包括失败谓词。
- 全部未知、部分 True/None、或尝试尚未结束：拒绝保存。
- recovery：需满足 verified success，并通过 `recovery_of` 引用同 episode、同作用域的先前 failure。

每条记录保存 episode、观测 ID、证据来源、验证器类别、记录时间及完整作用域。账本接受 `perception_predicates`、`controller_postconditions`，以及仅供独立调试的 `simulator_predicates`。`recall(..., policy_only=True)` 在 SQL 检索阶段过滤掉隐藏真值来源，单步运行时始终启用此选项，并拒绝把模拟器来源的结果写回；默认 `policy_only=False` 只适合独立审计。官方 policy 不得读取 BDDL/仿真器隐藏真值，后者只供外部评分；调试库不得晋升到正式 policy 记忆。不能用 VLM 猜测、临时观测或“看起来完成了”固化技能。来源白名单与字段检查不能识别伪装成传感器的真值；adapter 的真实数据来源、谓词集合完整性、校准工件仍须独立核对。

**Frozen**：使用独立冻结数据库文件和 `readonly=True`（SQLite `mode=ro`），禁止此连接写入。文件快照/哈希以及其他进程不得修改它，由评测 runner 管理。当前 v0 不实现 train/dev 证据晋升到官方 frozen prior；若需要，必须单独制定保留 `source_split`、`source_run` 和来源证据的白名单迁移，不能修改 split/run 标签后混用。

**Online**：每个评测 split/run 使用新的独立数据库。后一个 run 不继承前一个 run 的经验；测试经验不能写回训练/dev/冻结库。内核也不自动把成功轨迹变成可执行技能或更新 WAM 权重。

## 尚需接通并验证

真实 WAM adapter、BEHAVIOR 动作映射、日志和 CLI 回答已接通；短闭环执行 301 步、Q=0。可靠 VLM 程序生成、动作/动力学/几何验证器、风险校准、主动观察与恢复、完整任务及泛化评测仍待完成。当前不提供已验证的自主学习或安全保证。


## 2026-09-22 新增模块

- `policy_server.py`：官方 WebSocket 协议的 WAM 服务（见 `docs/benchmark_environment.md`）；`--program` 模式按技能程序执行并用 VTT 闭合推进。
- `vtt_monitor.py`：VTT 闭合（进度/对齐）三值谓词，阈值须在 dev 数据上拟合；导航类无效。
- `grounding.py`：指令 → 候选技能程序的严格契约（感知/目录校验、去重、动作层分歧判定）；后端可插拔（`MockBackend`、`QwenVLBackend`）。`python3 -m harness.grounding` 为合成自检。未训练的字面解释（如 `place in(digital,car)`）被目录拒绝并报告，不会执行。
- `memory_vtt.py`：把闭环中每个技能的**实现 VTT**（`realized_vtt.jsonl`）按证据规则晋升到 SQLite 账本，并导出同作用域已验证成功的均值向量作为条件（`policy_server --memory-vtt`）。`python3 -m harness.memory_vtt self-test`。

## 2026-09-22 调度边界修复

换技能前先处理意图分歧；关键技能失败后 HOLD；所有 grounding 拒绝分支都阻止静默选择；冻结 bank 不变；VTT 对齐不会自动写成 verified 成功记忆。`hold_baseline` 已修正 smooth gripper 的半开问题，并保留指关节范围参数。


## 2026-09-22 本轮接线与验证

- `Orchestrator` 首次分歧返回 ASK；等待回复时 HOLD。`answer(index, question_id=...)` 拒绝过期/非法/重复回复。`scripts/clarify_policy.py` 通过本地 WebSocket 查询及提交 episode+question ID，合法答复清空旧动作；下个观测重新提议。
- `memory_vtt promote` 必须有独立 `closure_labels.json`，纯 VTT 闭合不会晋升或更新 bandit 胜负。标签按 `index@end_step` 绑定具体尝试。当前实际可用的是人工复核晋升，在线感知自动验证尚未接通。
- `memory_vtt export` 输出 `verified-vtt-v1`，以 `skill(objects)` 为键；同一文件必须属于精确的任务/场景/机器人/控制器/模型/split/run 作用域。`--memory-vtt` 或 `bank_in` 只能在 orchestrator 下配合完整 `memory_context` 加载，旧裸 bank 快照被拒绝。
- 导出 `counts` 表示支持样本数；新均值条件臂的实测 successes/failures 均从 0 开始。不能将支持轨迹成功次数当作该均值条件的执行成绩，冻结选择仍可能保留 prototype。
- `scripts/check_agent_loop.py` 覆盖合成 ASK→回复→执行、复核→SQLite→冻结加载及拒绝边界；这是 CPU 合成检查。实际 Qwen3B 新版提示单例仍遗漏导航，结果见 `outputs/grounding/qwen3b_promptfix_single.json`。
