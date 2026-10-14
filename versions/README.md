# WAM 各版本核心代码导航

本仓库现在以代码、环境依赖、版本差异与小型结果报告为主。2026-10-14 已按用户要求取消权重、数据集和大型观测档案上传，未完成的 Release 已删除。本地实验原件保留。

现行模块保留原目录以维持 Python 导入关系；下面按版本分类。冻结历史源码按原路径放在 `snapshots/<阶段>/`，不是把当前文件复制后冒称旧版。原始路径、SHA256 和分类说明见 [snapshot-index.json](snapshot-index.json)。

历史 `WAM V3`、五任务 `WAM V3+VTT`、`Precision V3` 是不同阶段；不能只凭文件名里的 V3 混为同一个模型。完整任务迁移和实测成绩见 [演进与任务转换](../history/EVOLUTION.md)。

| 阶段 | 核心代码 | 训练/评测入口 |
| --- | --- | --- |
| 01-upstream：原始LiLa-WAM | [LiLa-WAM](../third_party/LiLa-WAM) | 见上游仓库 README |
| 02-behavior-mobile：BEHAVIOR移动操作适配 | [wam_behavior.py](../harness/wam_behavior.py)<br>[wam_v3.py](../harness/wam_v3.py) | [train_wam_v3.py](../scripts/train_wam_v3.py)<br>[train_lilawam_behavior.py](../scripts/train_lilawam_behavior.py)<br>[train_wam_task_v3.py](../scripts/train_wam_task_v3.py) |
| 03-fixed-bimanual：固定双臂与柜体迁移 | [robotwin_policy_worker.py](../harness/robotwin_policy_worker.py)<br>[robotwin_dataset.py](../harness/robotwin_dataset.py) | [train_robotwin_wam.py](../scripts/train_robotwin_wam.py) |
| 04-dustbin-and-multitask：投瓶基线与五任务共享 | [robotwin_multitask_training.py](../harness/robotwin_multitask_training.py)<br>[robotwin_vtt_adapter.py](../harness/robotwin_vtt_adapter.py) | [train_robotwin_wam.py](../scripts/train_robotwin_wam.py)<br>[train_robotwin_multitask_wam.py](../scripts/train_robotwin_multitask_wam.py) |
| 05-hjepa-and-precision：珠海任务及H-JEPA/初代Precision | [wam_hjepa.py](../harness/wam_hjepa.py)<br>[wam_precision.py](../harness/wam_precision.py)<br>[zhuhai_hjepa_training.py](../harness/zhuhai_hjepa_training.py)<br>[zhuhai_precision_training.py](../harness/zhuhai_precision_training.py) | [train_zhuhai_wam.py](../scripts/train_zhuhai_wam.py)<br>[train_zhuhai_wam_hjepa.py](../scripts/train_zhuhai_wam_hjepa.py)<br>[train_zhuhai_wam_precision.py](../scripts/train_zhuhai_wam_precision.py) |
| 06-precision-v2-and-dagger：Precision V2及DAgger改版 | [wam_precision_v2.py](../harness/wam_precision_v2.py)<br>[zhuhai_precision_v2_training.py](../harness/zhuhai_precision_v2_training.py)<br>[zhuhai_dagger_training.py](../harness/zhuhai_dagger_training.py) | [train_zhuhai_wam_precision_v2.py](../scripts/train_zhuhai_wam_precision_v2.py) |
| 07-precision-v3：Precision V3失败支线 | [wam_precision_v3.py](../harness/wam_precision_v3.py)<br>[zhuhai_precision_v3_training.py](../harness/zhuhai_precision_v3_training.py) | [train_zhuhai_wam_precision_v3.py](../scripts/train_zhuhai_wam_precision_v3.py) |
| 08-precision-v4：Precision V4及Future16/32 | [wam_precision_v4.py](../harness/wam_precision_v4.py)<br>[zhuhai_precision_v4_training.py](../harness/zhuhai_precision_v4_training.py)<br>[zhuhai_precision_v4_future16.py](../harness/zhuhai_precision_v4_future16.py) | [train_zhuhai_wam_precision_v4.py](../scripts/train_zhuhai_wam_precision_v4.py)<br>[train_zhuhai_wam_precision_v4_future16.py](../scripts/train_zhuhai_wam_precision_v4_future16.py)<br>[eval_zhuhai_wam_precision_v4_future16.py](../scripts/eval_zhuhai_wam_precision_v4_future16.py) |

## 冻结历史源码

| 快照分类 | 文件数 |
| --- | ---: |
| [02-behavior-mobile](snapshots/02-behavior-mobile) | 225 |
| [03-fixed-bimanual](snapshots/03-fixed-bimanual) | 561 |
| [04-dustbin-and-multitask](snapshots/04-dustbin-and-multitask) | 20 |
| [05-hjepa-and-precision](snapshots/05-hjepa-and-precision) | 26 |
| [08-precision-v4](snapshots/08-precision-v4) | 14 |
| [09-shared-and-other-snapshots](snapshots/09-shared-and-other-snapshots) | 696 |

共 1,542 份源码/补丁快照。分类由原路径生成，仅用于查找；实际 run 与代码版本必须以原训练契约的 SHA 绑定为准。共享文件集中在 `09-shared-and-other-snapshots`。

这些源码快照保留原实验上下文，并非每个目录都能独立运行；运行时依赖根目录 `harness/`、`scripts/`、`configs/` 和第三方代码。部分旧版本权重已在历史清理中删除，不承诺每一步训练都可重跑。

## 结果与数据来源

- [阶段、任务与实测结果](../history/EVOLUTION.md)：同协议与跨任务结果分开。
- [17份原始证据索引](../history/evidence-index.json)：原报告字节与SHA不变。
- [268份run记录入口](../history/run-records-index.json)：仅列原工作区路径；大型原始记录未上传。
- [现存检查点目录索引](../history/retained-checkpoints.json)：仅元数据，不含权重。
- [数据来源与划分记录](../history/data-provenance)：仅小型来源/划分文件，不含数据集。

当前发布范围见 [publication-status.json](../history/publication-status.json)。
