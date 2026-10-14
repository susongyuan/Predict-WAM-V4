# Predict-WAM-V4

**完整研发轨迹：** [WAM 版本演进、任务转换、数据与结果索引](history/EVOLUTION.md)。历史数据与权重正在以分包 Release 发布；源码/结果索引和实际上传进度需区分。

固定双臂操作任务的 Precision WAM V4 研究代码。当前发布版本是 **Future +16/+32、Chunk-VTT16、exec8**，于 2026-10-14 从工作区整理；包含模型、数据处理、训练、推理、仿真评测、诊断及历史版本兼容源码。

本仓库是代码和环境依赖发布，不包含训练数据、DINO 权重、策略检查点、仿真资产或大型特征缓存。准备这些外部资源后才能训练及物理仿真评测。源码保留原实验的严格谱系和 SHA 校验；更换机器不是直接运行一条命令即可复现实验。

## 模型与范围

V4 从原 Precision V2 80k 检查点初始化，在已有结构上增加因果 Chunk-VTT16 预测与门控融合：

- 当前三视角冻结 DINO 特征、因果历史、实测机器人状态、TRAIN-only Task-VTT 构成输入。
- Chunk 预测器为 7184 → 512 → 512 → 3072，输出三视角未来 +16 的特征差分；3072 → 768 投影经可学习 alpha 融合。
- Future 辅助监督使用 +16/+32，权重各 0.25；IDM4、full32 精修、10 Euler 步、每次执行前缀 8 保持原协议。未来观测只作训练标签。
- 兼容键名 `hjepa_aux.future8_offset` 保留，当前语义为 +16，不能据键名误读成 +8。
- 新增训练使用 160 条原版 TRAIN 示教（80 source + 80 randomized），不追加 120 条 DAgger 纠正；初始化的 V2 80k 历史曾使用 DAgger。
- 当前纯 WAM 基线关闭 VLM、高层规划、ASK、记忆写入及恢复扩展。完整机器人系统是另一个建设目标，不能将其描述为本版本已验证能力。独立 OGM-WAM 实验不在本发布范围内。

## 代码导航

| 路径 | 用途 |
| --- | --- |
| `harness/wam_precision_v4.py` | V4 网络、Chunk-VTT、融合与相机射线 |
| `harness/zhuhai_precision_v4_training.py` | V4 数据包装、模型迁移和严格加载 |
| `harness/zhuhai_precision_v4_future16.py` | +16/+32 标签、损失与契约绑定 |
| `scripts/train_zhuhai_wam_precision_v4_future16.py` | 当前训练入口 |
| `scripts/train_zhuhai_wam_precision_v4.py` | 分阶段训练、优化器及恢复 |
| `scripts/eval_zhuhai_wam_precision_v4_future16.py` | 当前单次仿真评测入口 |
| `scripts/eval_precision_v4_future16_final5.py` | 训练完成后固定五种子评测队列 |
| `scripts/eval_precision_v4_future16_lineage.py` | 谱系相关评测入口 |
| `scripts/finalize_precision_future16_lineage_20261013.py` | 本轮结果独立核验 |
| `scripts/with_project_gpu.py` | NVIDIA 驱动一致性检查 |
| `scripts/gpu_lock.py` | GPU 使用互斥 |
| `configs/` | 历史和当前配置 |
| `third_party/` | 第三方工作树及原始许可证 |
| `environment/` | 精确依赖快照、来源和评测摘要 |

历史 V2/V3 与其他任务脚本保留用于导入依赖、检查点兼容和审计，不代表应启动历史训练。旧失败 V3、错误 exec16 分支不得自动重启。

## 环境安装

原训练环境：Linux x86_64、Python 3.10.21、PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、transformers 4.57.6、NumPy 2.2.6、h5py 3.15.1、SciPy 1.15.3；实际单卡 RTX 5060 Ti 16GB。需安装兼容的 NVIDIA 驱动；仿真需独立安装 NVIDIA Isaac Sim 并接受其许可证。

```bash
git clone git@github.com:susongyuan/Predict-WAM-V4.git
cd Predict-WAM-V4
python3.10 -m venv .venvs/wam-fixed
.venvs/wam-fixed/bin/python -m pip install --upgrade pip
.venvs/wam-fixed/bin/python -m pip install -r requirements.txt
```

`requirements.txt` 引用训练环境的精确版本快照。`environment/training-inventory.json` 和 `conda-base-inventory.json` 同时记录 Python 与 Conda 底层依赖。原环境是继承 Conda site-packages 的 venv，快照是实际安装清单，不是经干净机器求解验证的跨平台 lockfile。若特定平台包无法安装，应对照清单处理，不能默默替换版本并声称严格复现。

仿真环境单独准备：

```bash
python3.10 -m venv .venvs/robotwin-sim
.venvs/robotwin-sim/bin/python -m pip install --upgrade pip
# 先按 NVIDIA 官方流程安装匹配版本的 Isaac Sim，再按实际清单检查依赖：
# environment/requirements-simulation-snapshot.txt
```

仿真清单可能包含 vendor 包，需要对应 NVIDIA 包索引及运行时，不能保证普通 PyPI 一次安装成功。不要把仿真和训练依赖盲目合并。第三方来源、提交及本地修改状态见 `environment/third-party-provenance.json`，仓库中保存的是实际源码副本而非只记录上游 commit。

GPU 命令统一从以下包装器启动：

```bash
python3 scripts/with_project_gpu.py -- .venvs/wam-fixed/bin/python YOUR_SCRIPT.py
```

包装器首先检查当前驱动；健康时直接运行。不匹配时会查找原机器的本地验证 bundle，该 bundle 不随仓库分发。新机器应安装匹配驱动，不能复制原机器驱动路径或跳过验证。

## 必需的外部资源与路径

1. 原 V2 80k 初始化检查点：
   `outputs/zhuhai2026_tasks12/wam_precision_v2_task2_dagger30k_diagnostic_20261011_v23/train/checkpoint_step_080000.pt`。
   SHA256：`1de0c8b337bd3a01eec102e3b44df121dfa999bb9324d1c51b2f2a26f264f93c`。
2. 原 TRAIN/VAL 划分、训练 manifest、统计量及 HDF5 示教。路径与 SHA 绑定在原实验契约中；应从原工作区取回，不能用随机数据替代。
3. 冻结 DINO 模型及对应图像预处理。模型目录位于原工作区 `models/`，不随本仓库上传。
4. 原任务的机器人 USD、运动链、场景及物体资产，含 `data/isaac_zhuhai/franka_official_v1/`。RoboTwin 的 background_texture、embodiments、objects 外链资产不包含在第三方源码中，需从原合法来源取得。
5. 特征缓存默认 `data/zhuhai_hjepa_hybrid_features_20261009`；支持按需编码与缓存，全部预建不是启动前提，但缺失原数据或编码器不能运行。

代码仍有原工作区路径 `/home/user/Robotic/project2` 和证据文件依赖。移植时检查 `rg '/home/user/Robotic/project2|outputs/|data/' harness scripts configs`，根据新环境逐项重新生成配置与契约。不要全局替换已签名的历史 manifest，也不要绕过 source hash 检查。本发布没有伪造或改写原实验谱系。

## 训练与恢复

先查看帮助并生成计划；不带 `--execute` 不开始训练：

```bash
.venvs/wam-fixed/bin/python scripts/train_zhuhai_wam_precision_v4_future16.py --help
.venvs/wam-fixed/bin/python scripts/train_zhuhai_wam_precision_v4_future16.py \
  --out outputs/new_v4/train --gpu-lock-owner new-v4 --steps 50000 --batch-size 4
```

原预算新增 50000 更新，累计从 80000 到 130000：前 1000 更新只训练预测器；1000–1999 开启 projection/alpha；其后低学习率联合训练；主干 LR 1e-6、新模块 LR 1e-4，每 2500 保存里程碑、每 500 保存完整恢复状态。默认 CLI `--steps` 是 30000，因此复现 50k 必须显式指定。

准备并核验资源、契约、磁盘、RAM 和 GPU 锁后，按 `scripts/gpu_lock.py --help` 获取互斥，才可在上述命令添加 `--execute`，并通过 `with_project_gpu.py` 启动。已有目录不会被静默覆盖；`--resume` 需要 manifest 和合法完整状态，failure_guard 存在时必须先诊断并保存证据，不能盲目删除 guard。

历史正式运行前 5000 更新 batch4，其后 batch23，最终 RAM 特征缓存 12GiB；这与从头 batch23 或固定 batch4 的样本预算不同。不要把示例命令当作完全相同的历史训练重放。原 full-state 包含优化器及 RNG，model-only 权重不能冒称精确恢复。

## 评测、核验与录像协议

```bash
.venvs/wam-fixed/bin/python scripts/eval_precision_v4_future16_final5.py --help
.venvs/wam-fixed/bin/python scripts/eval_precision_v4_future16_final5.py \
  --training outputs/new_v4/train --out outputs/new_v4/fixed5 --gpu-lock-owner new-v4-eval
```

固定种子 932101–932105，每例 2400 步；127500 与 130000 各五次，严格原评分及官方制动。实际执行前须核对谱系修复入口及训练绑定，不应直接把旧队列结果当作最终结果。最终结果核验逻辑见 `finalize_precision_future16_lineage_20261013.py`。

默认录像关闭，不创建评测专用第三视角；保留机器可读动作、评分、规则与必要机载观测。机载传感器档案不是视频。失败不自动重试，不替换种子择优汇报。

CPU 定向检查示例：

```bash
.venvs/wam-fixed/bin/python -m unittest scripts.test_wam_precision_v4
```

部分测试有意读取原训练 manifest、缓存或证据文件，因此没有外部资源时不会全套通过。CPU 单元检查不能替代闭环物理评测。源码发布时的检查结果见 `environment/release-checks.json`。

## 已验证结果与局限

2026-10-13 最终独立核验：原 V2 80k 0/5；V4 127500 0/5；V4 130000 0/5。最终 V4 平均任务进度下降，不能宣称成功率改善。精确比较值见 `environment/evaluation-summary.json`。五个固定场景不能证明泛化或独立因果贡献；训练阶段 batch 改变也使样本预算发生变化。

原始动作、观测、检查点、失败日志及完整证据仍保存在原工作区，未随代码上传；摘要不能替代原始证据。当前仓库未在干净机器上完成训练或仿真复现。

## 发布完整性与许可证

`environment/source-manifest.json` 记录导出源码的 SHA256；不包含 Git 元数据和清单自身；README 及检查收据也有记录。发布不包含 SSH 私钥、访问令牌或本机虚拟环境二进制文件。

第三方代码遵守各自原始许可证，许可证文件随源码保留。本项目自有代码尚未指定开源许可证；仓库可见性不构成额外授权。
