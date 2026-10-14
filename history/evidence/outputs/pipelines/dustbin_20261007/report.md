# 双臂Aloha瓶子入垃圾桶：30k WAM实验

更新：2026-10-07T13:38:26.856619+08:00；状态：complete

任务：RoboTwin2 `put_bottles_dustbin`，官方clean50示教，40/5/5划分。双臂机体；三个刚体瓶投入敞口垃圾桶。**不包含开盖或真实柔性物体。**

纯WAM：三RGB+原生14D commanded state、零条件、chunk32/exec8、K1/Euler10；无VTT/VLM/反馈恢复/专家动作。GT只评分。

|检查点|有效完成|原生成功|原生失败|初始化无效|其他未完成|成功率|
|---|---:|---:|---:|---:|---:|---:|
|25000|5/5|4|1|0|0|0.8|
|30000|5/5|3|2|0|0|0.6|

两检查点使用相同预登记920001–920005种子，每条独立进程、保存onboard和第三视角录像。每条最多1700控制步；5100秒episode/5400秒total/5460秒systemd界限，防止用旧700步预算误截断。

原生成功仅检查三个瓶子原点在垃圾桶的指定xy/z范围；不证明严格完整容纳、释放或持久稳定。无效/中断不计策略失败。

训练证据：`outputs/runs/put_bottles_dustbin/wam_30k_20261007/train`；测试逐次结果和录像：`outputs/runs/put_bottles_dustbin/pure_25k_30k_5each_20261007`。

## 最终独立核对

| 种子 | 25k | 25k步数 | 30k | 30k步数 |
|---|---|---:|---|---:|
| 920001 | 成功 | 516 | 成功 | 521 |
| 920002 | 成功 | 515 | 成功 | 516 |
| 920003 | 失败 | 1700 | 成功 | 780 |
| 920004 | 成功 | 521 | 失败 | 1700 |
| 920005 | 成功 | 657 | 失败 | 1700 |

- 两权重SHA256、全部10条原生终局、录像帧数、动作数和逐动作`zero_single_task`条件已核对。VTT适配/监测均为null，无专家、外部VLM或恢复。常规执行器限幅及原生TOPP保持行为仍存在并完整记录，不冒称无低层控制器。
- 25k在920003失败，30k在920004/920005失败；三条均为1700步终局，不是超时/环境错误。不能用5个固定种子的差异宣称25k普遍优于30k。
- 视觉复核证实机体运动与桌上物体变化；第三视角未覆盖垃圾桶内部全过程。成功结论依赖官方原生谓词，严格完整容纳/释放/5秒稳定仍未知。
- [机器核对回执](final_verification.json) · [逐次CSV](per_episode_results.csv) · [缓存清理回执](posttest_cache_cleanup.json)。

## 2026-10-08追加：TV-WAM五任务共享50k（独立实验）

本段作为既有实验汇总追加保存，不与上面的单任务25k/30k开发种子成绩合并。共享50k固定种子922001–922010，每任务10次，原生成功34/50（68%）：

|任务|原生成功|有效测试|
|---|---:|---:|
|投瓶入垃圾桶|4|10|
|易拉罐放入塑料盒|1|10|
|打开笔记本|9|10|
|按铃|10|10|
|按订书机|10|10|

40TRAIN/5VAL/5封存每任务；TRAIN-only VTT、冻结DINO、共享动作模型与VTT残差投影联训，投瓶旧单任务30k初始化优势已披露。头部+双腕、native14指令状态、历史与门控、chunk32/exec8；本轮没有VLM、外围监督恢复或专家替代动作。原生谓词成功不等于真实按压行程/功能触发或长期稳定容纳；没有消融或同协议基线，不能宣称改进的因果增益。

- 最终权重：`outputs/runs/multitask5_wam_v3_vtt_20261008/extension_50k/attempt2/train/checkpoint_step_050000.pt`；SHA256 `8960be6ceb4d5293d2c5455ac4308421f64f1e31bf64f0646b7208735f63a5f4`。
- [50条成绩](../../runs/multitask5_wam_v3_vtt_20261008/extension_50k/attempt2/eval/summary.json) · [完成回执](../../runs/multitask5_wam_v3_vtt_20261008/extension_50k/attempt2/completion.json) · [独立复核](../../runs/multitask5_wam_v3_vtt_20261008/extension_50k/attempt2/final_audit.json)。全部50条原始机载观测、动作和评分保留原位。
- [五任务成功回放合辑](../../runs/multitask5_wam_v3_vtt_20261008/extension_50k/attempt2/success_replays/five_tasks_success_replays.mp4) · [回放来源manifest](../../runs/multitask5_wam_v3_vtt_20261008/extension_50k/attempt2/success_replays/manifest.json)。由原成功回合头部+双腕观测导出，10fps，末尾2秒标注定格；没有重新测试、不是实时速度。已有5条单独回放全部保留。
- [旧结果保全manifest](../../zhuhai2026_tasks12/old_results_preservation_manifest.json) · [缓存清理候选](../../zhuhai2026_tasks12/cache_cleanup_candidates.json)。仅重建DINO特征payload与项目自有Python字节码纳入清理，原始HDF5/权重/训练记录/评测证据不删除；原训练resume须显式重建缓存。

清理执行完成：[清理回执](../../zhuhai2026_tasks12/cache_cleanup_receipt.json)。释放派生特征及字节码逻辑空间约204.65GiB；全部旧结果文件大小及已登记SHA256复核一致，250条原HDF5、230份缓存metadata与所有权重/机载证据/回放保留。
