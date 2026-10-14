# 25k WAM＋现有中下层：五种子配对测试

更新时间：2026-10-07T17:10:40.170645+08:00；队列状态：completed。

中下层有效完成 5/5，原生成功 4；同权重纯WAM基线4/5。

|seed|纯WAM|中下层|控制步数|HOLD/重观测次数|动作完全相同|
|---|---|---|---:|---:|---|
|920001|成功|成功|516|0|True|
|920002|成功|成功|515|0|True|
|920003|失败|失败|1700|0|True|
|920004|成功|成功|521|0|True|
|920005|成功|成功|657|0|True|

实际启用：WAM25k、transaction execution、actual robot encoder feedback、41-sample motion window、at most2 measured-stall HOLD/clear/reobserve、episode-scoped controller-outcome memory。
未启用：VLM、ASK、VTT adapter/bank (no admitted adapter for this task/checkpoint)、full strict TaskSupervisor (no admitted task-specific perception/postconditions)、physical target grasp/containment verification、cross-episode task-success memory。
中下层采用原有控制运动停滞阈值，不是空抓/滑落/投放语义诊断；未将官方评估GT注入控制或经验。完整系统尚未接通，本轮不能冒称全功能系统验收。
与基线相同：50示教来源、相同25k权重与归一化、same5种子、native1700步、5100/5400/5460秒限制、serial三RGB、none denoiser、native8、无settle、保存录像及onboard。初始化无效同seed最多重试一次；中断不当失败。
原生成功只检查三个刚体瓶原点位于敞口垃圾桶范围，非开盖/柔性/严格完整容纳。若中下层未触发动作干预且与基线动作一致，不能声称它提高成功率。
详细证据：`outputs/runs/put_bottles_dustbin/middle25k_5each_20261007/comparison.json`及逐seed录像。

## 最终独立复核结论

- **只测25k，5次有效：4成功、1失败，0无效初始化、0中断、0重试。** seed920003仍在1700步上限失败；其他种子成功步数与纯WAM相同。
- 3909次真实机器人反馈已逐条记录，所有运行均0次停滞HOLD/清队列/重观测、0条控制结果记忆写入、0条任务成功记忆。并非仅修改mode标签，但本轮监测没有触发行为改变。
- 对照同seed纯WAM，5条原始动作序列和限幅后动作序列均通过float64字节级一致性核查。**本轮没有证明中下层提高成功率；成功来自未被改动的WAM执行路径。**
- 失败种子920003在step>=160的1540个运动窗口里，最小实测关节路径约0.3566rad、命令路径约0.5009rad，始终高于原停滞阈值0.05/0.1rad。因此“还在动”的失败不会触发机械停滞监督，不应靠本次成绩临时放宽阈值并混入同组统计。
- 现有路径未含匹配当前task/checkpoint的VTT、完整TaskSupervisor感知/物理后条件、空抓/掉落/入桶语义验证。本轮是已接通的中下层子系统测试，不是原设计全功能中下层验收。
- 独立核对：[verification.json](verification.json)；配对报告：[comparison.json](comparison.json)。两步smoke seed920010只作接线检查，不算入这5次成败。
