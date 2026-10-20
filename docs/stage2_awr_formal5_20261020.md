# 2026-10-20 正式 AWR：最多五轮启动记录与操作

用户已明确要求启动最多五轮正式训练，并批准整次运行最多26次额外采集尝试。
本次从原始Stage1 s595000新建运行，不修改已完成先导实验的request、模型或预算。
训练目标、奖励、冻结GMT、初始化、学习率和完整评价协议沿用已验收配置。

**已于北京时间2026-10-20 22:20:54启动。** 启动核对时状态为
`running / evaluation_stage1_fixed / macro=0`：部署校准完成，正在建立新运行的冻结
Stage1基线，尚未完成第一宏迭代。现场核验8个训练rank和8个GMT/PhysX后端存活，
至少2个完整评价任务已写入逐任务账本；没有用父进程或会话存在冒称Actor更新完成。

正式GENMO源码固定`6c5fda0e006e949e31a8003506e596403b0bcee7`，
GMT固定`4ca256fe79d2dda22240011947ead276690e40f7`。
回执父进程PID3033480，torchrun PID3033569，训练rank PID3033587–3033594。
固定request摘要为`6cf432adaaaba61c80929917e1fd75188fb421217864410438f25a08a7ca4d8b`。
[启动现场](validation/stage2_awr_formal5_startup_20261020.json)、
[冻结资产和空闲资源预检](validation/stage2_awr_formal5_preflight_20261020.json)、
[Server-2 CPU测试](validation/stage2_awr_formal5_cpu_tests_20261020.json)
均已归档。这是启动时点证据，后续进度和结果以实时state及正式报告为准。

| 参数 | 正式配置 |
|---|---|
| 最大宏迭代 | 5 |
| 第一轮完整Episode | 512 |
| 第2–5轮每轮新增 | 128 |
| 完整Episode目标总量 | 1024，包含生成后物理失败 |
| FIFO活动容量 | 512条完整Episode |
| 全运行额外尝试上限 | 26，所有轮次共用，不按轮重置 |
| 训练采集尝试上限 | 1050；独立评价不计为训练数据 |
| Actor | LR=1e-5、5 Epoch、全局256 AWR＋128 Stage1 BC |
| BC系数 | 每宏迭代开始校准一次，整轮固定，B/C共用 |
| Critic | 双网络交叉拟合；比较MC/FVI，按原内部验证规则选择 |
| 对照 | 每轮同Replay自蒸馏＋BC与AWR＋BC |
| 验证 | 每轮固定32任务，最后一轮追加全54；保留全部失败 |

配置：[stage2_awr_formal5.yaml](../configs/closedloop/stage2_awr_formal5.yaml)。
默认[单轮配置](../configs/closedloop/stage2_awr.yaml)仍然是1轮。
新配置只增加五轮上限和明确授权的补采总账，未改变Actor/Critic/奖励/采样参数。

仅原固定任务全部有结果、缺额确实来自首次GENMO决策前的真实预热失败时，才进入既有
补采事务。失败快照和原计划不变，生成后失败仍是有效完整Episode。
`collection_budget.json`按各轮不可变`wave*.json`预留扣除次数；失败或中断不退还，
恢复继续使用该轮已分配的上限。已有预热失败任务不会因为恢复而被再次当成新任务派发。
达到26次仍不足则保存证据并退出；数值、身份、数组校验和其他异常不会自动无限重试。
模型选优继续使用原成对物理门槛，未通过的模型不会替换`best_saved`。

正式源码独立目录：

```text
/home/user/liwei/awr-worktrees/GENMO-formal5-20261020
```

正式运行及其旁路启动/退出/日志：

```text
/data1/user/liwei/GENMO_outputs/stage2_awr/formal5_20261020_01/
/data1/user/liwei/GENMO_outputs/stage2_awr/formal5_20261020_01.launch.json
/data1/user/liwei/GENMO_outputs/stage2_awr/formal5_20261020_01.exit.json
/data1/user/liwei/GENMO_outputs/stage2_awr/formal5_20261020_01.log
```

`exit.json`只在真实子进程退出后生成；缺少它不是训练成功的证明。
正常运行时应同时查看worker PID、GPU计算进程、state阶段及日志更新时间。
源码启动后保持固定提交；操作文档后续更新不拉取到活跃训练目录。

首次启动由前台回执父进程执行，放在持久终端或独立后台会话：

```bash
cd /home/user/liwei/awr-worktrees/GENMO-formal5-20261020
/data0/user/liwei/closedloop_stage10_v3_runtime/env_isaac_genmo/bin/python -B \
  tools/launch_stage2_awr_formal5_server2.py \
  --run-dir /data1/user/liwei/GENMO_outputs/stage2_awr/formal5_20261020_01
```

停止、恢复和只读进度：

```bash
cd /home/user/liwei/awr-worktrees/GENMO-formal5-20261020
bash tools/run_stage2_awr_server2.sh stop \
  --run-dir /data1/user/liwei/GENMO_outputs/stage2_awr/formal5_20261020_01
bash tools/run_stage2_awr_server2.sh resume \
  --run-dir /data1/user/liwei/GENMO_outputs/stage2_awr/formal5_20261020_01
tail -n 60 /data1/user/liwei/GENMO_outputs/stage2_awr/formal5_20261020_01.log
```

先等待原进程退出再恢复，恢复不改变五轮/补采上限或源码身份。正常停止会排空正在执行
的Episode/评价任务并保存完整优化步，强制中断不拼接旧物理中间状态。
再次启动`launch_stage2_awr_formal5_server2.py`会拒绝已有目录/回执，避免覆盖正式产物；
恢复用原`resume`入口，并为新的启动另留日志和回执。

本次代码检查：69项AWR CPU测试通过，包括跨轮26次共享、8+7+11耗尽、恢复不重置、
中断预留不退还、未封存上一轮/篡改计划拒绝、原预热失败不重试以及正式参数与原配置
逐项相同。本地pytest7.64秒，Server-2同69项通过（pytest9.43秒），服务器测试禁用
CUDA可见设备。AST、bash语法、diff检查通过，测试临时目录自动清理。
这组测试与启动成功均不构成五轮物理效果结论；尚未提前记录学习收益。
