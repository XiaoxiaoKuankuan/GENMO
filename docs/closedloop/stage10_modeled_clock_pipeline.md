# 第二阶段独立部署时钟与采样外围批量化

本文记录2026-10-09在 `feat/stage10-gpu-vectorized` 分支完成的修改及服务器1八卡验收。
目标是让训练环境数不再隐式改变部署任务，并消除重复的逐环境数值处理。原20步
DPPO、冻结GMT、真实PhysX反馈、BC、KL、GAE、全局160条及每500轮保存规则保留。

## 两种时钟

新配置显式使用 `modeled_deployment.v3`。目标部署场景是单机器人、单请求；profile
绑定服务器1八张卡各自N=1的16次关键路径实测以及八份来源文件SHA。测量值约
0.243～0.261秒，按50Hz控制网格向上取整得到0.26/0.28秒；前缀预算固定0.38秒。
这是服务器1上的部署代理测量，不是真机延迟承诺；更换目标硬件或部署服务时应重新
测量并显式更换profile，不能静默覆盖历史身份。

每条动作的延迟由base seed、任务sample_id、音乐起点、decision_tick确定性抽样。
键不含训练rank、env_id、环境数、批量形状或run UUID；无需额外可变RNG。批次排队、
GPU计算和审计I/O仍完整计入训练墙钟，但不决定模拟到达时刻。startup校准只测
训练管线性能，不重新设置模拟延迟。旧 `deployment_critical.v2` 实时时钟仍可显式使用。

模拟等待期间按原规则推进旧参考、50Hz控制和200Hz物理；提交必须发生在正确控制
点，奖励、实际执行时长与GAE没有省略。P仍由前缀保护规则和真实参考支持决定，
不是强制截短为某个数。本次N1/N8任务观测到的真实P均为19。

时钟profile SHA进入checkpoint的每环境状态，恢复时必须匹配；独立审计器根据原
配置和任务重算每条到达记录。旧时钟checkpoint不能作为新时钟run的完整恢复。
初始化、周期和结束评估沿用同一profile，新曲线必须与旧时钟曲线分开解释。

**P的历史统计口径纠正：** 原摘要的18/51是全部30坐标已知的帧数，漏计最后仍有
两个自由root坐标的前缀末帧；对应真实P是19/52。本次统一读取权威的
`generated.prefix_frames`，原始历史产物不改写。环境数与时钟耦合的判断不受此影响。

## 采样外围修改

| 文件 | 修改及原因 |
|---|---|
| `gem/closedloop/dppo/deployment_clock.py` | 独立profile、确定性延迟和来源SHA，消除训练拓扑对部署时钟的影响。 |
| `env_adapter.py`、`vector_collector.py`、`vector_environment.py` | 分离模拟到达与观测墙钟；校准不再覆盖profile；保留参考保护和因果执行。 |
| `vector_generation.py`、`gem/closedloop/online_conditions.py` | 同P前缀成组编码，实际条件统一传GPU，20步扩散和CFG合批，世界坐标转换及完整链回传按批处理；CPU切片只保留所属样本的存储。 |
| GMT `vector_columns.py`、`vector_backend.py`、`vector_service.py` | GPU证据直接形成时间列，再按环境创建列视图；省去深层逐步对象展开后再次打包，四个物理子步全部保留；首次区间严格对照原路径。 |
| `gem/runtime/trajectory_blocks.py` | 多段真实执行直接拼接列块，不重复恢复完整控制步对象。 |
| `vector_metrics.py`、`parallel_support.py`、`vector_validation_learning.py`、训练/审计工具 | 绑定新配置与恢复身份；记录P、模拟延迟、共享批次、RPC/journal/ACK及证据分项。 |

环境独立预算、事务序号、参考身份、episode和音乐奖励窗口仍各自维护。这里消除的是
重复的大张量处理与设备同步，不把不同机器人的因果状态混为一份。两个journal依然
先持久化再ACK；未丢弃原始链或物理执行证据。GMT CUDA事件在有界区间末读取，
避免每控制步额外等待；`gmt_seconds`是GPU事件时间，主机提交及重叠物理阶段不能
直接相加来复原采样墙钟。

## 同工作量验收

全部测试位于服务器1，未在本地运行测试。配对设置固定模型、任务、噪声身份和
同一部署profile，N8每轮均160条真实上层转移、4800个实际控制步（包含轮末继续
执行的800步；第一轮预热另外计入预算）。每批真正对应8/8/4个不同环境请求。

| 采样场景 | 第1轮秒 | 第2轮秒 | 验证 |
|---|---:|---:|---|
| 相同时钟，旧逐环境外围 | 19.6655 | 15.1829 | 320条配对参考 |
| 相同时钟，新批量外围 | 17.7470 | 13.7661 | 所有扩散链、逐步奖励、时序和终止标记完全一致 |
| 新外围，每批额外注入500ms墙钟等待 | 20.8431 | — | 160条与未注入链、奖励及时序完全一致 |
| 新外围，每卡1个环境 | 34.0612 | 32.5777 | 全局160条；3981/4000控制步，真实P仍为19 |

相同时钟下，外围重构的连续状态采样减少1.4168秒，约9.33%；首轮减少1.9185秒。
新N8与N1的任务覆盖、连续片段及控制量不同，不能称为逐动作配对的纯提速实验。
旧时钟N8约33～36秒的采样还执行了更多物理步骤，不能把这部分差额全部算成算子优化。

直接列式证据的第二轮分项最大值由0.9450秒降为0.6814秒，GPU/CPU阶段有重叠。
生成批次、实际物理、世界RPC与journal分别记录，没有把采样残差全部归为磁盘。

正式八卡入口另从Stage1新建run，在第1轮保存并退出新进程恢复至第3轮，再从第3轮
恢复至第6轮。最终独立审计30项全部通过，0失败、0未执行，含八rank/64环境时钟
身份、完整执行计数、模型/优化器恢复、冻结GMT关闭回执及全部32任务评估记录。
最终版本的部署时钟、条件、GPU回传、GAE、前缀、产物审计回归在服务器1八rank各
69项全部通过；测试中的真实GPU回传使用各自LOCAL_RANK，未跳过GPU测试。

| 正式轮次 | Actor更新数 | 采样秒 | Actor阶段秒 | 完整KL秒 | 核心秒 | 完整轮墙钟秒 | 边界 |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 1 | 17.8274 | 6.0234 | 1.0893 | 32.3090 | 41.4202 | 初始化后首轮、受控结束保存 |
| 2 | 2 | 21.6156 | 11.2241 | 1.0361 | 41.0145 | 41.7867 | 新进程恢复、环境重建 |
| 3 | 3 | 13.2116 | 16.9360 | 1.0316 | 36.0612 | 45.4038 | 受控结束保存 |
| 4 | 4 | 21.8097 | 22.0219 | 1.0277 | 51.9511 | 52.7669 | 再次恢复、环境重建 |
| 5 | 4 | 13.2238 | 22.1264 | 1.0308 | 41.8527 | 42.4415 | 连续普通轮，无保存/评估 |
| 6 | 4 | 13.9264 | 21.6671 | 1.0302 | 41.3188 | 53.2714 | 受控结束保存 |

每轮Critic均80次，约2.62～2.82秒。完整轮墙钟包括轮末封存、保存（若有）和归档
入队，不含会话最终独立评估、关闭及后台归档排空。第5轮42.44秒才是本次完整4更新
普通轮直接观测；第6轮41.32秒只代表核心阶段，不能将其冒称完整墙钟。当前瓶颈转为
Actor阶段约22秒。本次未修改Actor算子或优化次数，不能宣称Actor纯计算提速。

六轮全局平均内部joint KL依次为0.018755、0.019408、0.018342、0.014450、
0.011322、0.009640，均未超过原0.03硬门槛。初期1/2/3次更新来自原0.015软停止，
不是减少算法工作量获取虚假性能。训练首轮1次真实物理失败被正确计入终止，后五轮
0次；这些有限结果不证明策略质量提升或长期稳定。每500轮保存设置未改，受控结束
按原合同额外保存用于恢复验收。

源代码边界：六轮完整正式验收使用GENMO `2998252` / GMT `8cee138`；后续GENMO
`491e5fe`只补齐取消等待、观测计时标签、比较工具形状检查和说明，GMT `04a6ea6`
只修正首次证据核验计数。其最终真实采集/worker故障结果另附报告，不把旧源代码
跑出的六轮声称为每个后续提交重新跑过六轮。文档提交不更改数值/物理路径。

最终源代码的`worker_exit_clock_final_v1`先完成真实八卡160条、DPPO/BC/Critic、
严格概率及完整KL并保存第一轮，再故意杀死本测试rank2物理worker。九项故障核验
全部通过：八rank记录同一连接故障、未发布第二轮、原2,562,973,608字节断点SHA
未变、64个环境SQLite预算SHA链正确且不低于保存值、全部GPU工作进程退出。
这是预期故障测试通过，不能把测试入口的非零退出改称正常完成第二轮。

完整证据保留在服务器1：
`/data1/user/liwei/GENMO_outputs/closedloop_stage10/clock_pipeline_validation_20261009`。
精简JSON报告和字节/SHA清单位于本地：
`/home/weili/bumi-stage10-gpu-vectorized/时钟与采样外围验收_20261009`。
仅清理本任务临时pytest依赖及临时测试目录，成功/失败原始证据、模型和日志保留。

## 可复现入口

```bash
cd /home/user/liwei/GENMO-bumi-stage10-gpu-vectorized
bash scripts/validate_stage10_runtime_v4_server1.sh vector-collection \
  --config configs/closedloop/stage10_8gpu_server1_gpu_vectorized.yaml \
  --gmt-repo /home/user/liwei/legged_lab_gmt-gpu-vectorized \
  --output /data1/user/liwei/GENMO_outputs/closedloop_stage10/新的时钟采样验收目录 \
  --num-envs 8 --rounds 2 --comparison-seed 42
```

追加 `--inject-generation-wall-ms 500` 可做墙钟污染反例；该选项仅存在于有限测试
入口，不会进入正式配置。加 `--updates` 执行原Critic/DPPO/BC及完整KL验收。
配对分析使用 `tools/eval/compare_stage10_clock_pipeline.py`，不会修改原rollout。

正式接口有限启动/恢复仍使用：

```bash
bash scripts/train_stage10_gpu_vectorized_server1.sh \
  --output-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/新的正式接口验收目录 \
  --stop-after-iteration 1
bash scripts/train_stage10_gpu_vectorized_server1.sh \
  --output-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/同一验收目录 \
  --resume latest --stop-after-iteration 3
```

每次测试使用新目录，现有训练、模型和历史日志不覆盖。新时钟必须从明确绑定的
Stage1权重新建run；学习率5e-9、软KL0.015和硬KL0.03保持原设置。
