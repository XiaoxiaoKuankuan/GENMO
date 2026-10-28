# 2026-10-28 Stage2-A交替训练投影提速与八组正式配置

本文记录本轮实际代码、固定真实夹具、物理对照、部署边界。运行产物保存在Server-1，
不提交模型、Replay或音乐数据。性能优化没有改变冻结GENMO、冻结SuperMimic Actor、
十帧270维Adapter、独立WM真实H8监督和闭环BPTT主线。

## 方法及实现

Adapter仍为2654→512→512→256→270、三个SiLU隐藏层，输出十帧独立残差。
根XY关闭，根高度/根姿态/21关节保留。原SM六项Tracking、action-rate及原精度成本
保持；音乐权重1e-4，额外body_w Huber、软限位及Critic终端价值关闭。

`so3_projection.py`缓存静态差分矩阵、30维单位阵及线搜索步长。SO(3)数值求解中
仅原30维值/Jacobian/Hessian纯张量核采用128/256两个固定batch的编译核，活跃行
不足时复制独立行填充，计算后只取实际行；较大批量分块，不编译动态SQP循环。
第一轮乘子严格为零时Hessian精确为单位阵；内点QP的predictor/corrector共用一次LU
分解。收敛检查每四次同步一次，已收敛行冻结，最终KKT检查保持。

数值前向与隐式反向仍分开：前向解严格约束最邻近投影，反向使用真实非线性约束
Jacobian和Lagrangian Hessian解KKT；对提案、历史锚点、名义姿态和边界参数的VJP
保持。没有直通估计、截断主奖励梯度或扩大外部运动预算。最终实际FP32输出仍须
通过原证书，根角加速度对外预算3rad/s²，数值内部预留约0.01rad/s²余量。

尝试过6/9/3维局部导数组装：数学对照正确，但GPU编译及稳态不优，正式代码已移除，
研究证据和Git历史保留。没有把该尝试的结果当成正式实现。

`vector_control.py`在一次调用内复用名义Goal/13点姿态/旋转，避免重复构造；名义缓存
仍只读。新增原提案与投影结果的归一化RMS、关节/旋转距离P50/P95/P99/max。
可配置辅助成本为

`C_feas = mean(((raw_residual - projected_residual) / bounds)^2)`，

每步贡献`-dt * alpha * C_feas`，raw与projected两边均保留梯度。
它惩罚投影距离，可行的非零补偿成本严格为零，不以零残差为唯一目标。
主奖励仍完整穿过投影、冻结Actor及WM回到Adapter。

`alternating_learning.py`把microbatch日志先在GPU累计，Adam前统一检查有限性，结束后
合并回传CPU。孤立不可行样本按bad_mask拒绝并重算其余行，保存真实采样身份；不再
因一个样本连带拒绝整个256批次。非有限损失即便梯度有限也不允许Adam更新。

`timing.py`记录CUDA事件与外层同步墙钟，包含前向、反向重算、KKT、Actor、WM、
奖励及optimizer。嵌套区域为inclusive，不能直接相加；外层同步墙钟用于吞吐。
真实采集、WM10次、checkpoint、专用物理评价另计。为autograd引擎线程增加计时
上下文回退；这仅补全日志标签，不改计算图。

`alternating_entry.py`改为独立子进程创建真正4环境评价仿真，父2048环境不推进或复位。
同歌曲、同初态、同物理参数、无探索，复用完整评价合同；zero基线按4环境及预算缓存，
也嵌入完整checkpoint以便恢复。validation选best，evaluation只作独立报告。
`training_report.py`生成整个正式根的`comparison_latest.json`，严格区分前缀和整歌。

## 固定1024真实起点完整Adam更新

证据根：
`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/performance_20261028_2105`。
`full1024/fixed_starts.pt`来自真实collector的1024个不同环境，不复制补足。
各case加载同一保存checkpoint、相同Adam状态、相同模型和BPTT起点，全部1024有效、
拒样0；编译预热单独记录。以下单位秒，包含完整前向、反向和Adam，非异步提交时间。

| 时域 | 实现 | micro32 | micro64 | micro128 | micro256 |
|---|---|---:|---:|---:|---:|
| H8 | 原实现 |115.90|70.70|43.86|25.60|
| H8 | 本轮 |66.85|40.10|23.61|16.26|
| H16 | 原实现 |223.37|137.95|83.73|51.47|
| H16 | 本轮 |131.47|78.31|46.33|31.63|
| H32 | 原实现 |452.02|274.74|167.88|100.06|
| H32 | 本轮 |262.23|155.96|92.17|62.53|

实际选择micro256。H32原micro32→本轮micro256约7.23倍、时间减少86.2%；
同micro32优化约42%，同micro256约37.5%，不能把batch收益全算作SQP代码收益。
最优H8/H16/H32平均GPU利用率80.52%/79.36%/80.96%；Torch峰1411/1624/2031MiB，
nvidia采样峰3861/3947/4453MiB。这里没有2048物理环境及Replay，不代表正式整进程显存。

最优配置相对原micro32的H8/H16/H32目标差为0/1.19e-7/0，梯度相对L2差
0.001744/0.002680/0.002873，Adam参数最大差7.08e-8/1.10e-7/2.12e-7。
原实现自身跨micro归约也有约0.17%～0.27%梯度差，不能宣称逐位一致。

固定32样本H32原17.618→10.197秒，目标差2.38e-7，梯度相对差1.15e-6。
另用当前已保存u50真实模型及其1024起点，在相同micro256对照：H8 14.40→8.58、
H16 29.31→17.35、H32 57.86→36.20秒，目标差0，梯度相对差最大1.0e-7。
旧失败模型与当前模型活跃比例不同，不能用某个较快case承诺训练后期固定耗时。

H32本轮micro256：投影inclusive52.575秒，其中SO3数值50.288秒；Jacobian1.611、
Hessian3.683、IPM QP29.345秒；Adapter0.080、Actor0.125、WM0.216、奖励/历史1.586、
总backward7.293秒。内点QP和数值前向仍是主要瓶颈，MLP不是主要耗时。
Kineto独立短trace保留同步、eigh、LU、solve等算子证据，不与吞吐case混计。

Actor/WM的非投影compile同卡H32约67.55→61.31秒，但梯度相对变化约0.353%；
本轮正式保持eager Actor/WM，仅使用已验证FK与SO3数学核compile，避免增加重编译分支。

## 数值、辅助项和真实物理验证

`failed8_gpu/projection_failures.json`原八份失败输入全部FP32证书通过、隐式梯度有限。
根角加速度约2.990rad/s²；外部3rad/s²及KKT阈值没有放宽。
CPU回归覆盖SO3有限差分、锚点/名义/边界梯度、完整恢复、奖励合同、孤立拒样及有限性保护。

固定训练起点校准alpha=0.001：总梯度变化0.0888%，方向余弦0.999999606，
两次均未触发clip；实际动作变化RMS 0.00339497→0.00339410，未压制非零补偿。
该权重在训练夹具预先固定，未用evaluation选权重。

专用4环境对照使用同一已保存u20模型，原/新求解器各跑validation和evaluation整歌，
candidate各重复2次，zero各一次，共48个完整episodes，全部完成100%、约束失败0。
evaluation根RMSE两次均值0.075403→0.075255m，validation为0.103648→0.103901m；
差异均小于重复仿真波动。q/qd、身体姿态、原SM六项、jerk和真实接触脚滑亦在重复波动
量级。结论是本轮提速未观察到实质物理退化，不是新训练已获得显著收益。
完整指标和初态身份见`physical_equivalence_summary.json`及`results_summary.json`。

## 八组正式配置

八份配置：`configs/closedloop/stage2_fast_20261028/gpu0.yaml`至`gpu7.yaml`。

| GPU | BPTT | alpha | seed |
|---|---:|---:|---:|
|0|16|0|42|
|1|16|0|43|
|2|16|0.001|42|
|3|16|0.001|43|
|4|32|0|42|
|5|32|0|43|
|6|32|0.001|42|
|7|32|0.001|43|

共同：2048真实环境、Adapter batch1024/micro256、LR1e-5，WM LR3e-5、batch2048、
真实递归窗口8、每轮10次，100轮WM-only后连续交替，目标1000次有限Adam。
根姿态开启、根XY关闭、Critic关闭、采集/评价探索噪声0。使用相同64首冻结GENMO训练
缓存及独立validation/evaluation，每组模型/优化器/Replay/RNG/日志/checkpoint独立。
从零Adapter及同原预训练WM开始，不恢复旧退化模型。四条件双种子只比较时域和辅助项。

更新0/5/10做固定256步前缀；20/50/100/200/400/700/1000做整歌，best只按validation。
每50保存完整checkpoint，另保存零、首个Adam及best/latest。真实效果差不阻塞连续训练；
NaN/Inf、OOM、求解器异常、无法安全生成参考仍保存现场并停止对应组，不允许掩盖硬故障。
不使用DDP、torchrun、NCCL或跨组共享在线数据。

启动命令模板（实际root、源码SHA、PID和首更新另由launch_manifest及核验记录给出）：

```bash
/data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python -B \
  tools/launch_stage2_alternating.py --root NEW_TIMESTAMP_ROOT \
  --config-dir configs/closedloop/stage2_fast_20261028 \
  --session-prefix NEW_PREFIX --gpus 0,1,2,3,4,5,6,7 \
  --wait-for-free-gpus --poll-seconds 15
```

## 边界

没有测试H64或Stage2-B；本轮目标是已授权Stage2-A性能与八组启动。有限差分与真实整歌
对照不能证明训练最终一定优于零Adapter；新八组须用后续固定整歌结果判断。性能矩阵不含
全程采集/WM/评价/保存成本，正式日志会分别记录，不能用纯更新时间承诺1000次完成时刻。
旧实验主动替换前保存完整现场，原输出目录保留，不触碰无关任务。

## 实际启动与首更新核验

Server-1已在2026-10-28 23:01实际执行上述启动器，运行根为
`/data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/formal_fast_20261028_2301`，
session-prefix为`smfast8_20261028_2301`。运行源码冻结在
`734acc9781724b9ce367c3b1dab74ffc592099c9`，配套SM为
`a0911e4bbbffe96e1d6bc842b29f4b01c2ebe7a6`；后续文档提交不会热替换运行源码。

| GPU | 训练PID | tmux | 首次参数变化L2 | 首次更新秒 |
|---|---:|---|---:|---:|
|0|3485881|smfast8_20261028_2301_g0|0.00253025|3.66|
|1|3487374|smfast8_20261028_2301_g1|0.00253014|3.75|
|2|3487432|smfast8_20261028_2301_g2|0.00253009|3.96|
|3|3487491|smfast8_20261028_2301_g3|0.00253007|3.93|
|4|3487532|smfast8_20261028_2301_g4|0.00253261|7.73|
|5|3487593|smfast8_20261028_2301_g5|0.00253256|7.87|
|6|3487861|smfast8_20261028_2301_g6|0.00253253|7.72|
|7|3488145|smfast8_20261028_2301_g7|0.00253252|8.07|

每组首更均1024有效起点、拒样0，WM已1010次，训练角色真实转移647168；
网络近零初始化时投影活跃程度低，首更耗时不能冒充后期压力夹具吞吐。
23:04核验时各组更新5/5/5/5/4/4/4/3，训练真实转移约66万，整卡9543～12432MiB，
Torch训练峰7360.8～7361.9MiB。评价子进程运行时整卡包含额外4环境，记录与纯更新峰不同。
实际CUDA日志已出现`backward/projection.SO3_KKT_implicit_backward`，线程回退计时有效。

八份首更完整checkpoint逐一以CPU mmap只读解析通过，每份约3.72GB，保存Adapter/WM/
Critic与三个优化器、冻结bundle、Replay、collector残差/音乐状态、四类RNG、两个专用零基线。
Critic关闭时其优化器不执行step，不把“保存了Critic”当成启用了终端价值。
CPU恢复语义回归包含在123项测试中；没有为演示恢复而打断新正式训练。

所有GPU都已完成第5次专用256步配对评价，参考约束失败0。早期前缀q和身体姿态误差
未改善，不能称新模型优于零Adapter，也不能拿前缀完成率0当整歌摔倒率。
第20次起自动执行整歌，结果好坏不作为每次Adam门禁。

运行根包含`launch_manifest.json`全部八条完整命令、`launch_verification.json`的PID/argv/
cwd/GPU/首更/显存证据、`checkpoint_verification.json`及自动`comparison_latest.json`。
每组目录为`gpu0`至`gpu7`，独立`training.jsonl`、`timing.jsonl`、`latest_checkpoint.json`、
`best_checkpoint.json`及物理评价JSON；外层日志为`gpuN.log`。精确配置为根目录
`config_gpuN.yaml`，运行启动器可用的源配置为仓库八份YAML。

原GPU0/1/2/3/7在u260/261/184/183/117主动保存退出，GPU4/5/6之前也已保存；
KeyboardInterrupt现场是本轮授权替换，不能记成新训练数值失败。
旧结果、失败基线及`genmo-rolling-20261022_182720`无关会话均保留。
