# GENMO 独立十帧参考 Adapter：事件音乐奖励与正式训练门槛

更新于2026-10-26，分支`feature/stage2-supermimic-adapter`，配套SuperMimic
`feature/genmo-worldmodel-bridge`。本轮用户最新指令指定**Server-1**运行；附件内的Server-2
路径不作为本次运行位置。本轮真实验收结果将在文末和`记录文本.md`追加，CPU通过不等于物理通过。

## 算法与执行边界

GENMO、SM Actor及所有原normalizer冻结。MLP保持三隐藏层512/512/256、SiLU、末层零初始化，
输出270=10×27：每帧根位移3、旋转向量3、RobotSpec名称映射的关节21。每20ms独立重规划
未来10点，不插值十帧、不广播当前残差、不把残差写回GENMO名义缓存或前缀。

InputSpec仍2654维：实际使用过的10×75控制历史、原Goal570、可观测误差、上一残差和修正
计划、已提交锚点、16×EDGE35/有效mask/相对时间、20×4音乐历史。xi与世界根状态只供WM和
Critic。Critic输入2835维，512/256/1。当前姿态继承旧计划，过去不可回写；SO(3)右组合，
统一原SM差分/FK。d11=2d10−d9只作用于真实存在的第11名义点，是导数halo，不加入Goal570。

`nominal_reference`、`corrected_reference`、`executed_state`保持独立。跟踪与终止始终使用
原GENMO参考。原MDP六项body_pos/body_ori/body_linvel/body_angvel权重1、root_pos/root_ori
权重0.5，身体集合、对齐、尺度完全复用；原动作变化−0.1与真实关节限位−2.0保留。
原SM返回已乘0.02，不再整体乘dt。附加项单独乘dt，reference_limit检查修正参考。

## 因果事件音乐奖励v2

旧逐点BCE把宽高斯肩部误当成多个运动事件要求，已经移除。EDGE35第34列产生固定绝对
音频拍点，600Hz整数时钟统一30Hz音乐与50Hz控制，不学习音频偏移。

执行能量仍为`mean(sqrt((qd/s0)^2+eps²)-eps)`，s0只由零Adapter真实训练集标定后固定。
对中心k使用k−2/k/k+2能量，两个sigmoid谷响应相乘，减去恒速响应后取正，乘以两侧能量
相对固定活动尺度0.1的平滑门槛。常值速度事件严格为零。弱谷仍有响应，缺少谷时没有
虚构的music梯度；不能拿正则或动作rate的梯度冒充执行状态路径。

- n时结算中心n−2的非拍点事件成本：`p*(1-b)`，b为最近固定拍点高斯包络，sigma0.04s。
- 拍点在±0.06s容差窗关闭后结算一次。覆盖率为`1-product(1-p*kernel)`，稳定使用
  `log1p/expm1`；缺失成本`1-coverage`，容差内总匹配质量超过1另加平方成本。
- 三项按音频固定beat/control-step密度归一化。R_music为负成本，音乐权重0.2仍是候选，
  必须报告真实分项和梯度规模；不靠降低权重修正排序。
- 事件延迟40ms，beat覆盖另等待固定60ms容差；原音频拍点允许预知，执行状态只来自
  已推进的真实/WM轨迹。窗口不完整、mask无效或无beat不补分；真终止后不构造后续状态。
- 尾随0.4秒活动不足成本独立相加，不乘掉Tracking。原布尔节拍函数只评价，报告F1、
  双向距离、事件密度、动作幅度、jerk及真实接触脚滑。

原信号数值回归：同步−0.760229、弱同步−0.930451、尖锐同步−0.288448、静止/恒速−1.055085、
四分之一周期错拍−1.589025、半周期错拍−1.572300、高频−6.903543、额外峰−2.498622。
原尖锐谷信号未改，原错误断言替换为正确排序要求。低活动趋近静止，不获取优秀奖励。
这些是合成数值结果；真实执行片段排序仍须服务器物理验收。

## 真实回放、尾支持与质量门槛

缓存必须保存冻结GENMO实际已生成且接管的源点，包括音乐结束后的真实尾部。支持至少到
`floor(music_end_tick/12)*12+132`；30Hz源点向上覆盖该tick，缺失直接诊断，禁止外推。
音乐完成按最后合法音乐控制tick判定，物理失败优先；reference_end不算歌曲成功或完整MC。

Replay记录true_start/head_truncated/true_terminal。容量截去当前episode前缀后，剩余数据
可用于连续WM窗口但不可用于MC。窗口不跨reset/song/参考版本/策略版本/滚动边界。
BPTT起点混合零Adapter和最近策略，WM验证仅取最新真实验证策略窗口。

Critic预热按完整真实MC拟合，逐epoch选独立验证MSE最佳网络和optimizer。默认事前门槛：
训练128点、验证64点、EV≥0.1、相对训练均值常数预测的MSE改善≥20%、失败前16步样本≥16、
失败附近MSE/常数MSE≤0.8。缺失败样本不是通过；有限预算内未通过则保存诊断并停止。
Critic参数在Adapter期间冻结，保留输入梯度；终端value进入目标，lambda目标stop-gradient。

WM从原bundle深拷贝，只用真实8步窗口监督，统计固定。每次拟合后以独立最新真实连续16步
核验1/8/16步：根位置≤0.05m、根旋转≤0.1rad、关节RMS≤0.1rad、关节速度RMS≤2rad/s。
启动前与定期真实方向审计分别直接扰动10×27残差及21维Actor动作；相同初态/前缀/RNG，
核验重复性≤1e−4，真实差分超过max(1e−6,5×重复回报差)，接触分支变化不能当通过。
残差扰动L2尺度0.005、动作0.01，记录实际根/关节/控制RMS、全参数梯度与噪声比。
相对最近实测WM参数漂移超过0.005触发重新验收要求，不能继续更新。

Adapter事务检查执行状态tracking+music参数梯度非零、参数有限且有变化、同状态残差和
控制确实改变、WM和Critic质量有效。失败回滚Adapter/Critic与optimizer，不增加accepted。
仅loss下降、正则梯度或模型回报提高都不能构成物理改善结论。

## 正式训练调度与恢复

reference_steps=10与bptt_steps=16独立；内部状态、历史、计划、音乐记忆保留计算图。
候选batch64、LR Adapter/Critic1e−4、WM3e−5，gamma0.99/lambda0.95按50Hz定义，clip1。
支持多轨迹计算图microbatch反传，并用batch1核对目标及全参数梯度；当前前向仍逐起点，
默认microbatch1。未把它描述为GPU矢量物理环境。

正式四组任务按song group、原音频SHA和特征SHA隔离：train/validation/audit/evaluation。
单物理环境只在真实episode结束后更新/审计；轮转必要训练歌曲和当前策略验证歌曲，
不会每轮重跑整个任务表。收集分段不会reset，恢复则必须新episode，因为求解器隐藏状态
未保存。日志报告真实/想象步数、吞吐、P50/P95/P99和20ms miss，不混淆仿真吞吐和实时能力。

零策略标定同时保存训练/验证完整真实回放，s0仍只由训练速度决定。多个分片按未截断
二阶矩/样本数合并，使用`--baseline-replay`核对任务、初态、音乐、名义与SM资产SHA后
复用物理数据；因果音乐记忆/奖励按最终固定s0重算，不修改SM奖励、真实状态或动作。
Critic多轮预热复用一次构造的完整MC特征/标签。`--preflight-only`禁止Adapter更新，
即使一个门槛失败也测量其他独立门槛并保存完整报告；失败不能解锁正式训练。
独立WM预热至少25次真实监督、最多200次后检查固定验证门槛，不能因原checkpoint
初始预测误差小而完全跳过独立WM拟合。预检恢复可复用已测Critic失败报告并继续WM独立
取证；此时失败状态仍保留，正式路径不能绕过Critic门槛。

启动完整H8/H16；首次及每25accepted小配对，每100完整固定评价/方向；每次WM真实验证与
执行梯度门槛。完成率下降、tracking相对下降5%、音乐代理下降0.1或F1下降0.05触发停止。
首次、每25、完成/停止写唯一编号完整checkpoint，其余只JSONL。保存网络、三optimizer、
RNG、回放元数据、音乐/残差/游标、接受计数、目标和审计锚点。`--target-updates 1000`是绝对
目标；恢复仍1000而非再加1000。行为v2不等价恢复旧逐点BCE，源码SHA只溯源不门禁。

## 入口与Server-1路径

Server-1实际代码目录为`/home/user/liwei/bumi-closedloop-worktrees/adapter/GENMO`；SM同级`SuperMimic`。
独立Python为`/data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python`，独立资产根
`/data0/user/liwei/GENMO_assets/adapter_20261026`；原Stage1继续只读使用
`/data0/user/liwei/closedloop_stage10_v3_assets/20261012/stage1_s595000`。
实际运行参数以验收目录的`assets/runtime.yaml`为准，不覆盖仓库示例。

```bash
python -B tools/prepare_stage2_supermimic_adapter_assets.py \
  --stage1-dir /data0/user/liwei/closedloop_stage10_v3_assets/20261012/stage1_s595000 \
  --sm-repo /home/user/liwei/bumi-closedloop-worktrees/adapter/SuperMimic \
  --sm-checkpoint /data0/user/liwei/GENMO_assets/adapter_20261026/ckpt_024000.pt \
  --robot-asset-dir /data0/user/liwei/GENMO_assets/adapter_20261026/sm/data/bumi3_4340/robot \
  --dataset-root /data0/user/liwei/datasets/bumi_music_umr70_mine_pass_90505_v1 \
  --output /data0/user/liwei/GENMO_outputs/stage2_supermimic_adapter/acceptance_20261026/assets
# 后续每个输出必须是不存在的新目录；不得覆盖失败证据或已有模型。
python -B tools/prepare_stage2_supermimic_adapter_reference.py --config RUNTIME --output CACHE --torch-device cuda:0 --visualizer none --deterministic
python -B tools/train_stage2_supermimic_adapter.py --config RUNTIME --tasks CACHE/tasks.json --calibrate-music --output S0 --torch-device cuda:0 --visualizer none --deterministic
python -B tools/train_stage2_supermimic_adapter.py --config RUNTIME --tasks CACHE/tasks.json --music-baseline S0/music_baseline.pt --baseline-replay S0/real_replay.pt --preflight-only --output PREFLIGHT --torch-device cuda:0 --visualizer none --deterministic
python -B tools/audit_stage2_supermimic_adapter.py --config RUNTIME --tasks CACHE/tasks.json --music-baseline S0/music_baseline.pt --physical --output AUDIT --torch-device cuda:0 --visualizer none --deterministic
python -B tools/train_stage2_supermimic_adapter.py --config RUNTIME --tasks CACHE/tasks.json --music-baseline S0/music_baseline.pt --variant music --formal --target-updates 1000 --output FORMAL --torch-device cuda:0 --visualizer none --deterministic
python -B tools/train_stage2_supermimic_adapter.py --config RUNTIME --tasks CACHE/tasks.json --music-baseline S0/music_baseline.pt --variant music --formal --target-updates 1000 --resume FULL_CHECKPOINT --output RESUME_NEW --torch-device cuda:0 --visualizer none --deterministic
python -B tools/evaluate_stage2_supermimic_adapter.py --config RUNTIME --tasks CACHE/tasks.json --music-baseline S0/music_baseline.pt --variant zero --output EVAL_ZERO --torch-device cuda:0 --visualizer none --deterministic
# no_music/music评价各自完整checkpoint，用--compare校验三组名义/初态/预算一致。
```

CPU验证命令：`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/weili/GENMO/.venv/bin/python -B -m pytest tests/closedloop/reference_adapter -q -p no:cacheprovider --basetemp=独立临时目录`。
本地与Server-1最终31项通过。GPU H16 microbatch1/2目标完全一致，全部1822734个参数
梯度最大差4.77e-7。8步music→Adapter数值梯度39.949802/FD39.941036。
20首名义缓存完整；12首训练/验证零策略全部music_end，共38492真实步。固定s0只由
8首训练歌30993步统计，最终标尺的真实479点执行qd音乐函数FD相对误差4.40e-10。
独立4首评价全歌完成，tracking(dt)0.0954~0.0973、F1=0.419~0.545；完整路径P50约22.8ms，
8028/8028控制步超过20ms，仍未达到实时50Hz墙钟能力。

2026-10-26实际正式启动门槛失败：Critic真实验证EV=-0.474264、MSE3.938287，差于常数
基线1.999986，且失败附近样本0。独立WM真实监督25次后，固定H1/H8/H16预测误差通过，
但真实方向16个case仅9个通过；存在明确反向、接触切换和重复性/噪声不确定。正式训练
未启动，accepted=0，没有第一有效更新。保留两份完整失败checkpoint及全部证据，恢复
不改变失败状态；不能从上述命令模板推断已启动。详见[Server-1实际验收报告](supermimic_adapter_acceptance_20261026.md)。
