# GENMO / SuperMimic 十帧音乐参考 Adapter：实现与验收边界

本文件落实2026-10-26提供的 `Codex_Adapter10_Music_Task.txt` 和
`GENMO_SM_Adapter10_Method.md`。实现位于独立分支
`feature/stage2-supermimic-adapter`、工作树
`/home/weili/bumi-closedloop-worktrees/adapter/GENMO`。

**本轮交付是代码构建与CPU数值验证；没有启动真实物理训练、正式训练或三组真实评价。**
22项新增测试通过，包含真实SM权重/FK的8/16步计算图；这不是8/16步物理验收。
指定音乐公式存在保留的尖锐速度谷反例，音乐代理的普遍反站立性质未通过。

## 1. 工作树、资产与复用范围

- Adapter基线：`939d1d3`；父方案最新核对为 `cefabb2`。父分支后续涉及旧GENMO
  更新、并行采集、恢复与评价；本实现新建Trainer，没有整体合入这些旧优化逻辑。
- SM：`XiaoxiaoKuankuan/-supermimic` 的 `feature/genmo-worldmodel-bridge`，本轮fetch
  核对为 `2d1f44d`。源码只读，直接复用RobotSpec、Actor、MDP、中心差分/FK、
  WorldModel接口和StreamingPhysicalEnv，不复制PD配置。
- 本机实际读取的固定权重：
  `/home/weili/supermimic_vibed/logs/bumi3_seven_filtered_v1/ckpt_024000.pt`。
- 实际读取的机器人资产：
  `/home/weili/supermimic_vibed/data/bumi3_4340/robot`。
  SM源码相对目录没有生成资产，因此必须显式传 `robot_asset_dir`。
- 固定共享GENMO/GMT工作树分支与干净状态保持；SM桥接工作树保持干净。
- Server-1仅只读检查。原supervisor为2959295，torchrun为2959360，8个worker为
  2959366–2959373；本轮复查PID和启动时间未改变，latest.json仍更新。
  没有向其训练目录同步源码，也没有启动新远程任务。
- 源码提交只用于追溯。恢复比较张量/归一化/实际奖励与采样合同，不比较源码SHA。

## 2. 实际文件与职责

目录 `gem/closedloop/reference_adapter/`：

| 文件 | 职责 |
| --- | --- |
| `contracts.py` | 600Hz时钟、具名InputSpec、增强状态、已提交姿态与残差计划 |
| `features.py` | 可观测Adapter输入；特权状态、物理参数与调度上下文仅加入Critic |
| `policy.py` | 2654→512→512→256→270，三个SiLU隐藏层，末层零初始化 |
| `reference.py` | 名义只读、SO(3)组合、10帧与导数halo、原SM差分/FK/Goal570 |
| `music_reward.py` | EDGE35音频固定目标、执行速度连续代理、两步延迟与活动成本 |
| `rewards.py` | 直接调用原SM奖励；附加成本；六项原尺度与原身体集合诊断 |
| `dynamics.py` | 独立WM深拷贝、真实多步监督、固定统计与参数版本屏障 |
| `critic.py` | 新2835→512→256→1 Critic、SM lambda-return、真实回报校准 |
| `replay.py` | 真实连续转移、episode/song隔离、基线与近期数据混合 |
| `collector.py` | 真实/想象共用单步；原物理接口；分段采集不reset |
| `imagination.py` | 每步重运行Adapter/Actor/独立WM；16步BPTT与终端价值 |
| `trainer.py` | 三优化器、真实预热、分项梯度、先WM后Adapter/Critic |
| `checkpoint.py` | 全网络/优化器/RNG/回放/音乐/残差/采集游标；不覆盖保存 |
| `evaluation.py` | 三组真实配对、节拍/密度/F1/活动/jerk/脚滑/延迟 |
| `nominal.py` | 冻结GENMO实际采用计划的缓存准备和资产合同核对 |
| `rolling.py` | 原GENMO异步生成接入新Adapter；关闭额外WM预瞄 |
| `audit.py` | 实际权重数值检查，当前分布8/16步物理重复性/预测/方向检查 |
| `application.py`, `entry.py` | 显式配置、资产装配与独立CLI |

入口为 `tools/train_stage2_supermimic_adapter.py`、
`tools/evaluate_stage2_supermimic_adapter.py`、
`tools/audit_stage2_supermimic_adapter.py`、
`tools/prepare_stage2_supermimic_adapter_reference.py`。
配置为 `configs/closedloop/stage2_supermimic_adapter.yaml`；任务和验收阈值有独立example。
测试为 `tests/closedloop/reference_adapter/` 下的contracts/music/learning/boundaries及夹具。

## 3. 三类参考/状态与输入合同

始终区分：`nominal_reference`是冻结GENMO原始缓存，`corrected_reference`只给Actor，
`executed_state`是实际或WM预测执行状态。Adapter不改GENMO前缀，不回写名义缓存。

| InputSpec字段 | 每样本形状 | 来源与单位 |
| --- | --- | --- |
| control_history | 10×75 | 当时实际使用的SM历史，使用冻结Actor.obs_norm |
| nominal_goal | 570 | 原始名义参考Goal，使用冻结Actor.goal_norm |
| observable_error | 48 | 编码器q21、qd21及IMU相对姿态rot6d6；qd误差×0.1 |
| previous_plan | 10×27 | 上次相对当时名义参考的绝对残差，除固定幅值界 |
| previous_corrected_plan | 10×27 | 上次修正物理姿态相对当前名义参考，处理版本切换 |
| plan_times / plan_valid | 10 / 10 | 绝对tick派生的相对秒数与有效mask |
| committed_anchors | 2×27 | 左支持和当前已提交姿态相对当前名义参考 |
| music / music_valid / music_times | 16×35 / 16 / 16 | 当前及未来15个30Hz原EDGE帧、mask、采样相对秒数 |
| music_memory | 20×4 | 执行能量、执行活动、名义活动和有效mask |

合计D_in=2654，由具名字段计算而非在网络中硬写总维数。网络参数1,822,734个。
没有xi、世界根位置/速度或未来真实状态进入Adapter。关节/IMU误差是可观测量，
完整state55、xi25、上一动作、固定航向、音乐/剩余时钟、已知调度摘要以及当前世界名义参考57维只加入Critic。
Critic共2835维；部署不需要Critic或WM。

## 4. 十帧与导数halo

网络直接产生 `d[1:10]`，每帧3维根平移、3维旋转向量和21关节角。
位置界为XY各0.03m、Z0.02m；旋转向量范数≤0.05rad；按RobotSpec名称识别踝关节
0.05rad，其他关节0.10rad。末层全零；没有插值、广播或累计残差。

使用固定episode航向H：`p_c=p_0+H*d_p`、`R_c=R_0*Exp(d_theta)`、
`q_c=q_0+d_q`。关节按名称映射到SM顺序。当前姿态继承旧计划第一点，左支持保留
上一已提交当前姿态。新的未来十帧可全部重规划，GENMO未来保护规则不作用于Adapter。

右侧必须真实存在名义t+11点，残差仅作 `d11=2*d10-d9`；halo不进入Goal。
中心差分与FK直接调用SM桥接，当前+未来十帧交给原goal函数，结果仍是570维。
单帧57维包含相对根位置3、姿态6、局部根线/角速度各3、关节位置21、关节速度21；
原SM的qd×0.1输入缩放不改变。起点左支持不足时沿用原桥接单侧差分。

新计划前九点与旧计划后九点通过绝对tick匹配；名义版本变化时比较修正后物理姿态。
幅值、窗口相邻差分、重叠和参考软限位均是软成本。记录halo极值、目标速度极值和
残差饱和比例，没有额外夹断halo或暗改差分公式。

固定缓存到尾部不具备完整十帧+halo时明确 `reference_end`，不补轨迹；评价分别报告
`supported_reference_complete`和`full_music_complete`，不能把缺少尾部支持称为全曲完成。

## 5. 奖励、音乐时间和已知反例

直接调用 `runtime.reward(state_after, action, previous_action, nominal_next)`。
SM原六项权重1/1/1/1/0.5/0.5，尺度0.3/0.4/1.0/3.14/0.3/0.4；原身体集合、
根XY/yaw对齐、动作变化-0.1与实际关节软限位-2.0不变。返回值已乘0.02。

`r_total = r_SM_dt + 0.02*(0.2*R_music -0.02*C_magnitude -0.01*C_smooth`
` -0.02*C_overlap -0.1*C_reference_limit -0.05*C_activity)`。
跟踪、硬终止和软终止均相对名义目标，活动成本单独相加。

音频拍点来自EDGE35[:,34]；每音乐帧20tick、控制步12tick。目标为固定拍点高斯
包络，sigma=0.04s；mu由有效音频自身区间固定计算，不能训练时间偏移。
执行能量严格按任务书的平滑绝对速度公式，m=0.01、tau=0.05；稳定log使用
`eps+(1-2*eps)*p`。无beat只关闭音乐项，不关闭跟踪或其他成本。

推进到n支付n-2，用n-4/n-2/n的已发生执行速度。起点不足五点mask；每个中心只支付
一次，真实终止后不补分。活动成本使用尾随20个有效控制样本，低于名义平均速度0.7倍
才惩罚；名义静止不强迫运动。s0从零Adapter**真实训练集**基线速度RMS固定标定，
不对当前策略在线归一化。

实际合成信号结果（奖励越大越好，非物理结果）：

| 信号 | 平均R_music |
| --- | ---: |
| 温和同步速度谷 | -0.807899 |
| 四分之一拍错位 | -1.112878 |
| 半拍错位 | -1.013890 |
| 静止 | -0.953953 |
| 恒速 | -0.953953 |
| 随机高频 | -4.093998 |
| 尖锐同步速度谷 | **-1.652870** |

首轮测试由尖锐谷失败暴露该反例：其谷肩部概率与固定高斯目标不匹配，损失可比静止
更差。后续测试同时保存温和信号排序与尖锐反例，没有更改任务指定的公式或权重。
**22项代码回归通过不等于音乐代理普遍反站立通过**。原SM跟踪和活动成本仍独立存在，
但尚未通过真实物理证明它们足以消除投机；不可自行据此启动正式长训。

## 6. 真实监督、BPTT与完整恢复

- 真实物理使用单环境原StreamingPhysicalEnv；batch64是64个真实起点逐样本梯度累积，
  不宣称64个物理环境或墙钟实时50Hz。
- 独立WM从原bundle深拷贝，仅net参数进入WM Adam；in/out统计固定。真实连续8步
  窗口沿用WorldModel.state_error及原误差归一化，另报告targets单步误差。
  训练/验证按歌曲及音频内容隔离，混合零Adapter基线与近期真实数据。
- 先采集零Adapter完整真实episode预热新Critic，独立验证轨迹报告MSE、解释方差、
  常数基线和失败前16步误差。预热有限loss不等于价值已物理可信。
- 每轮先拟合WM；锁住model_version后，每个BPTT步重新运行Adapter/Actor/独立WM。
  只detach真实起点；状态、历史、上一计划、锚点、音乐记忆内部保持图。
- Adapter目标为累计软存活加权短时回报及终端Critic。真终止/音乐结束/固定缓存耗尽
  不bootstrap；行政/BPTT截断可bootstrap。Critic拟合更新前轨迹的stop-gradient
  SM lambda-return，gamma=.99、lambda=.95均按50Hz控制步。
- Adapter更新时WM和Critic参数冻结，输入导数保留；没有GENMO/Actor优化器、PPO、
  概率比、KL或GENMO BC。三个学习率为1e-4/1e-4/3e-5，裁剪1.0。
- 当前CLI在完整episode结束后更新和使用同一环境审计；`collect_steps_per_round`
  为每轮最少采集量，完整episode可能使实际步数超过该值，报告真实采集量。
  底层collector可以分段，分段本身不reset。
- 参数分项梯度报告来自本轮第一个合法真实起点的加权折扣目标，日志明确scope；
  另外报告所有起点平均原始分项与加权分量，不用正则梯度冒充执行反馈梯度。
- checkpoint保存三个网络/优化器、冻结bundle与normalizer、固定s0、RNG、回放采样
  RNG、音乐/残差/绝对时钟、完整在途快照、model_version、更新与采集计数/游标。
  编号保存独占，不覆盖旧文件。**未保存求解器内部状态，恢复必须开启新物理episode**。

滚动模式通过 `RollingCollector` 接入冻结GENMO的原异步worker和scheduler；forecast
显式关闭，原GENMO只读真实历史和原前缀。新Adapter仍每控制步工作。pending时不采样
新的想象起点，已有局部想象在下一预约/已知计划边界截断并bootstrap，绝不注入未来
真实接管计划。其结论是“给定当前名义计划的局部策略更新”，不宣称对后续GENMO
状态依赖决策求导。当前正式CLI优先固定缓存；滚动连接器的真实物理接管未实测。

## 7. 实际测试、命令与未运行项

本机命令（CPU，输出目录必须不存在）：

```bash
cd /home/weili/bumi-closedloop-worktrees/adapter/GENMO
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/weili/GENMO/.venv/bin/python -B \
  tools/audit_stage2_supermimic_adapter.py --self-test \
  --output /tmp/genmo_adapter10_20261026_validation/cli_final
```

退出0，**22 passed in 9.40s**。原SM真实权重8/16步分别实际展开8/16步；自动微分与
中心差分：music→Adapter 0.674706 / 0.675216，执行状态项→Adapter
0.398882 / 0.399120，music→第一步动作 0.473494 / 0.473782。
故意detach执行状态的负测可检测；冻结bundle逐张量不变。checkpoint恢复后下一次
Adapter/Critic/WM更新逐参数完全一致，但该优化器接口测试使用**明确的张量夹具**，
没有把模型转移当作实际物理训练证据。

本轮还检查29个新Python文件中文头部/AST、四入口help及git diff空白错误。原mb_control套件在补传本机SM固定资产后为18 passed、2 skipped，0.64秒；剩余两项需要完整Stage1配置或真实UMR动作，未计为通过。
日志：`/tmp/genmo_adapter10_20261026_validation/cli_final/pytest.log`和`result.json`。
初次失败日志为同目录上级 `pytest_first.log`，后续中间日志为 `pytest_second.log`。
pytest产生的临时模型目录精确清理，文本证据保留供交付核对。

验收状态：

| 项目 | 本轮结果 |
| --- | --- |
| 270→10×27→Goal570、SO(3)、halo/重叠/时钟 | 已运行，通过 |
| 零Adapter完整ReferenceBatch/速度/Goal/Actor/SM奖励 | 真实SM权重与FK，合成名义参考，通过 |
| 执行状态项、music-only state/action/Adapter梯度与有限差分 | 已运行，通过 |
| 音乐同步/错拍/静止/恒速/高频/无beat/短序列 | 温和信号通过；尖锐谷反例保留，普遍音乐质量未通过 |
| 全checkpoint与下一更新等价 | 张量接口测试通过；没有物理求解器续接主张 |
| 冻结GENMO真实名义缓存准备 | 入口已构建，本轮未运行 |
| 8/16步真实物理重复性/模型误差/有限扰动方向 | 未运行，不能复用旧H16结论 |
| 真实WM拟合、Critic预热、有界Adapter训练 | 未运行 |
| 三组真实评价/脚滑/真实延迟 | 未运行 |
| 滚动异步GENMO与Adapter真实并发 | 接口已构建，未运行 |

本机用于测试的Python没有IsaacLab模块；未使用其他容器或服务器补跑物理任务，遵守
本次“先代码构建、不影响Server-1训练”的范围。配置的tasks、music_baseline、GENMO
checkpoint及physical_criteria仍需显式配置真实资产，example不是已验证数据。

## 8. 后续运行手册（本轮没有执行）

以下命令只在**独立、空闲且已安装SM所需IsaacLab3/Newton/MuJoCo-Warp的环境**中执行。
先制作独立runtime YAML，重定位仓库/固定SM/GENMO权重/统计/FK路径；任务表每行有
id、song_id、train或validation、seed、原EDGE35、music_start_tick和显式state55初态。
初态的21关节使用RobotSpec顺序。不要在Server-1正在训练的工作树或输出根上运行。

阈值example只展示字段和候选值，默认配置没有启用它。应在实测前明确选择并记录
`physical_criteria`，不得根据失败结果调宽。物理方向使用Adapter末层bias的单位方向、
epsilon参数扰动，并在真实重放前缀0/50步后分别检查H8/H16；接触分支变化或差分
低于重复性噪声底记为不确定，不通过。

下面 `python` 表示上述物理环境的解释器；所有 `/ABS/...` 必须替换为实际独立路径。

```bash
# 1. 通过已有有效生成流程准备真实采用的冻结GENMO原始名义计划。
python -B tools/prepare_stage2_supermimic_adapter_reference.py \
  --config /ABS/adapter_runtime.yaml --tasks /ABS/raw_tasks.json \
  --torch-device cuda:0 --output /ABS/new_nominal_cache

# 2. 只采集零Adapter真实训练集速度，冻结s0；不运行优化。
python -B tools/train_stage2_supermimic_adapter.py \
  --config /ABS/adapter_runtime.yaml --tasks /ABS/new_nominal_cache/tasks.json \
  --calibrate-music --torch-device cuda:0 --output /ABS/new_music_calibration

# 3. 先执行当前分布的8/16步真实物理审计。
python -B tools/audit_stage2_supermimic_adapter.py \
  --config /ABS/adapter_runtime.yaml --tasks /ABS/new_nominal_cache/tasks.json \
  --music-baseline /ABS/new_music_calibration/music_baseline.pt \
  --physical --torch-device cuda:0 --output /ABS/new_physical_audit

# 4. 有界训练：真实预热、WM监督与更新前后物理审计，失败退出2并保留完整状态。
python -B tools/train_stage2_supermimic_adapter.py \
  --config /ABS/adapter_runtime.yaml --tasks /ABS/new_nominal_cache/tasks.json \
  --music-baseline /ABS/new_music_calibration/music_baseline.pt \
  --variant music --updates 1 --torch-device cuda:0 --output /ABS/new_bounded_train

# 5. 正式训练命令，仅在音乐代理及新分布物理验收完成后显式执行。
python -B tools/train_stage2_supermimic_adapter.py \
  --config /ABS/adapter_runtime.yaml --tasks /ABS/new_nominal_cache/tasks.json \
  --music-baseline /ABS/new_music_calibration/music_baseline.pt \
  --variant music --formal --updates 200 --torch-device cuda:0 --output /ABS/new_formal_music

# 无音乐消融使用相同任务/种子/真实采集预算/更新数，将variant改为no_music，另用新目录。

# 6. 完整恢复，--updates为追加更新次数；新物理episode，输出目录必须新建。
python -B tools/train_stage2_supermimic_adapter.py \
  --config /ABS/adapter_runtime.yaml --tasks /ABS/new_nominal_cache/tasks.json \
  --music-baseline /ABS/new_music_calibration/music_baseline.pt \
  --variant music --formal --updates 200 --torch-device cuda:0 \
  --resume /ABS/new_formal_music/update_000200.pt --output /ABS/new_formal_music_resume

# 7. 三组真实评价，使用完全相同的名义缓存和任务。
python -B tools/evaluate_stage2_supermimic_adapter.py \
  --config /ABS/adapter_runtime.yaml --tasks /ABS/new_nominal_cache/tasks.json \
  --music-baseline /ABS/new_music_calibration/music_baseline.pt \
  --variant zero --torch-device cuda:0 --output /ABS/eval_zero
python -B tools/evaluate_stage2_supermimic_adapter.py \
  --config /ABS/adapter_runtime.yaml --tasks /ABS/new_nominal_cache/tasks.json \
  --music-baseline /ABS/new_music_calibration/music_baseline.pt \
  --variant no_music --checkpoint /ABS/new_formal_no_music/update_000200.pt \
  --torch-device cuda:0 --output /ABS/eval_no_music
python -B tools/evaluate_stage2_supermimic_adapter.py \
  --config /ABS/adapter_runtime.yaml --tasks /ABS/new_nominal_cache/tasks.json \
  --music-baseline /ABS/new_music_calibration/music_baseline.pt \
  --variant music --checkpoint /ABS/new_formal_music/update_000200.pt \
  --torch-device cuda:0 --output /ABS/eval_music
python -B tools/evaluate_stage2_supermimic_adapter.py \
  --config /ABS/adapter_runtime.yaml --compare /ABS/eval_zero/evaluation.json \
  /ABS/eval_no_music/evaluation.json /ABS/eval_music/evaluation.json \
  --output /ABS/new_paired_comparison
```

三组比较检查实际缓存内容、初态、音乐/起点、种子、物理合同、噪声和s0；有/无音乐组
还检查实际更新次数及真实采集量，预算不同就拒绝配对结论。评价输出原SM六项、原始
根误差、实际完成率、离散节拍距离/双向F1/密度、幅度/活动、加速度/jerk、真实接触
脚滑、残差饱和与完整控制路径p50/p95/p99和超时数。固定缓存的计时不含并发GENMO，
报告明确该边界；滚动并发实时性须另外实测。
