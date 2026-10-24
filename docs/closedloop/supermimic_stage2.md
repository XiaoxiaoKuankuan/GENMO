# GENMO × SuperMimic Stage2：服务器2真实验收与正式启动门禁

更新时间：2026-10-24。代码、资产装配、真实闭环、梯度与物理验收均已推进；**当前固定 `ckpt_024000.pt` 未通过正式配置需要的 WM 预测/方向门槛，正式训练没有启动，accepted Actor 更新为 0。**

正式启动脚本已经在服务器2实际执行，退出码为 2，最终唯一失败检查为 `world_model_physics_and_direction`。没有创建训练 tmux/PID，没有执行 Critic 预热或 GENMO 优化。不能把以下基线、数值梯度或模型内预测通过解释为 Stage2 训练收益。

## 工作目录、环境与固定资产

本地独立分支：

- GENMO：`/home/weili/bumi-closedloop-worktrees/supermimic/GENMO`，`feature/stage2-supermimic-worldmodel`。
- SuperMimic：`/home/weili/bumi-closedloop-worktrees/supermimic/SuperMimic`，`feature/genmo-worldmodel-bridge`。

所有测试在服务器2 `6000D-Server-2` / `ZP-NC579` 执行。本地只修改源码、文档和 Git；没有本地模型测试。没有切换固定共享工作树、修改原训练环境或停止原有正式任务。最初存在的旧固定评价自然结束后，才使用对应空闲 GPU。

服务器2：

| 项目 | 实际路径或身份 |
|---|---|
| GENMO | `/home/user/liwei/bumi-closedloop-worktrees/supermimic/GENMO` |
| SuperMimic | `/home/user/liwei/bumi-closedloop-worktrees/supermimic/SuperMimic` |
| 独立 Python | `/data0/user/liwei/envs/genmo_sm_runtime/bin/python` |
| IsaacLab 源码 | `/data0/user/liwei/GENMO_assets/supermimic_bridge_20261024/IsaacLab` |
| Stage1 | `/data0/user/liwei/closedloop_stage10_v3_assets/20261012/stage1_s595000/s595000.pt` |
| stats/FK | 与 Stage1 同目录的 `qpos30_train_stats.json`、`bumi_kinematics_robot_retargeter_fe934_v1.json` |
| 固定 SM pi/wm | `/data0/user/liwei/GENMO_assets/supermimic_bridge_20261024/ckpt_024000.pt` |
| 原机器人资产 | SuperMimic 下 `data/bumi3_4340/robot`，保留 XML 相对 mesh 路径 |
| 原配对库 | `/data0/user/liwei/datasets/bumi_music_umr70_mine_pass_90505_v1` |
| 验收根目录 | `/data0/user/liwei/GENMO_outputs/stage2_supermimic_worldmodel/acceptance_20261024` |

独立运行环境实际加载 Python 3.12.13、PyTorch 2.11.0+cu130、MuJoCo 3.15.0、Warp 1.17.0、Newton 1.8.0.dev0；IsaacLab 固定源码提交 `46d6cc3d7b115906fad12efd372fbf69f4e3d3ed`。原 GENMO 环境只读保留。依赖来源、失败安装和复核记录见验收目录及根目录 `记录文本.md`。

Stage1 SHA256：`51b522fbdeb0ec289e5ffc24e2fccca9888df90495f57f96714c1bb43c05f8e2`。

SM SHA256：`172fbc720583f415fdf27560c91bb8c6f6efe8165738d961cf2455694789cb54`。

选择 024000 的依据包含已有固定 SM 跟踪评价；该评价不能证明 WM 长时动力学或物理梯度可信。本轮在同一候选中固定 Actor、WM 和全部 normalizer，没有追随 `last.pt`。没有重新训练 Stage1 或重建音乐动作配对库。

## P0–P7 的实际状态

| 阶段 | 已修改及实际执行 | 失败或未验证边界 |
|---|---|---|
| P0 | 定位原权重、stats、FK、四库监督配置；装配 4287 训练任务、8 独立验证任务、24 固定评价任务；保留原 P6–30、15% 零前缀监督采样 | 正式配置未获准生效 |
| P1 | 真实 SM CPU 原网络/Bridge Actor 与 WM 最大差均 0；normalizer 固定，输入可求导；真实 MotionLib 对照通过 | 不表示模型动力学可靠 |
| P2 | 原 TrackingSim 与 Streaming 在 Newton 上 50 步对照通过；Stage1+SM 8/8 全曲完成，累计 364.54 秒物理时间、1066 次接管、11 次晚到拒绝 | 墙钟 522.19 秒，18227 个控制步超时；没有证明实时运行 |
| P3 | 5/10/15/20/32、候选45步及正式所需102步的三类 WM/物理对照；三关节方向、复位重复性、扰动幅度复查 | 102步已知参考完整验收1/8、方向5/8；**正式支持时域为0/未获准** |
| P4 | 原全20步DDIM、eta0、CFG2.5、独立ForecastContext、CFG拼批、解析参考缓存；GPU前向/反向/有限差分/主动断链全部执行 | 只证明数值执行链，不替代P3 |
| P5 | 三次实际生成、缓存覆盖、变长回报、第一参考到第二/第三决策梯度；累计主线与前瞻依赖分别记录 | 当前所需依赖长度超过固定WM可信验收范围 |
| P6 | 独立PlanAwareCritic、真实MC质量门槛、有效更新计数、两个优化器、v2完整checkpoint与恢复代码 | 真实Critic预热、有效优化、训练checkpoint保存后继续运行尚未执行 |
| P7 | 全曲基线、完整部署延迟、单/双GPU独立采集基准、CPU/GPU profiling；实际执行正式启动器 | 启动被P3阻断；没有Stage2/current配对收益、no_forecast训练或多GPU梯度同步证据 |

CPU 合同回归：17 项通过、1 项跳过。跳过的是需要显式 GPU 配置的整链 pytest；对应真实权重整链通过独立 GPU CLI 执行，不将 skip 当成 pass。

原物理/Bridge 最大状态差 `4.98295e-5`，动作差 `1.92225e-5`，奖励差 `1.26660e-7`，终止一致；FK 最大差约 `3.6e-7 m`，初态回读差 0。首次编译/图捕获单独预热并记录，未计为稳定运行延迟。

## 最关键的 WM 结论

完整前瞻标定采集80个请求，全部合法提交、没有晚到拒绝；延迟最小262、中位281、P90 298、最大342 tick（600Hz，约0.437/0.468/0.497/0.570秒）。使用保留的 bootstrap 360 tick 与历史延迟裕量36 tick，三决策主线90个50Hz控制步，最长累计模型依赖102步；三个前瞻分支各33步，依赖深度33/67/101。

P3用实测最大342 tick作为三次事件延迟，条件构造仍只读取过去延迟。为测量完整102步，在三次真实生成/接管后增加12步明确标注的配对诊断尾段。尾段不进入Actor目标、物理方向探针或正式BPTT，也不替代多决策闭环。

| 102步检查 | 实际结果 |
|---|---|
| 完整GENMO连续滚动状态预测 | 8/8通过 |
| 固定参考下SM闭环预测 | 8/8通过 |
| 固定真实动作序列的WM动力学预测 | 1/8通过 |
| 总回报方向、纯执行状态方向和优化候选联合门槛 | 5/8通过 |
| 同reset真实重放重复性 | 8/8通过 |
| 整体正式训练门禁 | 失败，`supported_control_steps=0` |

阈值在第一次P3前已声明，没有根据结果放宽。典型102步累计RMSE：

| 验证曲目 | 根XY误差/阈值 | 根姿态误差/阈值 |
|---|---|---|
| AIST `gPO_sBM_cAll_d11_mPO0_ch09` | 0.09410 / 0.05 m | 0.27492 / 0.10 rad |
| AIOZ `Utb4GaKWsiU_01_80_1500_dancer_00` | 0.14656 / 0.05 m | 0.68636 / 0.10 rad |
| FineDance `194` | 0.23048 / 0.05 m | 1.11162 / 0.10 rad |

5–32步已知参考预测通过，不等于这些短时域已经获得正式训练许可。候选45步的状态预测虽通过，方向仍有2/8失败；0.005、0.0025 rad复查显示方向对扰动尺度敏感。复查保留独立目录，没有用更小扰动结果覆盖原0.01 rad验收。

原Actor/WM与Bridge一致、真实物理步进一致、reset重复性通过，支持把当前主要阻塞定位为固定WM在所需时域/验证分布上的动力学与方向可信性不足。这里不声称其他checkpoint必然失败，也没有在GENMO Stage2中更新WM。

## 算法与关键实现边界

- 原60帧@30Hz qpos30/contact2、世界anchor、编解码、原历史及前缀保留。SM合同为state55、xi25、history10×75、goal570、action21，关节按名称双向映射。
- `ForecastContext`与真实历史分离。真实状态只在起点detach；多决策内部的状态、预测、缓存与输入导数保留。SM参数冻结不等于将act/step_model放进永久no_grad。
- 调度默认eager：提交或拒绝后在最早合法60tick公共网格请求，不额外等待0.5秒。fixed_interval仅作对照。预计延迟不读取当次尚未发生的实际延迟。
- 每个50Hz时刻只append一次75维观测；同tick更新参考只替换最后一槽。缓存包括左右速度halo，已消费支持点不能修改。
- 同一不可变参考的插值、速度、FK向量化缓存；merge、detach、源张量变化或从无图转有图时失效，trim保留正确缓存。CFG条件/无条件拼批仍使用同一个当前GENMO和原DDIM公式。
- GPU拼批与原Stage1同噪声输出最大差 `3.09944e-6`，小于既定`1e-5`；重算前向/梯度差0。纯执行状态损失到GENMO参数梯度范数约379.434，状态探针解析0.08957349、有限差分0.08952618；主动断链被识别。原监督loss约0.1329265、梯度约7.33530。这些不是训练收益。
- 八项奖励、0.02秒dt、变长折扣、终止和行政截断区分；原SM六项单独报告。音乐代理未启用，音乐表现靠原监督与独立节拍/幅度评价。
- accepted_actor_updates要求至少两次合法接管、无BC/无Critic状态链连通、BC之前模型目标梯度有限非零、实际参数变化。无效样本不做BC-only更新；Critic次数单独统计。参考变化使用同状态、同噪声比较未知关节RMS。
- 正式Critic预热按四库分层选择16训练episode，与8验证歌曲分离，使用真实完整episode MC。预声明至少100次拟合、MSE≤0.25、解释方差≥0.1、相对训练均值常数基线改进≥20%；当前未执行，不能填“合格”。
- 有界smoke也不会在拟合两次后自动解锁Actor；未加载真实质量通过状态时只检查Critic管线，Actor保持固定。
- 训练域用独立held-out来源覆盖、名义Newton物理、实测时域与真实状态支持包络核对。超域样本拒绝，不逐首限制到P3的8首音乐，也不取消OOD检查。
- checkpoint v2保存两个优化器、计数、模型、RNG、监督游标、延迟RNG、回放、评价状态。编号文件不覆盖，latest/best原子更新。恢复从新物理episode开始，不保存求解器隐状态。源码提交仅记录。

## 正式候选参数与实际启动结果

候选参数尚未生效：目标5000次accepted Actor更新；GENMO LR `1e-5`、FutureEncoder LR `1e-4`、Critic LR `1e-4`，BC权重1，clip1，控制折扣0.99；physical batch1、Actor微批1（支持1–4逐样本累积），三决策；最多500万采集控制步/72小时。

计划GPU0训练、GPU1独立评价；每100次accepted保存完整checkpoint并运行固定24任务真实评价，首次有效更新也保存。best依据物理结果，latest用于完整恢复。连续两次评价若完成率下降超过0.1、执行时长比低于0.9、幅度比低于0.8或节拍分数下降超过0.1，则保留证据暂停。没有把模型回报用于挑选best。

实际最终预飞行目录：

`/data0/user/liwei/GENMO_outputs/stage2_supermimic_worldmodel/formal_20261024_234204_310323`

其中`config.yaml`与`preflight.json`真实存在；唯一失败检查为`world_model_physics_and_direction`。GPU、磁盘、环境、分支、资产、完整基线、执行梯度和完整路径延迟检查通过。**没有训练tmux/PID、没有训练checkpoint、Critic更新0、accepted Actor更新0。**

首次启动尝试误写了基线报告路径，保留在`formal_20261024_234048_897679`；更正路径后重新执行了启动脚本，最终仅WM项失败。精确命令、退出码和输出保存在验收目录`formal_launch_attempt.json`。

## 在服务器2复现、查看与恢复

先设置独立环境，不能改用其他共享环境：

```bash
ssh -A 6000D-Server-2
cd /home/user/liwei/bumi-closedloop-worktrees/supermimic/GENMO
export GENMO_SM_PYTHON=/data0/user/liwei/envs/genmo_sm_runtime/bin/python
export ISAACLAB_PATH=/data0/user/liwei/GENMO_assets/supermimic_bridge_20261024/IsaacLab
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
GENMO_SM_AUDIT=/data0/user/liwei/GENMO_outputs/stage2_supermimic_worldmodel/acceptance_20261024
```

实际执行过的正式启动命令（当前报告仍会阻断，不能手工把validated字段改为102绕过）：

```bash
tools/launch_stage2_supermimic_formal.sh \
  --config "$GENMO_SM_AUDIT/assets/formal_candidate_blocked.yaml" \
  --output-root /data0/user/liwei/GENMO_outputs/stage2_supermimic_worldmodel \
  --train-tasks "$GENMO_SM_AUDIT/assets/tasks_train.json" \
  --validation-tasks "$GENMO_SM_AUDIT/assets/tasks_validation.json" \
  --evaluation-tasks "$GENMO_SM_AUDIT/assets/tasks_evaluation.json" \
  --gradient-report "$GENMO_SM_AUDIT/gradient_batched_cfg/gradient_audit.json" \
  --baseline-report "$GENMO_SM_AUDIT/stage1_full_songs/physical_summary.json" \
  --training-gpu 0 --evaluation-gpu 1 --max-actor-updates 5000
```

查看当前实际阻塞与原始结果：

```bash
cat /data0/user/liwei/GENMO_outputs/stage2_supermimic_worldmodel/formal_20261024_234204_310323/preflight.json
cat "$GENMO_SM_AUDIT/wm_physics_measured_worst/model_physics_audit.json"
cat "$GENMO_SM_AUDIT/formal_launch_attempt.json"
```

将来有通过验收的固定组合包后，启动器执行的真实入口是：

```bash
"$GENMO_SM_PYTHON" -u tools/train_stage2_supermimic.py \
  --config /实际新run/config.yaml --run-mode formal --mode full \
  --max-actor-updates 5000 --output /实际新run
```

基础设施中断后，在同一启动命令追加`--resume-run /实际已有run`；启动器使用原run配置、检查latest完整checkpoint及原PID已退出，再传`--resume-latest`。记录为质量/数值失败的run不会盲目恢复。该完整优化恢复路径尚未通过实际训练验证。

正式运行时查看`logs/train.log`、`training.jsonl`、`status.json`、`evaluations/paired_*.json`及`checkpoints/latest.pt`；TensorBoard目录为`tensorboard/`。停止时先核对`launch_*.json`或`training.lock`的PID与`/proc/<PID>/cmdline`属于该run，再发送SIGTERM；运行器在当前更新边界保存完整checkpoint后退出。**当前没有训练进程需要停止或恢复。**

## 性能与保留证据

单卡顺序与双GPU独立真实环境完成同一两任务集合，各12次接管；含启动墙钟55.297秒/27.335秒，约2.02倍。该基准证明独立采集并行，未证明多GPU训练吞吐。单次显式profiling的峰值allocated约1.012GB、reserved约1.051GB；完整trace保留，不将带profiler的延迟当部署标定。

以下产物均在验收根目录，按本任务保留证据要求留存；pytest临时目录已删除，禁用pytest缓存与Python字节码：

- `bridge_cpu.json`、`physics_original/`、`physics_streaming/`：SM与真实物理接口。
- `cpu_final_contracts.xml`、`gradient_batched_cfg/gradient_audit.json`：CPU合同与完整真实权重梯度。
- `stage1_full_songs/physical_summary.json`、`calibration_batched/deployment_latency.json`：全曲基线与完整路径延迟。
- `wm_physics_45/`、`wm_physics_measured_worst/`、`direction_eps005/`、`direction_eps0025/`：真实WM与方向原始证据。
- `benchmark_collectors/benchmark.json`、`profile_collector/`：独立多卡采集与CPU/GPU性能证据。

继续正式训练的前提是固定WM在所需累计时域及独立验证分布上通过预测与物理方向验收。当前没有放宽阈值、解冻WM或将模型奖励改善当作真实收益。
