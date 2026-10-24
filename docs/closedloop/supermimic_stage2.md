# GENMO × SuperMimic Stage2 实施说明

2026-10-24。**P0–P7 代码初版；本轮未运行测试、权重加载、模型推理、物理或训练。** 用户最后明确要求“先编写所有步骤代码、先不测试”，因此方案中要求实际通过的验收全部延期。这里列出的命令是后续运行说明，不是已执行记录。

## 工作树与资产

| 仓库 | 独立工作树 | 分支与基线 |
|---|---|---|
| GENMO | `/home/weili/bumi-closedloop-worktrees/supermimic/GENMO` | `feature/stage2-supermimic-worldmodel`，从当前 rolling `85d9cb1` 建立 |
| SuperMimic | `/home/weili/bumi-closedloop-worktrees/supermimic/SuperMimic` | `feature/genmo-worldmodel-bridge`，从 main `bf833cf` 建立 |

固定共享GENMO/GMT工作树没有切换；原SM main没有修改；没有操作正式训练进程、GPU、模型、数据或日志。新方法不导入旧PPO/DPPO/AWR/RoboFFT训练器，不增加动作残差，不生成latent。原Stage1仍是60帧@30Hz的qpos30/contact2及原编解码/前缀语义。

SM候选固定为 `/home/weili/supermimic_vibed/logs/bumi3_seven_filtered_v1/ckpt_024000.pt`。文件存在；内部张量和质量未加载核验。运行时只接收同一个文件的pi/wm，统计一起冻结。不会追随`last.pt`。详细静态证据见 [supermimic_asset_audit.md](supermimic_asset_audit.md)。

本机未找到`s595000.pt/ckpt`；配置的`stage1.checkpoint`保持null。已有stats文件存在，但其与未来提供的checkpoint配套关系仍待严格加载检查。SM的派生XML/USD由`robot_asset_dir`显式指向原资产目录，只读使用。Stage1原配对数据配置、实际音乐任务、延迟profile和P3报告均需提供，不生成假资产或重新建库。

## 分阶段实现与实际状态

| 阶段 | 实际修改 | 本轮执行/失败/未验证 |
|---|---|---|
| P0 | 本地与远端基线、工作树、资产、进程和导入合同审读；独立分支 | Git/文件/源码只读检查已执行；未找到本机Stage1模型；候选SM内容未核验 |
| P1 | SM `bridge/bundle.py`、`runtime.py`、导出和验证CLI；固定pi/wm/normalizers | 未加载/导出/验证模型；parity、bf16差异、输入导数、冻结不变性待测 |
| P2 | 共享原SM物理初始化/执行器步进；连续环境；动态Tensor参考、名称映射、600Hz调度 | 原环境回归、MotionLib parity、Stage1+SM真实闭环均未执行；不能称已跑通 |
| P3 | 旧参考闭环forecast、58维上下文、已知/生成参考预测与真实方向审计 | 未取得8/16/24/32步或多决策时域证据；当前没有可支持正式训练的WM结论 |
| P4 | 零输出未来编码、共享CFG、全20步DDIM、纯Tensor条件、非重入checkpoint | Stage1同noise一致性、输入泄漏、整链/有限差分、重算梯度待测 |
| P5 | 三决策事件想象、旧参考等待、覆盖、逐步折扣、lambda targets、近期真实回放 | 跨决策梯度、覆盖支持和变长时序对照待测 |
| P6 | 完整缓存感知Critic、真实异步采集、八项奖励、原监督正则、两optimizer、完整训练checkpoint | 预热/优化、隔离、冻结检查、保存恢复均未运行 |
| P7 | 短测/全曲评价/物理模型审计/纯导数入口、测试源、性能采样与本说明 | 没有测试通过或失败记录；所有运行验收未执行，未声称生成质量提升 |

未执行测试不是测试失败。当前已知缺项是资产/标定/验证证据未提供；隐藏的运行、数值和物理问题仍可能存在。

## 模块和梯度边界

`ForecastActor.stage1`持有原Stage1；独立`FutureStateEncoder`是58→128→原latent，末投影零初始化。58维为旧参考heading中的平移、相对旋转6D、线/角速度、21个q/qd误差和提前时间。相对帧时间也经零投影，invalid分支严格为零。音乐CFG只删除音乐，前瞻和原历史/前缀两支共享。

新`sampler.py`复用原1000步日程和DDIM公式，每步使用当前同一个GENMO，eta=0，无概率缓存、混合网络或冻结采样尾段。known逐坐标where后，decode、anchor、关节映射、30→50插值、差分与FK均保留图。原配对监督直接调用原training_forward和Stage1BumiLosses，完全不进入未来条件分支。

SM保存的state55、history10×75、goal570、action21、xi25以及WM101→54严格校验。`eval()`与`requires_grad_(False)`固定参数，`FrozenRunningNorm.update()`直接拒绝；act/step_model不加永久no_grad。xi在边界只normalize一次。世界线/角速度、wxyz，以及仿真xyzw转换沿用权威SM公式。

GENMO的原proprio48历史与qpos28的关节顺序不同。适配器按原Stage1历史名称排列，并使用q减active default；SM历史保留参考姿态误差6维。`ForecastContext`独立存在，真实历史从物理更新；想象历史明确标记来源。

首版真实物理为固定名义参数、无观测噪声、单机器人，使用SM相同IsaacLab3 / Newton / MuJoCo-Warp，50Hz控制、四物理子步、零执行器延迟、自碰撞关闭。上层可见完整仿真状态和xi，不是已实现硬件无特权部署。

缓存保存全部仍可能被读取的源点及左右halo。请求/预计到达/实际到达/提交/首次影响/中心消费分开记录。首次影响由当前+未来10帧、插值和速度差分的支持集计算；中心消费也考虑插值，不硬减0.2秒。超时侵犯保护支持即拒绝，连续超时或缓存不足明确终止；没有回写历史或平移计划。

真实采集worker计算生成时，主线程持续用旧参考推进物理。单卡资源竞争导致的墙钟超期会记录，不能将“异步代码”当作已证明实时。行政采集窗口保留episode；有界训练的优化阶段是离线仿真工作流，未提供与实机并行不停机学习服务。

多决策想象只detach初始真实快照，后续状态、前瞻、历史和缓存继续连图。每控制步r已乘0.02一次；区间保存实际n、Gamma和存活C。音乐结束与失败是终止，行政分段仍自举。Critic在无pending请求边界读取完整状态、两套历史、有效缓存/halo/contact/时间/版本、音乐、剩余时间、延迟与保护状态。Actor反传期间仅冻结Critic参数，输入导数保留；Critic targets在任何更新前固定detach。

八项奖励名称与选集显式迁移自原确认配置：22个身体、双足ankle_roll、12腿关节；尺度取rolling `reward_std`，根0.3m/0.4rad，权重归一化为(.5,.5,1,1,1,1,.8,.5)/6.3。SM六项单独报告。FK由当前生成参考计算，contact2不屏蔽代价。音乐代理关闭；根Huber候选公式已写但默认权重0，尚未启用验证。

只有GENMO和上层Critic两个optimizer。先固定初始GENMO收集当前任务预热，再启用终点价值梯度。checkpoint保存模型、两optimizer、RNG、确定性配对采样游标、外生延迟RNG、任务游标和紧凑真实回放；没有物理求解器状态，恢复明确新建episode。源码提交仅记录；冻结资产、几何/PD、数据和实际行为合同仍检查。

## 后续可复制运行流程（本轮没有执行）

以下命令在**同时具备GENMO依赖和当前SM IsaacLab3/Newton依赖的独立Python环境**运行。原训练容器没有被重启或改挂载；本轮也未安装环境。若使用容器，需先将两个独立工作树、实际Stage1/SM资产与原数据只读映射，并将配置路径改为容器内真实路径。不要在正在训练的共享目录切分支。

```bash
export GENMO_MB_REPO=/home/weili/bumi-closedloop-worktrees/supermimic/GENMO
export SM_BRIDGE_REPO=/home/weili/bumi-closedloop-worktrees/supermimic/SuperMimic
export PYTHONPATH="$GENMO_MB_REPO:$SM_BRIDGE_REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$GENMO_MB_REPO"
```

1. 在独立运行配置副本中填入实际Stage1 checkpoint、其配套stats/FK、原配对data_config和任务清单。任务结构见`configs/closedloop/supermimic_tasks.example.json`；music_features是现有30Hz EDGE35 `.npy [N,35]`，initial_state可为`.npy [55]`，未给时从名义站立开始。P3已知参考是GENMO原生关节序、world/wxyz、30Hz `.npy [N,28]`，不是新配对训练库。未见音乐/未见动作两个字段不确定时保持null。

2. P1显式导出和真实权重张量验证，输出目录均使用新路径：

```bash
python "$SM_BRIDGE_REPO/tools/export_genmo_bundle.py" \
  --checkpoint /home/weili/supermimic_vibed/logs/bumi3_seven_filtered_v1/ckpt_024000.pt \
  --robot-asset-dir /home/weili/supermimic_vibed/data/bumi3_4340/robot \
  --output /tmp/genmo_sm_bundle_024000.pt
python "$SM_BRIDGE_REPO/tools/validate_genmo_bridge.py" \
  --checkpoint /home/weili/supermimic_vibed/logs/bumi3_seven_filtered_v1/ckpt_024000.pt \
  --robot-asset-dir /home/weili/supermimic_vibed/data/bumi3_4340/robot \
  --device cuda:0 --bf16 --output /tmp/genmo_sm_bundle_validation.json
```

FP32容差预置atol=1e-6/rtol=1e-5；bf16差异单列，不以FP32逐位相同要求替代。导出的文件与原文件可任选其一固定整个run；更换文件后必须重新绑定固定资产。

3. P2先使用`--mode stage1`的真实baseline。初次没有延迟profile时在配置中显式填入候选`clock.bootstrap_latency_ticks`，或传同名CLI选项；这只是首次测量预算，不是已标定延迟，不默认旧路线0.26秒。

```bash
python tools/evaluate_stage2_supermimic.py --config /实际路径/stage2_local.yaml \
  --mode stage1 --max-seconds 5 --output /独立测试目录/stage1_sm_p2
```

输出`physical_results.jsonl`、`physical_summary.json`、紧凑真实轨迹和`deployment_latency.json`。短测因行政截断不计全曲完成。完整前瞻增加额外成本，正式使用前需以新完整部署路径重新标定，不能将无前瞻baseline延迟直接当完整配置标定。

首次完整路径标定使用`--mode calibration`：仍加载原Stage1和零输出未来分支，但实际执行WM前瞻，因此无需先有Stage2训练checkpoint。该模式不执行优化，报告明确标为calibration。其`deployment_latency.json`包含前瞻开关和固定SM资产，训练入口检查匹配。

```bash
python tools/evaluate_stage2_supermimic.py --config /实际路径/stage2_local.yaml \
  --mode calibration --max-seconds 5 --output /独立测试目录/full_path_latency
```

4. P3先审核WM的预测与真实方向。`--criteria`参考`supermimic_model_criteria.example.json`，事先填入实际验收阈值；未提供阈值仍能输出测量，但不会判通过。所有任务/种子从同一显式reset重新运行，并重复baseline验证复位可重复性。已知参考、Stage1生成参考、三个关节方向及优化候选分别保留证据。

```bash
python tools/audit_stage2_supermimic_model.py --config /实际路径/stage2_local.yaml \
  --mode full --criteria /实际路径/wm_criteria.json --output /独立测试目录/wm_physical_audit
python tools/audit_stage2_supermimic_gradients.py --config /实际路径/stage2_local.yaml \
  --finite-difference --output /独立测试目录/genmo_sm_gradient_audit
```

前一项是离线配对物理审计，生成计算可离线暂停，不作为实时证据；后一项只检查数值链路。接触力、执行器子步力矩和求解器状态不在state55中，报告不会伪造这些预测。模型失准时保留报告，不自动更新WM或扩大时域。

5. 根据通过的P3报告填`imagination.physical_validation_report`及`validated_model_steps`；入口要求同一冻结资产、已审计任务范围和实际时域。批次默认为真实采集1、Actor微批1；微批最多4采用逐样本梯度累积，保留全链。

```bash
python tools/train_stage2_supermimic.py --config /实际路径/stage2_local.yaml \
  --updates 4 --profile --output /独立测试目录/stage2_bptt_short
```

默认最多8次显式更新；前2次是候选Critic预热设置，不是经验证最佳值。输出training.jsonl分开标记模型目标/真实采集/原监督/梯度，编号checkpoint不覆盖。`--resume /实际路径/stage2_000004.pt`恢复训练状态，但创建新物理episode和新输出目录；恢复不了未保存的求解器隐状态。

6. 全曲固定评估去掉`--max-seconds`。`--mode full --actor-checkpoint ...`评估完整模型；`--mode no_forecast`必须加载对应无前瞻训练模型，不接受只在评估删条件的伪消融。增加`--profile`可记录CPU/GPU活动、阶段标签和峰值内存。报告分别记录失败/完成、漂移、动作幅度/速度、节拍、多样性、超时/耗尽、控制墙钟超期；模型回报不进入真实收益结论。

## 测试源与后续验收顺序

- SM `tests/bridge/`：真实checkpoint parity、normalizer冻结、输入导数、纯网络无Isaac/MuJoCo导入、小角旋转导数和非单位姿态坐标转换。
- GENMO `tests/closedloop/mb_control/`：依赖halo/中心消费、晚到/重复提交、两次覆盖梯度、变长折扣/终止、Critic状态缓存梯度、零投影学习顺序、实际动作MotionLib parity及真实权重整链。
- 实际资产测试通过环境变量`SM_BRIDGE_CHECKPOINT`、`SM_ROBOT_ASSET_DIR`、`SM_BRIDGE_MOTION_FILE`、`GENMO_SM_TEST_CONFIG`选择；缺少资产则skip，不生成假checkpoint。
- 按P1→P2→P3→P4→P5→P6→P7运行，先确认CPU/FP32数学，再单卡物理/反传。测试产物使用精确独立目录，完成后按仓库约定清理；正式模型和数据不清理。

本轮仅完成代码、中文日志和静态审读。实际训练收益、音乐质量、实时性、WM可用范围和真实机器人效果全部待验证。
