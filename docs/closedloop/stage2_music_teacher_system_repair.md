# GENMO–SuperMimic 音乐教师系统修复与实证边界

本文件记录 `feature/stage2-supermimic-worldmodel` 的本轮工程实现、服务器2验证入口及真实证据。参数更新、模型损失下降、有限梯度和单元测试均不等于真实舞蹈质量改善。未经对应新目标的物理方向验收，不启动正式优化；已取得的负结果不删除，不重新启用旧实验的质量退化自动退出。

## 不变的模型与执行合同

- 原 Stage1 为 s595000；输出保持 60 帧、30 Hz、qpos30/contact2，原编解码和已知前缀语义保留。
- 当前 GENMO 完整 DDIM20、eta=0、CFG2.5；FutureStateEncoder 独立，不能把预测写成真实历史。
- 同一 BUMI SuperMimic checkpoint 同时提供冻结 pi/wm，参数及 normalizer 统计不变；SM 75 维观测、10 帧历史、570 维目标、21 维动作按关节名称映射。
- 真实 Newton 后端、真实接管 anchor、短时闭环 BPTT、在线异步连续覆盖及原配对数据 BC 保留。本轮不重训 Stage1/WM，不新增残差动作或 PPO 类目标。
- 源码版本用于追溯。资产、目标数据和实际数值合同需要匹配；源码哈希不是准入条件。

## P0：成本、教师与实际输出验收

`rewards.py` 独立输出 `action_rate_cost`、`joint_limit_cost`、21 维 action rate 的 sum/mean、各关节软/硬限位及距硬限位 0.05 rad 内频率。训练权重为 action rate=0、joint limit=2。没有同时改变动作平方和的归一化，冻结 SM 网络和执行器未修改。

`teacher_training.py` 使用以下独立项：

```text
L_actor = L_dance + 0.1 L_track + 2 L_local + 0.2 L_global
          + 1 L_BC + 0 L_action_rate + 2 L_joint_limit - 0 V_terminal
```

日志包括原始/加权损失、梯度范数、dance/local/global 两两余弦，及 Transformer 主干、历史前缀编码、FutureEncoder、输出头的分模块统计。实际候选参数位移与原始/加权梯度的内积单独记录，不能用 SGD 假想方向替代 AdamW 实际步。辅助投影默认关闭，硬限位梯度不被投影删除。

`output_trust_region.py` 在每轮开始保存 Actor 参数/buffer、AdamW 完整状态和 Python/NumPy/Torch/CUDA RNG。每个候选从同一状态出发，最多三次，学习率依次乘 1、0.25、0.0625。用同请求、预测条件、前缀和噪声重新生成，并重新计算同 anchor 的舞蹈 Loss；只有通过所有检查才保留更新。拒绝后恢复所有动量、step 和 RNG，accepted 计数不增加。

单次限制为关节坐标 RMS 0.010 rad、根位置坐标 RMS 0.005 m、SO(3) log 向量分量 RMS 0.01 rad。GENMO与Future基础学习率为2.5e-7。原0.015 rad和1e-6配置在跨歌曲真实执行中产生负结果，因此收至用户建议范围下沿，而非根据静止学生放宽。局部/固定教师累计偏差、活动范围、速度和相位限制均相对冻结教师。注意这些 RMS 是时间和坐标分量上的 RMS，不是三维向量模长 RMS。累计阈值仍需持续训练验证，不能把初步标定称为长期无坍缩证明。

`teacher.py` 保留独立原 Stage1 局部教师与旧独立运动学时间线 A；新增 `online_teacher.py` 构建原 Stage1+SM 真正在线消费参考 B 及真实状态 C。构建器核对原 Stage1 权重、FutureEncoder 零输出，并在 episode 结束后检查独立教师同噪声输出。保存每个控制时间戳、原始参考速度、完整请求/控制/噪声、实际交付事件及参考源版本。训练只读冻结文件，不用学生状态改写。

缓存身份包含音乐/初态内容、Stage1/统计/FK、SM/机器人资产、噪声规则、调度、真实延迟与接管事件、物理后端和目标定义。实际完成时刻与控制线程提交/拒绝时刻分别保存；尾部未交付请求属于删失事件，不猜测完成延迟。旧 A 与 B 的种子/请求身份不同，A/B 逐帧差异不能被直接宣称为相位错误。

`dance_losses.py` 的执行目标在真实状态 anchor 固定一次 SE(2) 变换，整段 H 不再对齐。参考保持及累计信任域则在**更新前已记录参考**的 anchor 固定一次变换，比较后续相对舞蹈；不能把机器人已经落后的世界位置当作生成参考退化，也不能用候选自己重新拟合变换。教师 detach，绝对根偏差另报。L_local 约束完整可修改局部参考；L_global 在实际 50 Hz 消费时刻比较原教师速度与同一 SM 参考构造出的候选轨迹，仅允许当前计划相关的可修改源点及速度 halo 获得梯度。每个控制步从实际混合版本缓存中只替换该请求仍有效的源点，保留后来的覆盖和既有保护点。没有把 50 Hz 教师反插值回 30 Hz 后做差分：该旧路径会制造不存在的速度/加速度差异。最终定义为 `actual_50hz_mixed_version_recorded_reference_anchor_relative.v6`。

## P1：方向与样本覆盖

`short_audit.py` 对同一真实接管状态测试 H16/H32，根 x/z/yaw、左右腿和上肢三档小扰动；分开报告 dance、track、action rate、joint limit、局部/固定 prior 和组合目标。保存真实 Newton 基线、重复、正负扰动状态与接触，检查恢复误差、首次预测超差、数值收敛和真实方向。阶段覆盖按各自 H 的真实窗口计算，H16 不借用 H32 才发生的落脚。

`tools/recheck_saved_music_objective.py` 只在冻结执行资产、调度、执行奖励和教师轨迹相同时，重新运行新目标的 WM/autograd 并读取保存的真实 Newton 扰动轨迹重算真实侧。显式 `--reuse-execution-derivatives` 模式仅在 action rate=0 时复用不变的执行分量导数，实际重新求导新参考正则，逐分量核对保存的真实状态重算结果；完整路径另做等价检查。报告明确标记复用范围，不是新模拟，不替代另行执行的真实参数候选对照。原始物理结果始终保留。

`sample_distribution.py`、`replay.py`、`formal.py` 分开记录 A 全部真实请求、B 完整/合法 H anchor、C 超域、D 实际抽取、E 实际接受。分组包含音乐来源、真实关节/根部/腿部速度、幅度、接触与动作类别；先检查完整 H 原支持域，再在合法池内分层无放回抽样。记录超域分量和首次预测时刻，不扩大任何原状态边界。A 的接触切换若尚未观测则记 unknown，不伪装成实测 0。

支持域检查专用路径保留相同 SM/WM 状态递推，只省略不用的奖励/FK；不能用于 Actor Loss。服务器2对 H16/H32 与完整路径逐张量检查最终状态差 0。

沿用原实验支持范围时，`state_support_reference_report` 指向原报告，新旧包络取交集并记录为 `state_support_contract.json`。新方向报告收集到了更多真实状态，不意味着自动扩大训练状态边界。固定数值容差不变。

## P2：独立 Critic 和物理模型选择

`music_mc.py` 对保存的完整真实控制轨迹重新累计新 MC。每个价值状态固定一次目标对齐并一直保持到真实终止，Critic 显式观察该目标上下文、有效参考和调度。真实失败是终点，不是完整音乐成功；行政截断不能当 MC。Critic 奖励不混入 BC 或参考正则。

新的导出 schema 为 `genmo.real_mc_critic.online_teacher_alignment.v3`，旧奖励 Critic 不兼容。拟合失败也保存质量曲线、诊断权重和可修改未来参考的直接输入梯度；未通过版本不能导出为 ready。新 Actor 仍固定 V=0，独立拟合通过也不自动开启非零终点价值。

`evaluation.py` 和 `dance_evaluation.py` 同时保存实际状态、实际消费参考、固定音乐教师、上/下肢幅度/速度/内容/相位、相对根运动、节拍、接触、完整时长/失败和模型物理差异。固定外生交付回放与真实墙钟异步分开报告；共同有效时段是额外诊断，缺失尾部仍计失败。相同 seed 在不同请求时间表下不能冒充相同噪声身份。

`formal.py` 的 best 以原 Stage1 真实音乐误差作为初始基准，必须改善音乐主目标且满足完成/时长/内容条件才替换。新配置每次分别运行固定事件和真实墙钟评价，以固定事件主对照选best，另存墙钟结果，不混合回报。正式训练采集器仍为真实墙钟异步，不被离线评价设置改写。跟踪分数不单独决定 best。评价负结果只记录，不启用自动质量退化退出。新的候选更新拒绝属于单步事务验收，与终止正式实验不同。

## 服务器2入口及证据

所有 GPU、Newton、模型和单元测试均在服务器2运行。本地仅编辑、阅读和 Git 检查。运行配置在仓库模板基础上显式填写原资产、数据和冻结在线教师 manifest，运行产物写独立目录。

```bash
# 在服务器2，使用已同步的 feature/stage2-supermimic-worldmodel。
export GENMO_SM_PYTHON=/data0/user/liwei/envs/genmo_sm_runtime/bin/python
export ISAACLAB_PATH=/data0/user/liwei/GENMO_assets/supermimic_bridge_20261024/IsaacLab
export PYTHONPATH=/home/user/liwei/bumi-closedloop-worktrees/supermimic/SuperMimic
export PYTHONDONTWRITEBYTECODE=1

# 参数 CONFIG / OUTPUT / CRITERIA 指向独立的实际运行配置、输出和预声明标准。
"$GENMO_SM_PYTHON" tools/build_stage1_online_teacher.py --config "$CONFIG" --output "$OUTPUT" --mode full
"$GENMO_SM_PYTHON" tools/audit_stage1_teacher_timeline.py --config "$CONFIG" --output "$OUTPUT" --device cpu
"$GENMO_SM_PYTHON" tools/audit_stage2_music_repair.py --config "$CONFIG" --output "$OUTPUT" --mode full
"$GENMO_SM_PYTHON" tools/audit_stage2_supermimic_teacher.py --config "$CONFIG" --output "$OUTPUT" --mode full
"$GENMO_SM_PYTHON" tools/audit_stage2_supermimic_short.py --config "$CONFIG" --output "$OUTPUT" --mode full --criteria "$CRITERIA"
"$GENMO_SM_PYTHON" tools/relabel_stage1_online_mc.py --config "$CONFIG" --output "$OUTPUT"
"$GENMO_SM_PYTHON" tools/fit_stage2_supermimic_mc.py --config "$CONFIG" --shards "$TRAIN_MC" "$VALIDATION_MC" --output "$OUTPUT"
"$GENMO_SM_PYTHON" tools/evaluate_stage2_supermimic.py --config "$CONFIG" --output "$OUTPUT" --mode full --actor-checkpoint "$INFERENCE_CHECKPOINT"
```

上述命令是各自入口示例，必须给每次调用不同输出目录。推理导出没有 AdamW/RNG，不可用于恢复正式训练。保持终点价值为 0；模板的 H16 是待验证候选，不能拿旧 H32 报告授权。

本轮真实证据根为服务器2：

`/data0/user/liwei/GENMO_outputs/stage2_supermimic_worldmodel/music_repair_system_20261026`

| 证据 | 实际结果与解释 |
|---|---|
| `old_formal_failed_snapshot.json` | 旧正式实验自行退出于 86 个有效更新，连续没有合法执行反馈；本轮没有删除或重启它。 |
| `online_manifest_extended.json`、`timeline_abc_v9/` | 24 首原 Stage1 在线轨迹，22 首完整；157/124 的真实参考耗尽失败保留，不冒充完整目标。 |
| `fixed_delivery_stage1_reproduction_v8.json` | 3 首固定交付回放全部完整，与原在线参考及 state55 最大逐张量差均为 0。 |
| `output_trust_calibration_v15.json` | v5定义16个独立候选，H16共20次尝试、5接受3拒绝；单候选中位1.16秒、一轮3.48秒。H32共21次尝试、4接受4拒绝，分别2.21秒/6.63秒。属于标定历史，非最终v6训练授权。 |
| `candidates_train_h16_v15/` | 四首训练音乐8候选：5接受3拒绝；接受候选4个真实短时误差改善、1个恶化约0.20%。不是连续训练。 |
| `teacher_gradient_h16_v15/teacher_gradients.json` | v5真实请求20步DDIM梯度通过；局部/固定音乐保护前缀梯度0，原Stage1同噪声最大误差4.411e-6。v6使用新报告，不沿用这个结论。 |
| `sampling_funnel_refined_v12.json` | 2761真实请求；H16合法1286/2719，H32合法1031/2701。H16合法高速872、接触切换539；未放宽支持域。 |
| `critic_fit_v9/` | 16首训练/8首验证、49206真实控制步重算新MC；125更独立验证MSE0.0002372311、EV0.127586，通过预声明拟合标准，Actor的V仍为0。 |
| `paired_evaluation_v17.json` | v5单候选三曲两种延迟对照全部完整；固定事件音乐误差均值0.0123973→0.0168094，F9u约翻倍，两种模式均不符合best条件。三曲此后用于尺度校准，不能继续称完全未使用的独立检验。 |
| `new_objective_physics_merged_v16.json` | v5共32个真实anchor、6个方向、3档尺度；H16通过。H32在Utb4的2352 tick软关节限位数值导数仍未收敛，不使用旧H32通过记录。v6参考目标改变后重新计算。 |
| `reference_relabel_full_equivalence_v15.json` | 一真实anchor两时域36条扰动，分量复用与完整WM重算的组合导数最大差4.08e-9，真实中心差分差1.12e-6，方向判定全相同。 |
| `teacher_metadata_v17/` | 最终构建实现独立真实采集gLO整曲成功，保存实际源点/混合版本及原Stage1同噪声检查；不改写24首冻结目标。 |
| `unit_v19.log` | 40 passed / 3 skipped；跳过的是要求显式外部资产环境变量的旧夹具入口，真实资产、DDIM与Newton另由上述独立审计实际执行。 |
| `new_objective_physics_merged_v19.json` | 最终v6重新核对同一32个真实anchor，H16完整通过、H32仍未通过数值导数；已去除跨H阶段覆盖误并集。 |
| `output_trust_calibration_v19.json` | 最终v6八个候选共十次尝试，八个接受；真实短时误差七降一升，接受最大关节单步0.009946 rad。单候选中位1.163秒，一轮中位1.168秒；三次尝试成本另见保留的前轮标定。 |
| `paired_evaluation_v19.json` | 最终v6单候选三首校准歌曲全部完成；固定事件音乐误差均值0.0123973→0.0165265，仍未改善，不选为best。真实墙钟结果另列，不混合平均。 |
| `parallel_music_teacher_v19/` | 两个独立Newton采集器实际运行，两轮均有采集时间重叠，各获得四个完整真实anchor；冻结资产不变，不冒充优化收益。 |
| `formal_verified_start_v19.json` | 新正式进程从s595000重新启动并完成一次有效更新；325个AdamW参数状态均含完整step/一二阶矩，Actor/Critic/RNG/replay/运行游标检查点完整。首微批两个来源、均为真实高速且包含接触切换。 |

最终v6组合方向、真实候选与启动状态追加到 `记录文本.md`，不以旧目标报告授权。原始Newton数据、完整请求、教师、候选与失败记录都保存在上述根目录；各临时源码包另在 `source_snapshots/` 留存用于复核，源码哈希只记录。新奖励长期训练不坍缩、真实音乐质量持续改善以及非零Critic价值消融，均需要独立持续实验证明。

本轮正式目录为服务器2 `/data0/user/liwei/GENMO_outputs/stage2_supermimic_worldmodel/music_teacher_system_formal_20261027/formal_20261027_012121_930751`。启动时实现提交213b8c7、入口修复5571bf2；GPU4训练、GPU7第二采集器、GPU5/6独立评价。沿用此前5000个有效更新目标及72小时资源预算；初始化不恢复旧退化权重。首批明确限制在已准备好的14首完整目标，未覆盖原4287首全部音乐，缺失原因见 `formal_initial_teacher_pool.json`。首次评价包含8首音乐，分别执行固定交付回放与真实墙钟；评价期间Actor计数停留在1属于正常等待，不能称整曲收益已经通过。
# 2026-10-27 正式八卡运行补充

八卡配置为 `configs/closedloop/stage2_supermimic_music_teacher_8gpu.yaml`，需与服务器已完成
物理验证、具有真实资产路径的音乐教师配置合并。GPU0同时采集一个Newton环境并执行
唯一Actor AdamW更新；GPU1至7各运行独立GENMO、冻结SM和Newton，返回同Actor版本
的真实anchor。训练微批8按样本累积梯度，这种流水线不是DDP。正式日志逐采集器保存
真实控制步数、anchor和所抽样的collector_id，不能仅凭8张卡占显存宣称完成训练。

评价时所有采集请求已经收束，8张卡复用为完整歌曲分片；固定延迟与墙钟异步仍分别
报告。首更先写完整检查点，整曲评价从第100次有效更新开始，此后每100次执行。
评价子进程异常保留失败记录并继续训练，评价负结果不退出、不覆盖原始证据。

八卡实验从原s595000重新初始化，保留旧双卡实验的第1更完整检查点和被迁移中断的
独立评价现场，不将中断评价记为质量失败或完整比较。H16、完整20步DDIM、CFG2.5、
原BC、冻结教师/pi/wm和现有候选输出信任域均不改变。候选不通过只回滚该步并继续
真实采集；连续无效更新只报告。新配置不设墙钟或控制步预算，正常完成目标为启动器
指定的5000次有效更新。实际程序/硬件故障仍记录异常，不能把损坏状态冒充正常训练。
