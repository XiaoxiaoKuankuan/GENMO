# 第8步：冻结 GENMO / GMT 的 Isaac 闭环基线

本入口连接当前两个工作树，在 Isaac Sim / PhysX 中进行真实动力学评估。GENMO、GMT、统计量及环境参数全程冻结，不构建训练 runner、optimizer、Critic 或 DPPO。独立评估目录位于 `outputs/closedloop_stage8/`；已有音乐训练入口和旧播放器继续使用原配置。

2026-09-29已完成10+100校准、48集真实双模型评估及独立时序审计。旧同步音乐视频因PhysX→USD姿态不同步已撤回视觉验收；修复记录位于 `outputs/closedloop_stage8/render_sync_fix_20260929/README.md`。**长Mine连续30s标准未通过，baseline_ready=false**：两模式各6/24失败，主要为世界yaw累积误差。代码与协议验证通过不代表跟踪质量验收通过。完整报告位于 `outputs/closedloop_stage8/implementation_20260929/README.md`。

## 固定接口与执行语义

GENMO 使用 `inputs/checkpoints/stage1_s350000_20260928/s350000.pt` 和同目录配套 stats、kinematics。GMT 使用另一工作树中的 `logs/rsl_rl/model_135000_stage2.onnx`、SHA 绑定兼容配置及当前 `bumi3_4340_lowpd` Isaac 资产。135000 的兼容配置具有明确来源边界，不能等同于拥有原训练全部配置。

`FrozenStage1Actor` 复用原 `Stage1Actor.sample()`：GRU 历史编码、MLP 前缀、qpos30/contact2、120点@30Hz、DDIM20、CFG2.5 均保持。只有音乐参加 CFG 丢弃。线上条件使用现有十字段 validator；不读取动作 target。

实际历史为 H=50 的 48D 原始物理观测，频率50Hz。关节字段按 GMT PhysX 名称顺序，归一化由原 Actor 的独立 proprio normalizer 完成。旧权威参考构造 physical qpos30 前缀，前缀与后续共用旧参考 anchor；网络内部仍 normalize 后重新应用 mask。不能通过实际机器人的漂移重新对齐参考来隐藏跟踪误差。

整个系统采用600Hz整数时钟：GENMO决策周期300 tick；GMT控制步12 tick；PhysX物理步3 tick。完整决策区间有25次GMT推理、100次物理推进。历史仅在实际控制步后追加，快照和额外观察没有推进副作用。失败控制步立即结束转移，保留终止状态，下一episode必须显式reset。

## 环境与参考保护

主配置为 `configs/closedloop/stage8_frozen_isaac.yaml`。Stage8环境在创建前覆盖为Z=0固定平地，关闭所有startup/reset/interval随机事件、扰动、观测噪声及课程，执行器实际lag为0。名义质量、COM、摩擦、PD、action offset、关节限位以及地面物理材质由Isaac读取并形成运行指纹；初始化、reset与退出核验。保留重力、碰撞与真实接触。

失败函数维持原含义及严格 `>`：根高度误差0.40m、相对参考根旋转误差1.20rad、脚/肘相对根高度误差0.30m。旧阈值0.20/0.60/0.15只记录首次越界和占比，不触发reset，也不能作为旧阈值运行的实际失败率。

后端独占权威30Hz源姿态和最终50Hz六数组。候选先prepare，再在控制边界commit；两次均检查episode、parent plan、请求和保护水位。保护包括已经承诺的内容、GMT已读命令窗口、预计到达后的0.20s前视以及插值/速度差分支持点。位置/关节容差1e-5、速度1e-4，四元数符号等价；最终保护区直接沿用原50Hz数值，同时验证源支持点独立重算一致。

P最少12，按延迟和支持范围延长，P>18单独报告。末端root XY位移没有下一点时保留逐坐标未知mask。120个30Hz点生成199个50Hz点，缺少右侧差分支持的尾点不能持续消费。参考耗尽明确结束，不重复末帧无限运行。

暂停等待模式在计算期间暂停物理。模拟部署延迟模式冻结请求快照，按实测生成、通信与转换耗时安排候选到达，等待期间继续消费旧参考；每环境至多一个请求，忙时漏掉决策而不积压。预算固定为10次预热后100次端到端测量P95+0.04s。它是本机共享GPU条件下的测量，不能直接声称满足其他部署硬件的实时性。

## 代码分工

| GENMO文件 | 职责 |
|---|---|
| `gem/runtime/closedloop_protocol.py` | 版本化元数据、NumPy数组及Unix socket RPC；仅标准库/NumPy |
| `gem/closedloop/frozen_actor.py` | 严格权重加载、冻结采样、确定性噪声、模型与buffer前后指纹 |
| `gem/closedloop/online_conditions.py` | 音乐对齐、实际历史、旧参考前缀、共同anchor及十字段验证 |
| `gem/closedloop/coordinator.py` | 校准、决策周期、延迟事件、候选提交、有限推进和终止 |
| `gem/closedloop/evaluation_music.py` | 四库固定val分组选样、音乐文件SHA、35维特征只读加载 |
| `gem/closedloop/baseline_metrics.py` | 实际控制轨迹、事件、计划、误差/幅度/音乐与缓存统计 |
| `gem/closedloop/baseline_provenance.py` | 明确源码/资产SHA（含untracked）与共享计算环境证据 |
| `gem/closedloop/baseline_video.py` | 按实际trace裁出episode、同步原val音频、ffprobe验证及中间片清理 |
| `tools/eval/run_closedloop_baseline.py` | 前置核验、启动和回收两个worker、有限评估及验收摘要 |
| `tools/eval/audit_closedloop_baseline.py` | 只读复核控制时间、消费plan、历史计数和保护区证据 |

GMT侧参考转换、无自动reset环境、在线命令和实际参数检查见另一仓库的 `docs/closedloop/stage8_frozen_isaac.md`。公共转换函数与原离线转换器共用；Isaac入口不导入MuJoCo。实际模块路径写入运行身份文件，避免editable安装引用旧工作树。

## 复现命令

仅检查当前输入和硬件，不加载模型：

```bash
cd /home/weili/bumi-closedloop-worktrees/GENMO
/home/weili/GENMO/.venv/bin/python -B tools/eval/run_closedloop_baseline.py --preflight
```

完整固定评估，自动使用全新时间戳目录，不创建后台任务：

```bash
cd /home/weili/bumi-closedloop-worktrees/GENMO
/home/weili/GENMO/.venv/bin/python -B tools/eval/run_closedloop_baseline.py \
  --config /home/weili/bumi-closedloop-worktrees/GENMO/configs/closedloop/stage8_frozen_isaac.yaml
```

默认四库每库两个独立val音乐组，各seed42/43/44、paused/latency两种模式，共48个episode，每段最多30s，短音乐自然结束。按组ID排序、组内最长样本、相同时sample ID排序。音乐来自原val manifest，音频/特征SHA必须匹配；没有用train音乐替代缺失val，也不声称该val对旧预训练模型未见。

有限联调可使用 `--datasets Mine --modes paused --max-episodes 1 --seconds 5 --calibration-warmup 1 --calibration-samples 2`，输出会标记小规模，不能据此称完整基线通过。`--video`支持逐episode导出配乐视频；正常计时矩阵不录像。`--calibration`显式复用校准，核对模型SHA、主机/GPU、设备/解释器/线程及相关时序，不静默修改旧校准身份。

## 证据和验收边界

每个实验保存resolved config、preflight、两个worker实际身份、calibration、逐控制步trace、生成计划NPZ、事件JSONL、分episode摘要与图、最终run_summary及worker_shutdown。程序退出0只说明有界程序正常结束；`run_summary.acceptance`分别报告冻结、保护区、完整48集、10+100校准及至少一条预选长Mine音乐连续30s多次重规划无reset。

跟踪误差和音乐指标仅用music阶段，warmup另列。音乐节拍指标复用现有实现：按实际50Hz时间对应EDGE35节拍，以真实动作和参考动作分别计算；缺节拍/时长不足返回null。幅度、速度、加速度、原阈值诊断和根位置漂移都应与终止率一起看，放宽阈值下“不失败”不等于贴合参考或音乐表现优秀。

保留本轮验收根目录：`outputs/closedloop_stage8/implementation_20260929/`。其中 `isaac_backend_acceptance` 为独立真实后端协议验证，`baseline_fourset` 为完整双模型矩阵。最终结果与同步视频在该目录验收报告中列明。所有测试worker及临时socket退出即清理，已有文本演示服务和模型/训练数据不修改。

本轮完整矩阵最初的advance事件proprio计数字段为null，原因是日志键名遗漏；原始证据不回填。独立审计通过真实初末计数、严格历史append和逐步时间轴核验，并明确这一限制。修正后的视频运行验证了逐advance历史计数。最后还修复了RPC丢回复后保留服务/advance缓存的重连行为，最终192项回归通过；没有为此改变生成或物理计算。

## 2026-09-29 渲染故障修复

旧CPU PhysX路径同时关闭Fabric与Kit默认的USD变换写回，导致真实物理状态运动、渲染网格停在初态，原跟随相机继续移动。旧 `implementation_20260929/mine_video/` 的媒体同步不代表画面正确，视觉验收已作废，原始轨迹和文件身份不回填。

GMT新增 `render_sync.py` 仅在需要渲染且无Fabric时通过PhysicsContext公共接口打开变换写回；每个录像控制帧按名称比较22个刚体PhysX/渲染位姿，固定容差1e-4m/1e-4rad，超差立即报错。相机改为世界固定，逐帧读回实际相机矩阵、记录RGB帧SHA。额外render/snapshot不推进物理或历史。

`physical_diagnostics.py` 按实际状态记录link姿态、隐式PD的computed/applied力矩估计、足部净接触力和URDF支撑球离地高度；净接触力包含全部碰撞对象，不冒充纯地面力。审计把warmup与music分开，物理、协议、视觉三类状态分别报告。渲染修复当时沿用了0.65m资产初态；下述1mm专项修正了该初始化错误，旧运行保留作为历史证据，不回填。

## 1毫米初态与20首专项验证

当前Stage8独立配置要求 `environment.initial_foot_clearance_m: 0.001`。GMT在应用135000模型默认关节后，按当前Isaac URDF的16个足底碰撞球和同一FK求根高，结果约0.475377408m。物理reset与启动参考共用这份qpos；原资产0.65m默认值、训练配置、GENMO高度基准与stats保持原样。Isaac构造时的仿真初始化之后先显式reset，再以严格容差读回根、关节和足底余量，不增加控制步或GENMO历史。

`stage8_grounded20_isaac.yaml`从既有val清单固定选择AIST++ 2组、AIOZ 8组、FineDance 5组、Mine 5组，按组ID排序，组内取最长样本。AIST++ val只有2个独立音乐组，不能以重复舞者/seed凑5首。每曲seed42、两模式，共40个episode，每段音乐最多30秒；短特征自然结束，失败立即停止。该专项与原48集/3seed矩阵分别标记。

1秒真实warmup及50步观测保留，首次落地响应不伪造或删除；专项报告分别列前5步、末10步、全部warmup与音乐首秒诊断。视频共享原片在全部逐集导出并核验帧区间、SHA和配音后才删除。每集 `isaac_music.mp4` 保存在对应episode目录，本地HTML索引统一浏览。

```bash
cd /home/weili/bumi-closedloop-worktrees/GENMO
/home/weili/GENMO/.venv/bin/python -B tools/eval/run_closedloop_baseline.py \
  --config configs/closedloop/stage8_grounded20_isaac.yaml --video --max-episodes 40
```

结果摘要同时报告 `requested_matrix_completed`（本次请求规模是否执行齐）与旧 `full_matrix`（原48集定义），进程退出0不等于动力学质量通过。


### 1毫米专项实际结果（2026-09-29）

20首不同val音乐×两模式共40集及40段配乐视频已完成，初态40次读回余量约0.999814mm。暂停/延迟均16首正常结束、4首姿态跟踪终止，失败率20%；BangBang两模式连续30秒，Talkdirty仍在25.72/25.76秒终止。54,946帧同步与固定相机、历史/时序/计划保护审计通过，2121次提交保护区修改0。两个模型冻结，运行源码91文件不变。此专项完成不代表原48集矩阵重跑或全部跟踪质量通过。

交付目录：`/home/weili/bumi-closedloop-worktrees/GENMO/outputs/closedloop_stage8/grounded1mm_music20_20260929/`。打开`README.md`看验收解释，`run/music_sweep_videos.html`看40段视频，`run/music_sweep_report.md`看逐曲指标，`failure_analysis.md`看失败轨迹分析。
