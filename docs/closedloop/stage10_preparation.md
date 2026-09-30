# 第十步正式训练前准备：完整数据、连续更新、恢复与评估

本入口实现正式长训练需要的调度能力，默认配置仍限定在准备验收范围。它读取四库
全部train/val/test，不使用200首音乐选择文件。代码能力、文件审计、真实物理执行
覆盖和模型质量分别报告，不能把全量文件检查说成每条训练数据都已物理执行。

**当前奖励更新边界（2026-09-30）：** Stage9/Stage10 共用的 Tracking 已改成
`gmt.motion_tracking.v1`，参数和语义见[六项 Tracking 说明](stage9_dppo.md#tracking-与-frozen-gmt-当前任务对齐)。
本文下面的三轮恢复验收与全 val 数字来自旧五项 Tracking，作为历史证据保留。
`preparation_contract_v2_20260930` 也属于旧奖励身份，不能在当前代码下直接
`--resume latest` 或用 `--initialize-stage9` 绕过奖励身份检查。新版需建立独立 run、
重新采集执行奖励并建立初始评估基线；训练器、采样、Critic、DPPO 与 KL 门槛未更改。

## 入口与配置

- `tools/train_closedloop_stage10.py`支持`preflight/train/eval`；默认preflight。
- `configs/closedloop/stage10_prepare_server1.yaml`为独立配置，不继承旧selection。
- 当前清单声明train4287、val239、test239；运行时读取metadata并逐来源核对实际数。
- 训练从完整train池按任务概率20/35/25/20抽来源，源内打乱无放回。报告另外记录
  实际控制时间占比，不能将任务概率直接称为控制步比例。
- 随机起点以30Hz源帧计，音乐和配对活动度A_target同步切到真实末尾；10秒行政
  窗口仅截断采集，不把它冒充音乐终止。已承诺参考不为匹配离线前缀而截短。
- Actor/Critic仍只消费已验收条件；A_target及来源字段只进入奖励/审计。BC前缀
  改为P=0概率15%，其余6..30，覆盖当前约P21的实际条件。

## 连续训练及可恢复边界

每轮当前Actor采新数据→记录采集时Critic旧/下一价值→固定GAE/targets→Critic→
Actor候选→解析KL/GMT冻结/源码资产核验→清空Buffer→保存完整checkpoint→发布
latest。每条采集转移独立持久化，chunk及总manifest包含SHA，避免反复重写整批。

默认64条/轮、Critic80步/batch32/学习率1e-4、Actor一次累积更新；初始lr1e-9，
候选[5e-10,1e-9,2e-9]从同初态固定梯度独立比较，最大合格者保留。仍使用真实
joint_sum、std_floor=.001、eta=.1、CFG2.5、20去噪步、KL均值门槛.02、clip=.01、
BC权重.1。gamma=.99、lambda=.95及两网络梯度裁剪都显式接线，不仅写在YAML。

预算分开保存accepted_iterations/optimizer_attempts/generations/control_steps/
physics_steps。候选失败不退还尝试；不确定物理消耗不退款。checkpoint以逻辑已
发布轮次恢复，账本可能包含发布前失败消耗，绝不倒退。默认最多6次接受更新及18次
候选尝试，只是本次准备配置，不表示已启动正式长期训练。

`--resume latest`只接受本run最近已发布的完整状态；旧checkpoint不能在同run分叉
重写逻辑轮次。新worker session重新reset，旧Buffer/pending不复用。模型、优化器、
随机状态、采样游标、BC计步、实际lr和版本都恢复，随后继续采集和优化。

执行请求与显式噪声种子使用的decision/attempt/episode_count也全部保存；缺少decision
的旧Stage10 checkpoint明确拒绝，不默认为零。恢复身份显式绑定base seed、模型、
timing、runtime与diagnostics；实际输入配置和解析配置另外保存SHA。换一个YAML路径
不能绕过这些校验。预算及容量属于下面说明的操作配置，不混入模型算法身份。

首次保存`checkpoints/initial.pt`；若第一轮尚未发布成功，可显式恢复这个完整初始
边界并保留已消耗预算。`--initialize-stage9`是明确的Actor/Critic权重迁移，验证
资产、条件、奖励、随机核和物理契约；优化器/采样器/计步重新建立，不称full resume。

每个进程持有run写锁；SIGINT/SIGTERM请求在下一完整轮次边界结束。一般异常停止
session并保留证据，恢复最近完整checkpoint；不在部分更新的Critic/BC状态上继续。
各session/轮次目录不覆盖旧证据；运行时磁盘余量、配额和checkpoint预留均检查。
不自动删除原始执行数据或正式checkpoint。

轮前检查完整的下一轮接受更新及候选尝试额度，不够就以budget_exhausted在采集前
正常停止。采集过程若提前耗尽生成/控制/物理预算，保留已执行证据并明确失败退出，
恢复最近完整checkpoint；不会把未完整落盘的半轮称为成功。初始校准还没有形成
initial.pt时的故障需要新建运行目录，此时没有任何已接受训练更新丢失。

审计器从latest沿实际session的resume路径和SHA选择有效发布链。历史失败、缺尾行
日志及未被采用的publication保留并明确报告；故障后成功恢复可以验收，未恢复的
末次故障仍失败。硬中断前已发布、尚未写session摘要的轮次，从已fsync的session_start
和immutable publication核验，不依赖目录排序猜恢复关系，也不删除历史来换取通过。

## 正式首段预算、容量及受控扩展

`configs/closedloop/stage10_formal_server1.yaml`保持算法/奖励/采样/执行契约，仅将总
预算改为40接受轮、120候选、32000生成、1000000控制、4000000物理步。40是这个run
的总上限，不是每次resume额外增加40。run配额180GiB、文件系统至少30GiB空闲、
checkpoint暂存预留4GiB保持不变；该配置不等于已经授权或启动长训练。

容量规划工具只读已完成运行的实际文件大小，以每轮checkpoint和证据最大值、额外
两轮失败重试及25%余量估算，再独立检查run配额和文件系统空闲。已有目录的全部
字节都计入；抵扣已完成轮数时必须验证latest、publication、checkpoint大小和SHA，
只有accepted摘要却未发布的中断尝试不抵扣。它不删除文件，也不自动扩预算。

```bash
PYTHONDONTWRITEBYTECODE=1 /home/user/liwei/GENMO/.venv/bin/python -B \
  tools/eval/plan_closedloop_stage10_capacity.py \
  --reference-run /data0/user/liwei/GENMO_outputs/closedloop_stage10/preparation_20260930 \
  --target-run /data0/user/liwei/GENMO_outputs/closedloop_stage10/preparation_contract_v2_20260930 \
  --config configs/closedloop/stage10_formal_server1.yaml \
  --output /data0/user/liwei/GENMO_outputs/closedloop_stage10/formal_capacity_20260930.json
```

准备预算接正式预算使用同一个run的`--resume latest --extend-budget-reason <明确理由>`，
并传formal配置。入口先核完整checkpoint和模型/数据/执行身份，再写不可覆盖的扩展
事件，最后原子替换账本；原used和阶段计数不归零。事件绑定父checkpoint SHA、配置
SHA和原因，上限只能单调增加。若事件写成、账本替换失败，该孤立事件不授权扩限，
下次重试保留它并重新发布。后续恢复只传formal配置及`--resume latest`，无需重复扩限。

建议按10/20/30/40总轮次边界停止训练，分别运行独立全val评估并审阅失败、活动度、
奖励分项和KL后继续；test保留给最终报告。当前没有后台自动无限续训或自动删除证据。

新版正式运行的配对基线使用本run的`checkpoints/initial.pt`，后续与第10/20/30/40轮
使用同一版本、同一完整任务计划分别评估后比较。旧5f33e3e初始/第4轮的全val结果
保留为本次工程与算法路径验收证据；其训练身份不同，不能直接作为新版正式模型的
配对比较输入，比较工具会拒绝这种身份混用。新版三轮恢复测试不冒称又完成了全量
模型质量评估；控制管理修复的真实验收与旧四轮质量对照分别报告。

## 独立评估

eval必须显式提供完整Stage10 checkpoint。它不自动加载Stage1来冒充当前策略，也
不更新Actor/Critic/optimizer。完整val/test池按明确样本数和种子形成固定清单。
`eval_count: all`遍历该划分全部配对样本；同歌多舞者合法存在，另记独立audio/group
数量，不把239个配对样本说成239首独立歌曲。

默认val全239样本×seed42/1729，每样本以10秒为行政截断目标，在合法决策边界结束，
实际时长以真实执行步数为准；输出逐episode和按来源/种子
聚合的奖励分项、原始误差、活动度、失败/拒绝、延迟和前缀。有限时长行政截断明确
记录，不称全曲验收。评估恢复其自身随机状态，不推进训练采样器，也不进入训练Buffer。

## 服务器1有限检查命令

以下记录新版独立准备验收流程；已存在的成功目录不能作为新run重复创建。该配置
不关联旧200首清单。正式继续使用同run、同状态和显式预算，不重置账本。旧版
preparation_20260930的checkpoint缺少decision计数，只作历史训练与全val对照证据，
新版入口明确拒绝从该旧checkpoint续训；使用下面的preparation_contract_v2_20260930。

```bash
export LD_LIBRARY_PATH=/data0/user/liwei/closedloop_stage8_runtime/sysroot/usr/lib/x86_64-linux-gnu

CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/user/liwei/GENMO/.venv/bin/python -B tools/train_closedloop_stage10.py \
  --config configs/closedloop/stage10_prepare_server1.yaml --mode preflight \
  --output-dir /data0/user/liwei/GENMO_outputs/closedloop_stage10/full_data_preflight_20260930

CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/user/liwei/GENMO/.venv/bin/python -B tools/train_closedloop_stage10.py \
  --config configs/closedloop/stage10_prepare_server1.yaml --mode train \
  --output-dir /data0/user/liwei/GENMO_outputs/closedloop_stage10/preparation_contract_v2_20260930 \
  --stop-after-iteration 1

CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/user/liwei/GENMO/.venv/bin/python -B tools/train_closedloop_stage10.py \
  --config configs/closedloop/stage10_prepare_server1.yaml --mode train \
  --output-dir /data0/user/liwei/GENMO_outputs/closedloop_stage10/preparation_contract_v2_20260930 \
  --resume latest --stop-after-iteration 2

# 容量检查通过后，明确扩展总预算，但本次验收只到第3轮。
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/user/liwei/GENMO/.venv/bin/python -B tools/train_closedloop_stage10.py \
  --config configs/closedloop/stage10_formal_server1.yaml --mode train \
  --output-dir /data0/user/liwei/GENMO_outputs/closedloop_stage10/preparation_contract_v2_20260930 \
  --resume latest --extend-budget-reason '正式训练前验证同run预算扩展；本次只执行至第3轮' \
  --stop-after-iteration 3
```

评估命令通过`--mode eval --checkpoint <latest.json指向的完整文件> --output-dir <新目录>`
执行；默认全val，`--eval-split test`选择独立test。相同checkpoint/清单/种子可重复
对照，但实际latency执行仍受测得推理/I/O耗时影响，不宣称位级相同物理轨迹。

## 完整评估分片与配对比较

`tools/eval/run_closedloop_stage10_shard.py`显式加载完整checkpoint，使用同一父计划
按样本索引mod N分片，同样本全部seed保留。每片使用独立GPU、运行目录和执行
journal；新增工具实现单独保存SHA，原训练身份仍严格校验。通过每片原审计后，
`python -B -m gem.closedloop.dppo.evaluation_shards`核完整父任务集合恰出现一次，
重算整体/来源/seed统计；预算总和仍须符合一次完整评估上限。

`tools/eval/compare_closedloop_stage10.py --initial-report <初始merged/report.json>
--final-report <更新后merged/report.json> --output <新文件>`执行CPU只读配对比较。
两个checkpoint必须不同，其余父任务、噪声、时长、训练和评估身份必须一致；同时
报告任务总回报、实际时长、失败，以及实际控制区间均值，防止失败后少执行造成
表面奖励率上涨。四轮比较只验证小步变化，不能宣称收敛。latency执行受实际耗时
影响，相同任务/噪声不保证跨负载逐位相同轨迹。

## 当前验收状态

服务器已完成全4765条内容与身份审计，版本5f33e3e的四轮真实训练、2→4恢复后更新、
19项独立训练审计和物理journal审计通过；同版本初始/第4轮完整val各478任务、全集
合并、独立审计和配对比较均passed。平均任务回报39.25925125→39.25328172，整体
基本持平（约-0.0152%），每组物理/基础设施失败均0；不能称为学习质量提高。
最终代码2ffb58d完整GENMO CPU回归546项通过，GMT相关38项通过。全val结束、进程
退出后才pull新版服务器源码，并完成新目录三轮恢复与预算扩展验收：三个session均
passed且exit0，第1轮→恢复第2轮→显式扩限恢复第3轮，全部正常关闭GMT。
旧报告标注的完整恢复不包含后来发现遗漏的decision计数，不能把旧标记当作无遗漏
证明。新版独立训练审计19/19通过，execution_counter_contract=verified；旧四轮与其
全量对照继续保留原版本身份，不冒称是新版第3轮模型的质量结果。

新版每轮64条，共192条转移、4785训练控制步；含校准/预热共5435控制/21740物理，
与独立journal审计相符。三轮均选2e-9，mean joint KL为0.00834182/0.00380293/
0.00103455。3 worker的实际观测历史、trace/ACK/物理计数通过；legacy cache0，
窗口最多source129/reference214，208次prepare均9.443ms、P95 12.904ms。
正式容量扩限前预估148.58GiB，第3轮后复核147.89GiB，均低于180GiB配额并保留
文件系统30GiB余量；实际写入继续受磁盘守卫控制。奖励v2、GMT参数与归一化未改。

完整证据在服务器`/data0/user/liwei/GENMO_outputs/closedloop_stage10/`，本地详细说明
和小报告副本在`/home/weili/bumi-closedloop-worktrees/stage10_preparation_results_20260930/`。
本轮完成的是正式训练前工程准备及有限验收，没有启动长期正式训练，也未验证收敛、
长曲播放、域随机化、多环境训练或真实硬件。

## 历史旧 Tracking 续训命令（当前版本不适用）

下面命令记录的是旧 Tracking 验收后给出的续训方案，未执行。该 run 在旧奖励身份下
结束于第3轮并发布了40轮预算扩展；当前奖励已变化，恢复检查会拒绝它，不能继续
执行这条命令作为新 Tracking 正式训练。保留命令是为了历史可追溯，不放宽身份校验。

```bash
cd /home/user/liwei/GENMO-bumi-closedloop
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  LD_LIBRARY_PATH=/data0/user/liwei/closedloop_stage8_runtime/sysroot/usr/lib/x86_64-linux-gnu \
  /home/user/liwei/GENMO/.venv/bin/python -B tools/train_closedloop_stage10.py \
  --config configs/closedloop/stage10_formal_server1.yaml --mode train \
  --output-dir /data0/user/liwei/GENMO_outputs/closedloop_stage10/preparation_contract_v2_20260930 \
  --resume latest --stop-after-iteration 10
```

该命令在原奖励及代码身份下的语义是完整恢复第3轮；当前版本须使用新的独立运行
目录，沿用 Stage1 Actor weights-only 初始化，再重新建立 Critic、优化器及新基线。
本次 Tracking 修改没有启动任何正式训练，也没有将旧全 val 报告当作新奖励验收。

## 八卡共同更新同一模型的启动入口

新增 `tools/train_closedloop_stage10_8gpu.py` 和
`configs/closedloop/stage10_8gpu_server1.yaml`。八个 GPU 进程共同训练同一个 Actor
与同一个 Critic：rank0 用一个冻结 GMT／CPU PhysX 环境采集本轮完整执行转移，
将本轮数据发给其他 rank，八卡分担更新计算并同步梯度，rank0 统一发布 checkpoint
和运行账本。它不是八个独立训练，也不是八个并行 GMT 环境；采集阶段仍是单环境。

本入口继续读取完整 train 数据池，与旧200首清单无关。全局每轮仍为64条上层转移，
不会变成每卡64条；每条轨迹的奖励、GAE、旧采样概率及去噪自由动作掩码语义保持原样。
新配置逐项保留正式单卡配置的六项 GMT Tracking、Critic／DPPO／BC 参数、
Actor 学习率候选 `5e-10 / 1e-9 / 2e-9` 和联合 KL 门槛 `0.02`，仅新增：

```yaml
stage10:
  distributed:
    world_size: 8
    backend: nccl
    collection: rank0
```

启动脚本默认使用服务器1的GPU `0,1,2,3,4,5,6,7`，通过单节点 `torchrun` 启动8个
进程。必须指定新的运行目录；默认只运行到第1个接受更新，以便先核验启动与共同更新：

```bash
cd /home/user/liwei/GENMO-bumi-closedloop
bash scripts/train_stage10_8gpu_server1.sh \
  --output-dir /data0/user/liwei/GENMO_outputs/closedloop_stage10/gmt_tracking_8gpu_new_run \
  --stop-after-iteration 1
```

需要连续执行首段时，在**新的独立运行目录**显式选择40轮，并在 `tmux` 中运行：

```bash
cd /home/user/liwei/GENMO-bumi-closedloop
bash scripts/train_stage10_8gpu_server1.sh \
  --output-dir /data0/user/liwei/GENMO_outputs/closedloop_stage10/gmt_tracking_8gpu_first40 \
  --stop-after-iteration 40
```

启动脚本不自动删除输出目录，也不覆盖已有run。启动验收产生的临时数据、日志和
checkpoint须按仓库测试产物约定，在核验并记录结果后精确清理；正式训练产物保留。

首版八卡启动验收只支持新的run，未实现多卡完整恢复。当时的模型从配置指定的
Stage1 Actor checkpoint 按 weights-only 初始化，Critic、优化器和运行账本重新建立。
下节七天运行版本增加了同一训练身份下的显式恢复；旧 Tracking run、单卡 run 和
旧版源码身份仍不得静默混入。跨 Stage9 初始化不在此八卡入口的支持范围内。

八卡通信、一次共同更新与长时间稳定／恢复是不同的验收范围；启动成功不代表训练
已经收敛，也不代表吞吐会获得8倍加速。应以本次运行的真实共同更新、参数一致性和
冻结 GMT 检查记录判断启动是否通过，不以GPU显存占用代替训练正确性的证据。

2026-09-30 服务器1实测 `dfb811d` 已通过一轮八卡同步训练：NCCL all-reduce=36，
8个rank初始化、Critic更新、Actor更新及最终KL检查后的模型指纹均完全一致。
完整数据目录审计通过，本轮从完整train池抽取4个任务并收集全局64条真实转移，
1600奖励控制步；含校准/预热总计1850控制步、7400物理步。零更新概率差与ratio-1
均为0；Critic更新80步，批内MSE322.498→60.883；Actor三个学习率候选均通过，
最终选2e-9，联合KL0.0105090024低于0.02。Actor/Critic均真实改变，GMT模型与
运行参数/归一化指纹不变。唯一主进程成功保存并发布checkpoint，全部rank与GMT
正常退出，作业exit0。验收临时run随后精确清理；摘要写入根日志。

这次历史验收证明八卡共同更新同一个模型已可启动，当时未执行40轮长训练、完整
多卡resume或新模型全val质量评估。40轮命令是显式新run训练命令，不能用于接续
已经清理的1轮临时验收。

## 七天正式八卡训练与显式恢复

`configs/closedloop/stage10_8gpu_server1_7day.yaml` 保持八卡同步学习、rank0单GMT
采集结构，训练仍使用完整train池，全局64条转移／轮。六项GMT Tracking、音乐奖励、
Activity Gate、Stable／Physics、Critic、DPPO、BC、Actor学习率候选和联合KL
门槛均与八卡启动配置相同。本配置只扩展运行时间、计算预算与存储管理：

| 参数 | 七天配置 | 含义 |
|---|---:|---|
| `run_control.max_walltime_seconds` | 604800 | 从`run_manifest.created_at`起的七天绝对截止，恢复不延长；到期在完整轮次边界受控停止 |
| `limits.accepted_iterations` | 10000 | 接受更新轮数的独立硬上限，并非预计七天完成量 |
| `limits.optimizer_attempts` | 30000 | 每轮最多三个学习率候选的总尝试预算 |
| `limits.generations` | 1000000 | 真实生成总预算，包含校准等额外生成 |
| `limits.control_steps` | 25000000 | 50Hz实际控制步预算，包含校准、预热和训练执行 |
| `limits.physics_steps` | 100000000 | 200Hz物理步预算，保持控制步预算的4倍 |
| `storage.max_run_bytes` | 2 TiB | 本run最大存储配额，正式输出放在服务器1的`/data1` |
| `storage.min_free_bytes` | 100 GiB | 文件系统最低剩余空间 |
| `storage.checkpoint_keep_last` | 2 | 保留最近两份已发布checkpoint |
| `storage.checkpoint_keep_every` | 100 | 额外保留每100轮的里程碑checkpoint |
| `storage.archive_completed_iterations` | true | 对已完成轮次的完整执行证据归档保存 |

新run按配置中的Stage1 checkpoint进行Actor weights-only初始化；新建Critic、
优化器、采样器与预算账本。下面命令会持续训练到七天时间上限或其他预算／保护条件
先到达，不会因为原启动脚本默认1轮而提前结束。应在 `tmux` 等持久会话中运行：

```bash
cd /home/user/liwei/GENMO-bumi-closedloop
STAGE10_8GPU_CONFIG=configs/closedloop/stage10_8gpu_server1_7day.yaml \
  bash scripts/train_stage10_8gpu_server1.sh \
  --output-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/gmt_tracking_8gpu_7day \
  --stop-after-iteration 10000
```

中断后的恢复使用**同一run目录、同一七天配置及明确的`--resume latest`**。恢复
Actor／Critic、优化器和运行状态，实际执行消耗不会清零；它不是重新从Stage1权重
开始，也不能把旧奖励版本或不同训练身份的checkpoint当作兼容来源。七天截止时间
以首次创建run为准，包含作业中断时间，执行恢复命令不会重新获得七天额度：

```bash
cd /home/user/liwei/GENMO-bumi-closedloop
STAGE10_8GPU_CONFIG=configs/closedloop/stage10_8gpu_server1_7day.yaml \
  bash scripts/train_stage10_8gpu_server1.sh \
  --output-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/gmt_tracking_8gpu_7day \
  --stop-after-iteration 10000 \
  --resume latest
```

存储管理继续保留`initial.pt`，以及最近两份和每100轮的里程碑checkpoint。只有
已经满足保留策略、且新的checkpoint已经成功发布后，旧的非里程碑checkpoint才
回收。每轮`summary.json`和`lr_progress.json`仍可直接读取；rollout、raw_samples、
SQLite和fixed targets等完整执行证据采用无损`tar.gz`归档，`archive_manifest`
记录逐文件SHA和归档SHA。只有验证归档完整并发布清单后才回收对应的原文件，
不能把它理解为丢弃执行轨迹或只留奖励摘要。归档和checkpoint回收不会重置训练
预算；七天、10000轮及其他计算／磁盘保护条件择先停止。2TiB配额或100GiB剩余
空间保护先触发时不能为了凑满七天静默删除正式执行证据。

七天是运行时长目标，不是收敛保证。rank0单环境采集阶段可能使其他GPU等待，八卡
共同更新也不等于端到端8倍吞吐。训练质量仍需后续使用独立val评估；正式启动的
实际run路径、代码版本、启动时间及检查结果应另行记录，不能把本节命令当作已启动
七天作业的证据。

### 2026-09-30 七天正式八卡作业已启动

用户授权七天正式训练后，服务器1于北京时间2026-09-30 21:04:42启动新run，
执行代码GENMO `6374f40` / GMT `b17ab9c`。run首次创建于21:04:53，
绝对截止为2026-10-07 21:04:53；到期在完整轮次保存边界停止。

- tmux：`stage10_7d_20260930_210442`。
- 正式run：`/data1/user/liwei/GENMO_outputs/closedloop_stage10/formal_8gpu_7day_20260930_210442`。
- 持久日志：`/data1/user/liwei/GENMO_outputs/closedloop_stage10/jobs/formal_8gpu_7day_20260930_210442/training.log`。
- 外层启动记录、脚本和最终退出状态位于同一jobs目录；断开SSH不影响运行。
- 新run Actor从Stage1 `s350000.pt`权重初始化，Critic/两个优化器新建；先完成正式前2轮，
  再从该run第2轮完整checkpoint恢复至第3轮，沿用原采样器、RNG、预算和截止时间。

截至21:17:55，第3轮已经接受、保存并归档，后台继续第4轮。全量4765条
配对数据审计passed，实际训练使用4287条完整train池。前3轮各64转移，
Actor学习率均选2e-9，joint KL依次0.008900037656、0.004438272735、0.001220771161，
均满足原0.02上限。更新前ratio-1最大误差均0，各更新阶段八rank Actor/Critic
参数及buffer一致，两个网络更新彼此隔离，GMT参数/运行统计未变。

首2轮执行归档分别163507881/163418935字节，第3轮164120822字节；
第三轮已触发有记录的第1轮旧checkpoint回收，initial及第2/3轮完整状态保留。
前2轮独立CPU只读审计报告为jobs目录`first_two_readonly_audit.json`，
对完整执行转移、去噪链、GAE、checkpoint字节和计数核验passed；临时解包已清理。
完整八卡恢复与恢复后真实更新已实际验证，尚未声称七天完成或模型质量提升。

第二轮归档间隔约216秒，按早期速度粗算七天约2800更新；10000只是高预算上限。
实际完成量受采样/优化/归档速度和故障影响，以持续metrics和latest为准。

查看当前日志，无需重复启动：

```bash
ssh 6000D-Server-1
tail -f /data1/user/liwei/GENMO_outputs/closedloop_stage10/jobs/formal_8gpu_7day_20260930_210442/training.log
```

`latest.json`给出最新完整checkpoint；`metrics/*.jsonl`按session记录每轮更新、
归档、回收和恢复，`long_run_policy.json`记录不可在恢复时重置的截止时间。
