# 第十步正式训练前准备：完整数据、连续更新、恢复与评估

本入口实现正式长训练需要的调度能力，默认配置仍限定在准备验收范围。它读取四库
全部train/val/test，不使用200首音乐选择文件。代码能力、文件审计、真实物理执行
覆盖和模型质量分别报告，不能把全量文件检查说成每条训练数据都已物理执行。

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

## 独立评估

eval必须显式提供完整Stage10 checkpoint。它不自动加载Stage1来冒充当前策略，也
不更新Actor/Critic/optimizer。完整val/test池按明确样本数和种子形成固定清单。
`eval_count: all`遍历该划分全部配对样本；同歌多舞者合法存在，另记独立audio/group
数量，不把239个配对样本说成239首独立歌曲。

默认val全239样本×seed42/1729，每样本至多10秒；输出逐episode和按来源/种子
聚合的奖励分项、原始误差、活动度、失败/拒绝、延迟和前缀。有限时长行政截断明确
记录，不称全曲验收。评估恢复其自身随机状态，不推进训练采样器，也不进入训练Buffer。

## 服务器1有限检查命令

以下使用独立准备验收目录；先完成CPU测试和代码同步，再执行。该配置不关联旧200首
清单。长训练需另建明确预算和输出目录，不能靠重置准备账本延长本轮运行。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/user/liwei/GENMO/.venv/bin/python -B tools/train_closedloop_stage10.py \
  --config configs/closedloop/stage10_prepare_server1.yaml --mode preflight \
  --output-dir /data0/user/liwei/GENMO_outputs/closedloop_stage10/full_data_preflight_20260930

CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/user/liwei/GENMO/.venv/bin/python -B tools/train_closedloop_stage10.py \
  --config configs/closedloop/stage10_prepare_server1.yaml --mode train \
  --output-dir /data0/user/liwei/GENMO_outputs/closedloop_stage10/preparation_20260930 \
  --stop-after-iteration 2

CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/user/liwei/GENMO/.venv/bin/python -B tools/train_closedloop_stage10.py \
  --config configs/closedloop/stage10_prepare_server1.yaml --mode train \
  --output-dir /data0/user/liwei/GENMO_outputs/closedloop_stage10/preparation_20260930 \
  --resume latest --stop-after-iteration 4
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

服务器已完成全4765条内容与身份审计，版本5f33e3e的四轮真实训练、2→4完整恢复、
19项独立训练审计和物理journal审计通过；同版本初始/第4轮完整val评估仍在进行。
本地恢复计数、恢复审计和预算扩展修复正在完成最后回归；服务器评估期间不换源码。
后续将用新目录执行新版1→恢复2→显式扩限且停止在3轮的有限GPU验证，旧四轮与其
全量对照继续保留原版本身份，不冒称是新版第3轮模型的质量结果。最终结果完成后更新。
