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

首次保存`checkpoints/initial.pt`；若第一轮尚未发布成功，可显式恢复这个完整初始
边界并保留已消耗预算。`--initialize-stage9`是明确的Actor/Critic权重迁移，验证
资产、条件、奖励、随机核和物理契约；优化器/采样器/计步重新建立，不称full resume。

每个进程持有run写锁；SIGINT/SIGTERM请求在下一完整轮次边界结束。一般异常停止
session并保留证据，恢复最近完整checkpoint；不在部分更新的Critic/BC状态上继续。
各session/轮次目录不覆盖旧证据；运行时磁盘余量、配额和checkpoint预留均检查。
不自动删除原始执行数据或正式checkpoint。

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

## 当前验收状态

实现与CPU回归进行中。服务器已只读确认全4765条路径存在及清单划分无泄漏；完整
文件内容审计、真实多轮/续训/评估结果由后续日志补充，当前不预写全部通过。
