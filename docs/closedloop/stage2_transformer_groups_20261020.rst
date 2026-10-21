Transformer全层分组学习率：三小时实施记录
=======================================

本报告记录2026-10-20服务器1上的真实实验。GENMO固定分支为
``feature/bumi-music-closedloop``，GMT为``feature/bumi-frozen-gmt-backend``。
计时起点12:32，启动前分析截止15:32（北京时间）。选择A；新MC、两轮八卡更新、
32任务配对评估及完整checkpoint读回核验均已通过。满足约定的开训条件；
两轮尚未证明物理收益，正式续训按同一实验累计上限200轮执行。
14:13已启动正式恢复命令，距开始约101分钟；tmux为
``genmo_transformer_groups_train``，从上述第2轮完整状态继续，未重新初始化。
14:24复查已接受第3轮、累计22次Adam，8个训练worker与GPU进程仍在运行，
无退出记录，已继续后续采集。该轮14314条链、7/7接受、3次回溯，
全链KL均值0.058010、Max1.558444，耗时487.57秒。

第3轮最后主干LR又降至1.5625e-8、输出头1.5625e-7：三次拒绝均来自原有
``chain_joint_max=1.6`` 候选约束（1.8619、1.8710、1.6587超限），不是ESS。
各层最后一次仍有至少85.1%的参数改变，最小层RMS3.63e-9，最终关节探针
平均变化1.55e-5rad；此前五次均以主干1e-7更新。尾部KL仍会压低步长，
本方案没有根治这一限制，后续须结合真实物理收益持续判断，不能把开训称为解决全部问题。

问题及本次解决办法
------------------

并非“参数多所以不能训练”。同一批14315条真实链上，原始全参数冷Adam以1e-6
更新一次，完整链KL均值达到13.6286。旧行为19个随机步骤重放的log-prob及ratio
差均为零，独立Gaussian最大误差9.1e-13，没有发现采样核不一致。

逐层隔离时第0、1层最敏感，但A的有符号标准化均值变化证明，更主要的问题是
很多层朝相近方向叠加：叠加方向能量为单层能量之和的11.66倍；与实际整个主干
更新方向余弦0.999815，残差约1.93%。这是有限增量探针结果，不是可相加KL或
严格Jacobian分解。全局梯度裁剪到1也不能按同样比例缩小冷Adam的实际增量。

本次只实现“基础LR乘固定模块倍率”：全部16层Transformer与其他模块倍率0.1，
原动作输出MLP倍率1。共同回溯和恢复增长保持倍率不变；未缩放梯度冒充学习率，
未改PPO、噪声、奖励、Critic、GMT或BC。Transformer始终参与15～18步更新。

两候选：相同Stage1、数据、初始Adam及一个Epoch
--------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 42 29 29

   * - 指标
     - A：主干统一倍率0.1
     - B：第0/1层降至0.025
   * - 计划／最终接受更新
     - 7／7
     - 7／7
   * - 回溯次数
     - 4
     - 5
   * - 最终整批链KL均值／最大值
     - 0.06323／1.57561
     - 0.06800／1.57693
   * - 最终整批裁剪PPO目标（越大越好）
     - -0.004228
     - -0.009276
   * - 最后一次Transformer实际LR
     - 全部2.384e-8
     - 第0/1层2.980e-9，其他1.192e-8
   * - 相同普通探针最终关节变化RMS均值
     - 0.0002241 rad
     - 0.0002307 rad

A在接受次数相同时目标更高、KL更小，因而选A。两者的整批目标仍低于起点约0，
不能据此宣称PPO或物理效果已经提高。两者ESS比例均约0.0385%，只记录、不拒绝；
该权重集中问题没有因分组LR而消失。A的所有层每次都实际改变，不是仅输出头
更新，各层每次至少93.2%的参数实际改变；最差舍入为零比例约6.8%。分离探针
还确认主干变化进入了有来源证据的GMT中心参考；覆盖约0.8%的
旧控制点，只证明这部分参考变化，不冒充新Actor的真实机器人运动。

验收与当前物理结果
------------------

服务器1已完成相关数学／多组生命周期回归91通过1跳过、条件缓存追加1通过、
TensorBoard12通过、分布式及恢复集成45通过。覆盖参数唯一分组、冻结参数排除、
共同回溯、Adam/BC/RNG恢复、checkpoint组身份、完整链流式梯度、尾批及空rank。
服务器没有ruff，未声称执行过。临时测试源码已清理，验收证据单独保留。

物理短训使用原Stage1 s595000重新初始化。旧MC资产与更新后的源码绑定不符，
不放宽身份检查，重新预热。新MC已通过：96首训练音乐×2种子，独立32个验证
episode、2389个transition，MSE2931.5356→155.2302（降低94.7048%），
EV -0.000333→0.839488；选择纯MC候选，耗时1226.51秒。
两轮分别14315/15804条有效链，7/7与8/8次Adam全部接受；回溯4/0次，无提前停止。
最终全链KL均值0.063231/0.071310、Max1.575608/1.546372。首轮结果精确复现
离线A；两轮的八rank完整Actor/Critic指纹均一致。第二轮结束主干LR恢复至1e-7、
输出头1e-6；所有层每步实际改变，优化期间PhysX时钟保持暂停。

32任务原episode、侧车SHA、奖励积分和协议配对重验通过。完成数29/32→28/32，
新增FineDance075、seed42在109.14秒失败（Stage1完成126.02秒）。原三个失败
仍保留：AIST++对应任务43.88→43.90秒，FineDance075 seed1729为
103.60→118.40秒，FineDance138 seed1729为132.90→128.06秒。
基础设施错误、参考拒绝和行政截断均为零。

来源等权、来源内任务等权：每秒回报2.387237→2.387887（+0.0272%）；
积分回报-0.1482%，实际活动度+0.0796%。跟踪RMS变化如下，负数表示误差降低：

.. list-table::
   :header-rows: 1

   * - 跟踪量
     - RMS变化
   * - Anchor位置／姿态
     - +2.362%／-1.544%
   * - 身体位置／姿态
     - -0.263%／+0.020%
   * - 身体线速度／角速度
     - -0.177%／-0.148%
   * - 双脚高度／腿关节位置
     - -2.820%／+0.373%

RMS由来源等权的任务MSE开根；失败任务不删除，执行时长不同可能影响曝光，
未作显著性或长期泛化声明。八项没有全面改善，不能说影子与机器人已经整体更贴合。
按约定，初期单个新增失败不阻止开训；开训依据是正确更新与正常闭环执行。

完整checkpoint读回确认18组、320个Adam状态均到step15，8份rank状态齐全，
模型指纹与物理评估一致；16层累计参数RMS变化1.949e-7～2.128e-7，
不进入优化器的Actor状态逐位不变。checkpoint SHA256为
``516d95b8cd83a2d234e268aacf248f2b4223e9d6f740b6605a8189c29581d0e5``。
短训退出0，源码与冻结GMT不变，物理世界正常关闭，无未完成指令。

两轮墙钟562.26/432.28秒；环境237.00/203.25、PPO59.74/66.35、
BC26.27/26.20、Critic14.00/12.02、KL121.07/111.54秒。
单次接受更新的Actor阶段均摊27.84/23.84秒；PyTorch峰值约25.29/25.50GiB，
不含PhysX外部分配。KL与Actor总阶段有重叠，不能把这些计时简单相加。

正式配置与查看命令
------------------

选定``configs/closedloop/stage2_transformer_groups.yaml``：八卡×128环境；
初始基础／输出头LR1e-6、Transformer／其他1e-7；真实联合PPO clip0.10、
全局2048链、Epoch1、梯度裁剪1、BC初始目标比例0.1。上限从创建起就是200轮；
短训在第2轮停止后，从同一run的最新checkpoint及Adam状态续训，前两轮计入。
固定物理评估在短训结束及50/100/150/200轮，100/200轮正式保存，短训模型保留。

服务器路径前缀为
``/data1/user/liwei/GENMO_outputs/closedloop_stage10/``。
运行目录为``stage2_transformer_groups_20261020_physical``；
证据目录为``stage2_transformer_groups_20261020_analysis``。
本地精简证据在``outputs/analysis/stage2_transformer_groups_20261020``。
训练代码身份为GENMO ``7652aa5``、GMT ``f689ab5``。
后续报告提交仅修改本文和记录文本，不改变运行时代码、配置或训练身份。

登录服务器后查看短训日志::

  tail -f /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_transformer_groups_20261020_analysis/physical.log

正式续训启动后查看::

  tail -f /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_transformer_groups_20261020_analysis/formal.log

正式恢复命令（本次由独立tmux启动器执行，勿重复启动占用同一run）::

  cd /home/user/liwei/GENMO-bumi-closedloop
  bash scripts/train_stage10_8gpu_server1.sh \
    --config configs/closedloop/stage2_transformer_groups.yaml \
    --output-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_transformer_groups_20261020_physical \
    --stop-after-iteration 200 --resume latest

恢复保留模型、Adam、BC与rank RNG；PhysX按原有协议重新开始episode，不冒充
跨进程恢复了仿真中间状态。两会话共用一个run和一个TensorBoard目录。

TensorBoard只打开该run的``tensorboard_compact``目录；``Policy/learning_rate``
是基础LR，本配置等于输出头LR，``Policy/transformer_lr_min``是主干实际最小LR，
``Policy/transformer_update_rms``是主干真实参数变化。详细16层值仍保留JSON。

边界：本轮证明与否分开报告——算法和更新正常不等于物理收益已经成立；
正式训练后的持续改善须看固定音乐完成、八项误差、活动度和奖励组成。


2026-10-20 晚间显存滞留修复与同实验续训
--------------------------------------

原正式会话在第37轮OOM，GPU3的逐轮PyTorch已分配峰值从第3轮26.858GiB
升到第36轮74.272GiB，其他卡约27GiB；最后完整保存的是34轮/270次Adam。
独立复现确认bulk_cpu_copy和GMT to_host的递归局部函数形成循环引用，捕获的
分组表会让CUDA输入等到Python循环GC才释放。empty_cache不能释放仍有引用的张量。

本次只给这两个函数增加try/finally，清空分组/CPU查找表并断开递归函数引用。
原函数体、输出数值、dtype、别名和合批传输原样保留，没有增加生产强制GC。
全部16层Transformer、Full-chain PPO、BC、Critic、奖励、固定噪声核和GMT不改。
运行时代码为GENMO 2512c90、GMT 867665f，仍在原两个固定分支。

服务器1相关回归64通过，含完整链概率/流式梯度、部署时钟和GMT列构造。
关闭自动循环GC后，每个函数连续4次搬运64MiB CUDA输入，返回CPU结果仍存活时
已分配显存每次立即回到基线（增量均0）；正常及三条异常路径的引用释放均通过。
临时源码、pytest目录和传输包已精确清理，正式训练和异常执行证据保留。

旧checkpoint绑定全部源码，修复后不能透明加载。因此新增独立维护工具，要求
两个旧Git blob与原checkpoint SHA一致，全部绑定源码恰好只变两条指定路径，
AST证明原计算体与模块其他语句不变、只新增指定finally清理；任一其他差异都拒绝。
完整2.6GB学习状态读回逐值一致后生成新维护快照并留原SHA/身份/指针及证明。
严格训练加载器未改，原checkpoint保留，不重新初始化Actor/Critic/Adam/BC/RNG。
维护后仍按既有协议重建PhysX并采新rollout，不导入旧行为数据。

恢复后实际完成第35～37轮：

.. list-table::
   :header-rows: 1

   * - 模型轮次
     - 有效Chain数
     - 计划/接受Adam
     - GPU3峰值GiB
     - 八卡最大峰值GiB
   * - 35
     - 14322
     - 7/7
     - 25.596040
     - 25.596040
   * - 36
     - 15830
     - 8/8
     - 26.763201
     - 26.794644
   * - 37
     - 16010
     - 8/8
     - 26.809019
     - 26.815618

上表为逐轮reset后的PyTorch peak_allocated，不含PhysX等外部分配。
GPU3后两轮只增加0.045817GiB（约46.9MiB），八卡相近，没有复现原每轮约1.4GiB
的持续单卡增长。reserved缓存和nvidia-smi总量可能继续变动，不等于活跃张量泄漏。
这证明确定缺陷已修且三轮观察正常，不保证未来所有200轮都无其他OOM。

23次计划更新全部接受，累计270→293；候选回溯0/2/0、提前停止均无。
三个完整rollout联合KL均值0.023855/0.076203/0.062367，Max0.380011/1.263386/1.293636。
逐步核验全部16层实际增量均非零，第37轮最后主干LR1e-7、输出头1e-6。
这仍不是物理收益证明，原尾部KL会压低LR的问题也没有在本次被重新设计。

为验证持久化并按剩余预算正常结束，本次通过训练自带StopSignal在第37轮完成后
受控保存，退出0；source_unchanged、冻结GMT不变、world关闭和无pending均通过。
新checkpoint完整SHA核验通过：

``checkpoints/stage10_r000037_a000293_01792503211357382744_8614b47e2263_controlled_end.pt``

SHA256 ``6c923cd744eb02cc4d64ba0584881125375c11e597bfbb8264f38922fb219ae2``，
18组、320份Adam状态均step293、8份rank状态。已从此完整状态在原run目录继续
八卡训练，tmux仍为``genmo_transformer_groups_train``。原故障会话35/36轮虽未落盘，
已接受预算不退款；现已消耗39次接受预算而模型编号37，因此剩余161轮，命令
``--stop-after-iteration 198``。配置累计上限和总接受预算仍200，无暗中扩限。
固定50/100/150轮评估保持，最终在剩余额度结束时执行完整评估/保存。

从本地查看本次正式续训日志：

.. code-block:: bash

   ssh 6000D-Server-1 'tail -n 40 -f /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_transformer_groups_20261020_analysis/host_copy_continue.log'

同一任务TensorBoard继续写原``tensorboard_compact``目录，不增加第二个run。
测试、维护来源和三轮运行证据保留在本地
``outputs/analysis/stage2_transformer_groups_20261020/host_copy_repair``，服务器analysis中
对应``host_copy_repair_tests_20261020.log``、``host_copy_checkpoint_migration.json``和
``host_copy_repair_runtime.json``；原run/maintenance中还保留旧身份、指针与AST证明。

21:36正式恢复已越过严格加载与世界初始化，进入``train_000038``采集；
按LOCAL_RANK核验恰好8个worker，无退出记录。最新完成轮次仍37，不能把
正在采集第38轮写成已完成38轮。对应证据为``host_copy_continue_verified.json``。


2026-10-21 取消源码门禁并通过完整恢复与固定评估
--------------------------------------------

用户明确要求源码版本不再作为训练/评估通行证。过去源码哈希表被放进训练和评估
记录，再要求与当前文件逐项相同，导致工程修复后仍被拦截；这与PPO是否能正确
计算无直接关系。此次统一取消该要求，不重写任何旧checkpoint或初始基线。

具体改动：

* execution_compatibility.py统一生成去除源码表的比较副本，三类源码记录仅作追溯。
* training_checkpoint.py恢复及评估Actor加载不比较源码版本，仍校验实际模型/Adam/核。
* evaluation_runtime.py、periodic_monitor.py、tensorboard_metrics.py及比较/回报审计工具
  一起移除源码相等要求，覆盖固定参考、配对统计、best指针和验证曲线完整链条。
* critic_preheat.py移除源码AST扫描/相等检查，导出只保留原checkpoint的来源记录；
  真实Actor权重、价值目标、监督窗口、冻结编码器与结构校验保留，不因源码变化重做MC。
* training.py源码快照只作best-effort日志，源码变化或记录失败不再影响训练退出码。
* 删除退役repair_host_copy_checkpoint.py及源码迁移专用测试。今后不需要迁移旧文件哈希。
* 两仓库AGENTS.md写入用户长期约定，禁止换个名字重新增加源码门禁。

服务器1隔离回归156 passed（9.32秒），包括实际旧源码参考进入真实评估入口、
非空Adam/BC恢复、配对统计/曲线/best、Critic预热、完整链PPO和GPU搬运回归。
实际模型形状、概率核或奖励配置不匹配的反例仍拒绝。临时源码/pytest/传输包已清理。

实际八卡验证直接加载原第49轮完整checkpoint，无源码迁移、无重新预热；
session为0801785c-6a9b-41a1-a975-f2a21ba6e3d9。重新完成第50轮：14325条有效链，
7/7更新接受、0次候选回溯，Actor累计388→395。32个完整音乐任务全部执行，
固定参考入口、统计、曲线和完整保存全部通过，退出0，冻结GMT未变、world关闭。

第50轮完整checkpoint SHA256：
``0803bc1d0864962fb8171db2c516928397860162224c3168253b8a355f97d71c``。
路径为原run下``checkpoints/stage10_r000050_a000395_01792551314932299248_52c841582412_evaluated_periodic_r000050_c000020666121_a000050+controlled_end.pt``。

第50轮的实际物理结果与源码规则修复分开判断：Stage1物理失败3/32，当前4/32，
其中新增失败2个、原失败恢复1个；参考拒绝为0。来源等权每秒回报
2.387237→2.390481（+0.1359%），配对差0.003244分/秒，95%区间
[-0.004090,0.011393]跨零，尚无明确整体收益。实际活动度保留99.8863%。
八项跟踪RMS相对变化如下，负值为误差降低：

.. list-table::
   :header-rows: 1

   * - 指标
     - RMS变化
   * - Anchor位置
     - -2.292%
   * - Anchor姿态
     - +2.266%
   * - 身体位置
     - +0.268%
   * - 身体姿态
     - -0.171%
   * - 身体线速度
     - +1.634%
   * - 身体角速度
     - +2.043%
   * - 双脚高度
     - -0.810%
   * - 十二腿关节位置
     - +0.475%

没有因为新增失败而禁止训练，也没有修改奖励、Critic、Transformer训练范围、
Gaussian噪声、PPO或物理时钟。当前只是取消了人工源码门禁，不能把恢复训练称为物理收益。

10:56服务器1已从同run第50轮/395次完整状态继续正式八卡训练，tmux为
``genmo_transformer_groups_train``。10:58核验会话ed2c9524-cae0-4ef5-9931-a9421f96f402
已进入第51轮采集，LOCAL_RANK 0～7恰好8个worker，无退出文件。
总接受预算仍200；原未落盘的2轮及本次修复前丢失的第50轮均保留消耗，当前
预算已接受53而模型编号50，剩余147轮，实际停止编号197，不暗中扩限。

查看当前训练日志：

.. code-block:: bash

   ssh 6000D-Server-1 'tail -n 40 -f /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_transformer_groups_20261020_analysis/source_records_formal.log'

本地证据为``outputs/analysis/stage2_transformer_groups_20261020/source_records_fix``；
服务器analysis保留``source_records_tests_20261021.log``、
``source_records_validation_summary.json``及``source_records_formal_verified.json``。
原基线和旧checkpoint仍为原记录；训练的新版本仅记录在新会话/新checkpoint中。
