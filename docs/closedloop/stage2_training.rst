当前第二阶段训练
================

只保留一套配置和实现
--------------------

GENMO分支feature/stage2-asymmetric-denoising，配套GMT分支
feature/stage2-tracking-history。唯一第二阶段训练配置为
configs/closedloop/stage2_credit_server2.yaml，默认入口为
scripts/train_stage10_8gpu_server1.sh。脚本沿用原文件名，实际路径和资源配置来自
服务器2的当前配置；不会启动服务器1，也不会自动抢占已有GPU进程。

旧非对称候选、学习率扫描和随机20轮配置、多线程配额采集器、
旧完整checkpoint恢复及一次性工程迁移脚本已删除。历史实现从Git获取，不在当前
工作目录保留另一套入口。Stage1和MuJoCo sim2sim是独立功能，其配置继续保留。

当前参数与执行
--------------

* Stage1 s595000初始化可训练Actor；独立冻结Stage1网络用于前15步和确定性末步。
* 第15～18步为真实高斯转移，标准差下限0.04/0.06/0.08/0.10。完整执行20步。
* 实际生成60帧、30Hz、qpos30＋contact2；每0.5秒重规划，已承诺前缀不能修改。
* 八卡，每卡128环境，一个共享PhysX世界连续采集8秒；健康episode跨轮保留。
* 默认Clipped DPPO：Actor统一学习率3e-7、一个epoch、全局minibatch4096条有效链、
  计算微批128、clip0.01。每epoch打乱完整链，四步15～18全部参与；使用自由坐标
  平均log-ratio代理，保留真实joint概率诊断。约16000条有效链产生4次Adam更新。
* 原Stage1 BC每轮1024样本，按实际计划更新数分配（4步时每步256），动态目标梯度
  比例0.1；PPO和BC一次全局SUM、统一裁剪1.0，不重复除以卡数。
  soft KL提前停止时如实记录已经执行的步数和BC数量，不补做没有PPO依据的更新。
* Critic音乐/前缀编码器冻结，历史跟踪GRU独立训练1e-5，时序/Value层1e-4；
  完整MC训练和独立验证继续保留，gamma0.99每0.5秒，GAE lambda0.98。
* 八项GMT Tracking、配对动作活动门控、参考/实际限位0.7/0.3；音乐和独立稳定性
  奖励关闭，音乐条件、BC音乐动作损失、物理失败和事件罚分保留。
* minibatch复用PPO前向的坐标平均KL，达到0.00045时丢弃当前未执行更新并停止。
  整轮末一次完整四步KL，门限0.0009/0.0018/0.0021；失败恢复整轮Actor、Critic、
  Adam、BC和RNG，保留实际资源消耗。普通DPPO不逐minibatch复制模型或全链回溯。
* joint_score_pg仅保留为显式可选目标；配置需改为一次全批量minibatch及1个epoch。
  只有该目标使用最多9次0.5倍率Adam回溯，接受的完整KL仍可按实际身份复用。
* GMT50Hz、物理200Hz，每控制四个真实子步；训练优化期间世界暂停。

Critic90历史最后六维为已执行参考的根质心线/角速度，已从世界坐标旋转到当前真实
机器人机体系。Actor48不变，90+时间仍为91维，音乐/前缀冻结、历史GRU可训练。
GMT和GENMO同时采用Tracking90 v2；base_link参考与实际root使用同一刚体质心速度。

配置使用当前continuous_tracking.v2身份，不再识别旧训练配置。优化参数与数据、
GAE、BC和KL一起绑定；修改参数时同步更新optimization_sha256，不借用旧验收结论。
如需重新计算该字段，可在配置更改后显式执行以下命令；它只更新配置参数摘要，
不会赋予物理验收状态、修改checkpoint或启动训练::

    python - <<'PY'
    from pathlib import Path
    import yaml
    from gem.closedloop.dppo.actor_optimization import optimization_binding, digest
    path=Path('configs/closedloop/stage2_credit_server2.yaml')
    text=path.read_text();config=yaml.safe_load(text)
    before=config['stage10']['kl']['calibration']['optimization_sha256']
    after=digest(optimization_binding(config['stage10']))
    path.write_text(text.replace(before,after))
    PY

运行与恢复
----------

2026-10-19八卡有限验证首轮被最终KL拒绝，接受轮数为0。用户明确要求保持全部
参数、不再调参试跑。以下为运行接口说明，当前配置尚不能作为已通过的正式训练
基线；不得将本次新Critic预热通过视为Actor训练通过。

先确认目标八张卡空闲及模型/数据路径正确。新运行必须使用独立空目录，下面命令
仅为示例，不表示已自动启动新训练::

    bash scripts/train_stage10_8gpu_server1.sh \
      --output-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/新运行目录 \
      --stop-after-iteration 2000

同一份源码和配置的中断恢复::

    bash scripts/train_stage10_8gpu_server1.sh \
      --output-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/新运行目录 \
      --resume latest --stop-after-iteration 2000

累计最多2000轮，每100轮保留里程碑，30分钟滚动容灾保留最近2份。评估和MC每
100个已接受轮次执行，同一边界合并一份完整checkpoint，best保存实际权重。
恢复Actor、Critic、两个Adam、BC、各rank RNG和累计资源，PhysX按合同重建并
开始新episode，不恢复中途rollout。SIGINT/SIGTERM仍在合法边界保存后退出。

本次Critic输入改变了物理含义，checkpoint使用asymmetric_tracking.checkpoint.v2。
旧运行不能直接用新代码--resume，不能改SHA或绕过检查。首次运行从s595000创建
Actor和独立Critic，重新采集完整MC预热；后续只能恢复同源码、同配置的新身份。

诊断与证据
----------

唯一入口仍支持--evaluate-only、--evaluation-actor、--mc-validation-only、
--critic-preheat-only、--collect-only、--calibrate-only、--strict-evidence及
--verify-kl-reuse；这些是当前方案的检查模式，不是其他算法入口。

普通rollout在本轮优化及KL完成后释放；评估逐控制记录以单份无损压缩侧车保存，
异常/严格模式可保存完整证据。保留资源账本、显存/磁盘保护、参考截止和物理终止，
不设置48小时总训练期限。工程效率验收见docs/reports/stage2_efficiency_20261019.rst，
该报告记录对应历史实测提交；源码精简不自动继承新的物理学习收益证明。

本次有限验证结论（2026-10-19）
------------------------------

实测源码为GENMO 18acec6、GMT 58ce29d，之后仅追加结果文档，不改变算法和参数。
CPU核心回归GENMO 537通过/4项CUDA条件跳过，GMT 55通过；服务器2使用官方Isaac
数学实现的配套回归23通过。真实八卡已完成新v2输入的完整MC预热、32任务初始
物理评估和一次PPO更新尝试，但未完成原计划2～4个已接受轮次。

* 新Critic预热验证MSE 3076.654→181.038，EV -0.000337→0.803958；选中纯MC。
  32个验证任务中31个提供有效完整episode状态，共2389条状态。该验证组参与
  MC/TD候选选择，不能视为额外未使用测试集，也不证明Actor物理性能改善。
* 第一轮采到14315条有效链，计划4次Actor更新，实际执行3次（每次4096链）；
  BC实际768条。第4批2027链在更新前soft KL=0.001091处停止，尚未执行其Adam。
* 3次更新后完整四步平均坐标KL=0.001099451（限0.0009），最大单步均值
  0.002069020（限0.0018），P95=0.002701997（限0.0021），三项均超限。
  joint-chain KL均值5.550、P95 9.022；真正触发PPO目标裁剪的比例0.1764%。
* 8个rank均通过整轮回滚字节一致性检查；Actor、Critic、两个Adam、BC控制器和
  RNG恢复，已花费交互与优化尝试仍计账。接受PPO轮数为0，未保存失败Actor。
  最后有效checkpoint为iteration0的新v2预热Critic及原Stage1 Actor。
* 第1/2/3次更新的加权BC/PPO梯度范数比约0.100/0.089/0.598。平滑BC比例控制的
  目标0.1不是逐批硬上限，第3批BC梯度增大是后续需核对的问题；本次未做额外
  消融，不能据此认定它是KL失败的唯一原因。
* 真实物理执行完成8秒共享世界，预算物理子步1638400；50/200Hz及四子步不变。
  初始固定32任务有4次物理终止，训练采集中记录6次物理终止。冻结去噪/GMT检查
  通过；128条参考变化诊断的已承诺坐标全部不变。物理失败不被包装为成功完成。

首轮采集（含目标与预算）最长241.069秒；rank0其中GENMO生成89.642秒，首次列式
独立核对51.487秒，仅该进程首次执行；这些时间有嵌套，不能直接相加。Actor已执行
步骤及被丢弃minibatch计时合计67.885秒；最终完整KL及参考变化19.357秒（纯高斯
KL 17.343秒）。按文件时间与采集计时估计，从采集开始到拒绝证据落盘约511秒，
包含首次完整概率回放、Critic训练和异常归档，不能作为稳态接受轮吞吐。
拒绝路径提前抛错，未持久化普通轮总计时及Critic/概率检查独立耗时；该计时缺口
明确保留为局限，不把剩余时间猜测归因于PhysX或网络。

用户选择保持soft KL 0.00075及全部原定参数，本次不再试跑、不启动长期训练。
服务器2任务退出1且GPU进程已释放，服务器1未改动。详细小型结果保留于：

* 本机：/home/weili/bumi-closedloop-worktrees/GENMO/outputs/stage2_clipped_dppo_20261019
* 服务器2：/data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_clipped_tracking90v2_20261019

按测试产物清理约定删除本次不再使用的异常完整张量、两份被latest覆盖的初始化
checkpoint和临时USD缓存，共34.678GiB。保留2.560GiB，包括唯一有效完整checkpoint、
MC/评估压缩控制证据、恢复所需固定诊断输入及失败/回滚报告；原始失败rollout已
删除，后续不能声称仍可从该目录重放完整失败梯度。旧正式实验及其模型未改动。

2026-10-19后续：仅提前soft KL停止
--------------------------------

用户随后授权将soft KL从0.00075改为0.00045，并在服务器2重新进行有限八卡验证。
统一LR3e-7、clip0.01、全局4096链、四个可训练步骤、BC以及全部硬KL门槛保持不变。
继续复用现有更新前soft停止，停止剩余minibatch后仍须通过整轮完整四步KL；
soft停止不等于失败，硬KL拒绝仍整轮回滚。此前失败结论属于0.00075配置。
本次仅用经严格身份核验的相同v2 Critic预热成果初始化新运行，不恢复旧Actor更新
或旧rollout。

本次实测源码GENMO 62e0d50、GMT 28b1001；40项定向CPU回归通过，1项本地
配套GMT路径检查跳过。服务器2新采集的两轮均通过完整四步最终KL，未发生整轮
回滚；学习率始终为3e-7，没有通过回退学习率获得通过。

* 第1轮14315链，完成2次4096链更新，实际BC512条；第3批更新前KL
  0.000519181触发soft停止。最终平均坐标KL0.000522549、最大单步均值
  0.000960054、P950.001189551，joint-chain均值2.637826、P954.005357。
* 第2轮15806链，完成3次4096链更新，实际BC768条；第4批更新前KL
  0.000659282触发soft停止。最终平均坐标KL0.000664339、最大单步均值
  0.001326865、P950.001827228，joint-chain均值3.353583、P955.957159。
* 两轮的硬上限仍为0.0009/0.0018/0.0021，最终检查覆盖全部14315/15806链
  和每链4步。soft阈值不是更新后KL硬上限，也不是固定限制每轮更新次数；
  它根据当前minibatch更新前的策略变化决定是否继续，不能保证以后永不超限。
* 两轮各128条固定条件最终参考探针的关节变化RMS均值为0.001728748和
  0.001752003 rad，已承诺坐标全部不变。这里是最终确定性映射后的参考诊断，
  不能混同index18高斯均值变化，更不能代替完整歌曲物理结果。
* 第一轮396.120秒，第二轮301.712秒；采集含目标/预算分别237.213和207.573秒，
  Actor47.219和59.280秒，Critic10.127和11.049秒，最终KL及参考探针17.725和
  21.305秒。首轮还有完整概率回放及首次列式核对，不能将两轮差值全算作优化收益。
  soft检查复用梯度前向的KL，第三/第四批已花费的前反向11.975/14.328秒仍计入
  Actor耗时；本次未为省这部分计算增设额外KL前向。物理连续性及更新暂停检查通过。

同一运行的更新前后固定32任务（val组16首独立音乐×2seed）完整物理评估：
评估身份和任务清单完全一致，不声称Stage1预训练未见过这些音乐。完成数29→30，
物理失败3→2，新增失败0、恢复完成1；分来源平均回报113.808908→116.104896
（+2.0174%），按实际执行时间汇总的每秒回报2.258096→2.266974，平均执行时长
54.735→55.721875秒，实际活动度0.874671→0.876214 rad/s。
FineDance075/seed1729从103.60秒失败变为126.02秒完成；FineDance138/seed1729
仍失败，但执行132.90→141.20秒；AIST对应失败也未恢复。不将延迟失败当作完成。

收益主要集中在FineDance，其他三来源平均回报仅小幅变化。关节边界并非全面改善：
GMT消费参考最大硬越界0.053092→0.152976 rad，生成可修改部分最大硬越界
0.199535→0.216159 rad；两者越界比例分别0.041760%→0.044171%、
0.046725%→0.047252%。实际机器人最大硬越界0.000901→0.000723 rad，但实际
越界比例0.008047%→0.008332%。因此本次证明提前停止可以保留有效更新，并有
单次配对评估正向信号；仅2轮、5次更新不能证明稳定泛化收益，也不能保证后续
每轮均通过硬KL。参考越界增加是继续训练时必须关注的未解决问题，本次不额外改奖励。

测试退出0，两个世界正常关闭，GMT冻结及源码未变检查通过，八卡已释放。
iteration2实际权重SHA为0b7b57d6ec4a2ad2e30844e10c996c606d76e29f67287befcf5010de48cf80cc；
latest与best均指向该模型。CPU加载核验包含Actor、Critic、非空两个Adam、BC控制器、
8rank RNG/BC/collector状态和资源预算；本次没有另外启动恢复训练来声称完整恢复实跑。
最终评估与controlled_end共用一份2.458GiB完整checkpoint，没有重复保存同一最终状态。

本机证据：/home/weili/bumi-closedloop-worktrees/GENMO/outputs/stage2_softkl045_20261019。
服务器2：/data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_softkl045_20261019；
运行配置runtime.yaml额外绑定经严格核验的同v2 Critic预热文件，主配置预热字段仍为空。
本次按计划停止在第2轮，没有启动长期训练，服务器1未修改。已删除16处临时USD缓存
释放0.569GiB；剩余约4.986GiB保留最终模型、被基线及初评引用的初始化模型、
轻量轮次记录、评估压缩证据和恢复诊断输入，不保留普通rollout完整训练张量。
