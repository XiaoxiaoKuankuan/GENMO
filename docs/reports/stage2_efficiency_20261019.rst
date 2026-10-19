服务器2：第二阶段训练工程优化验收
================================

本报告保存对应历史提交的实测证据。其后的源码清理已删除一次性迁移、旧分片
评估和候选实现，当前唯一配置与入口见docs/closedloop/stage2_training.rst。
历史回归及物理数据不自动构成清理后源码的新GPU或学习效果验收。

目标和边界
----------

本次针对八秒连续世界运行的重复KL、逐步重型诊断、CPU参考处理和重复归档进行优化。
不更换奖励、Actor/Critic结构、扩散随机核、学习率、GAE、BC目标比例或物理时钟。
原训练在582轮通过SIGTERM安全结束，退出码0，原模型不覆盖。新的运行以显式工程
迁移记录父checkpoint、源码差异、数值回归证据和完整学习状态摘要；不是新算法
复用旧优化器，也不在正常checkpoint加载器中放宽身份校验。

GENMO分支feature/stage2-asymmetric-denoising，配套GMT分支feature/stage2-tracking-history。
共享本地训练分支未切换，本次临时检出目录在交付后清理。

实际修改
--------

1. kl_reuse.py、full_batch_actor.py、training.py：只复用已接受Adam事务中实际完成
   的全量四步KL。凭据绑定Actor参数/buffer版本、模块执行设置、真实随机核、
   rollout张量、全局有效manifest、clip及归约定义；任意rank不匹配时全部重算。
   所有有效链的15、16、17、18四步及冻结前15步/确定性末步身份仍完整审查。
   --verify-kl-reuse在会话首次复用时独立全量重算，核对结果且单列额外耗时。
2. full_batch_actor.py、actor_backtracking.py：普通轮不再做整模型FP64 Adam重建、
   参数逐元素变化和重复输出探针。首轮及每100轮保留完整诊断。必要全局梯度、
   PPO/BC范数与方向、动态BC控制、裁剪、非有限检查、Adam回溯保留。
   gradient_snr_diagnostic=false确实关闭额外半批梯度通信；首次Adam尝试不冗余
   恢复刚捕获的同一参数状态，重试仍从原Adam/BC/RNG/梯度完整恢复。
3. memory_execution.py、tensor_cache.py、execution_checks.py、frozen_prefix.py：
   因果历史及Actor/Critic共同条件批量逐元素核对后复用同一份张量；特权历史
   仍独立且不进入Actor。冻结身份在计算阶段入口/出口完整检查，阶段内部复用
   签名；动态mask和非有限检查仍全覆盖。可训练特征不跨optimizer step缓存。
4. vector_generation.py及GMT的inprocess_service.py、execution_journal.py：提交
   参考只传真实执行合同字段，不携带整棵生成诊断。同步exchange内同一不可变
   负载只做一次摘要；下一次调用重新计算，仍拒绝序列/负载变化和错误ACK。
   GMT closedloop_reference.py、vector_backend.py按合法推进区间批量读取参考
   和差分一致性。逐真实控制步的保护水位、10步前瞻、终止和四个物理子步不变。
5. compact_evidence.py及evaluation/periodic_monitor/evaluation_shards：逐任务
   50Hz控制记录无损列式压缩成单份文件；JSON保存摘要、大小、SHA及引用。
   合并报告不再次嵌入相同全部控制。审计和比较工具通过同一读取器还原。
6. evaluation_runtime.py、evaluation_world.py、vector_evaluation.py及
   mc_calibration.py：常规评估/MC使用内存journal和临时GPU扩散链；固定初始
   漂移诊断只保留每卡4条真实链。每控制奖励分项、积分、跟踪、活动度、限位和
   事件仍保存；每100控制步保留独立标量奖励核对。完整MC价值标签在当次训练/
   验证期间完整驻留内存，用完释放，默认不再反复持久化大bank。
7. training.py、training_checkpoint.py：同一边界先评估、再MC、最后只发布一次
   完整快照，包含全部维护原因及MC后的Critic状态。best保存实际权重。安全
   信号不再额外触发一次耗时整曲评估；所有rank一致停在合法边界并保存。
   周期严格诊断改为每卡8条真实链、保留最近2次；显式--strict-evidence仍可
   保存完整证据，异常路径也仍可保留完整当前rollout。
   正常训练终点与周期维护重合时也合并；退出凭据核对Actor/Critic真实Tensor
   版本、完成计数和资源水位，发生变化则重新保存。已经达到指定累计轮次的
   普通resume直接结束，不重复整曲评估。初始和预热容灾快照仍保留。

历史证据与恢复
--------------

tools/compact_stage2_run.py默认只清点，--apply取得停止运行的排他锁后才修改
归档格式。旧episode逐控制记录重构后逐项比较原值，引用SHA随格式迁移更新并
记录原SHA。Stage1、最佳点、近期百轮及当前恢复点的完整物理SQLite按原字节
gzip并验证解压SHA；其他重复底层journal明确退役，不宣称仍有全量PhysX回放。
已完成训练的旧全量rollout先提取每卡8条真实链并校验，再释放全量数组。
完整MC标签、固定初始诊断及所有模型保留；模型只对相同字节SHA去重硬链接，
不将不同Critic/Adam/RNG状态的checkpoint冒充同一个恢复点。预算SQLite不清理。

tools/migrate_stage2_efficiency.py只允许本次列出的工程源码差异，严格比较奖励、
随机核、物理、数据和训练参数。验证证据绑定目标源码清单；写后重新加载并核对
Actor、Critic、两个Adam、BC控制器、调度状态、各rank随机状态和训练计数摘要。
资源SQLite通过SQLite backup继承，实际消耗不归零。原latest和原checkpoint不改。
新目录按既有合同重建PhysX，从新episode继续，不伪造中途物理恢复。

参数
----

沿用configs/closedloop/stage2_credit_server2.yaml：八卡、每卡128环境、连续8秒
世界rollout；Actor统一LR3e-7、1 epoch、全局有效链容量16384、Actor/KL微批128。
joint-score单次on-policy更新加BC1024样本、目标梯度比例0.1；不是joint ratio
指数化PPO。60帧/30Hz参考、0.5秒重规划、20步15+4+1，随机四步std为
0.04/0.06/0.08/0.10；gamma0.99/0.5秒、lambda0.98；三倍四步KL上限
0.0009/0.0018/0.0021。Critic时序/Value学习率1e-4，历史GRU1e-5。
完整歌曲评估和当前Actor独立MC仍每100轮，百轮权重和30分钟容灾机制保留。
最多2000个已接受外层轮次，真实资源/显存/磁盘保护与物理终止不变。

数值及系统验证
--------------

* 初次集成本机GENMO CPU回归208通过、3跳过；补齐曲线消费者、正常终点、
  信号边界和预算水位后，服务器2最终回归213通过、3跳过（23.47秒），另跑
  顶层引用/KL/恢复完整性13通过。GMT真实FK/参考及
  journal回归36通过。服务器PyTorch2.7.1+cu128，不改运行环境或模型资产。
* 八卡Gaussian直接log-ratio与原联合概率相减最大误差3.61e-10，clip分支及
  去噪采样归一化检查通过。
* stable_tf32x3、60帧真实Stage1条件在batch1/4/8/16/32/64/128、置换、尾批及
  有/无梯度重算通过；实际Transformer长度为60，最大误差量级1e-12。
* 八卡Critic参数隔离/缓存、全局梯度SUM、空rank、三步非空Adam及checkpoint/
  RNG/SHA损坏拒绝通过。GMT GPU/CPU ONNX逐批对照按既有atol/rtol1e-4通过。
  最终提交c10f393另在八卡复验策略与Critic，真实网络前向后的Tensor版本读取、
  预算变化失效及原位写版本变化失效均通过。实际四轮闭环使用f72f8cf；之后仅
  training.py的退出维护调度改变，没有改变网络、概率、奖励和物理数值实现。
* 另有13个旧测试在未改动4404c5b上同样失败，涉及旧full20/配额fixture及缺少
  当前Critic条件/随机步骤的mock；未为这些旧夹具增加生产兼容路径，不能宣称
  整仓库所有旧测试均通过。cpu_baseline.log保留复现。

真实闭环、存储和运行状态
------------------------

原运行已完成离线整理，实际分配占用由299862876160字节（279.269GiB）降至
79225319424字节（73.784GiB），释放205.485GiB，约73.58%。这是停止轮次及
结束评估完成后的目录快照，晚于用户此前265GiB的读数；不混为同一时刻。
其中模型仍占54.320GiB，关键原始SQLite冷存5.311GiB，抽样训练链0.049GiB。
保留完整物理数据库的评估点为0、500、582；582同时是当前best和恢复点。
其余阶段的控制奖励/指标仍可读，已退役底层journal不再声称可做完整物理回放。
全部顶层evaluation_artifacts引用、初始/最终32任务逐控制奖励汇总和曲线复算
均通过；没有用重新计算的reward替换原始实际物理奖励。

完整工程迁移从原582轮实际checkpoint出发，Actor指纹
905ef4adf7d9f159a3511047d2745b62d760183471ca1c2da73ecd54aecc2dd0未变，
学习状态内容摘要bd4a41e378a279c0c9849f3b73f59dd90596f8279ce88e98c820f6a926b606f9
写后重载一致。原文件SHA为c8eaa6017abaa3efe748dcef318b20b3f1d2682b5551bde2b8a4b07891434d06。
新的完整checkpoint因工程身份和父证据改变，文件SHA不同；不把它称为原文件。

首轮583已真实接受：14322条有效链、57288个随机转移，LR3e-7、0次回溯。
复用结果与独立全量四步重算的全部KL统计精确相同，平均坐标KL1.4234324385e-5。
332.248秒墙钟中包含额外17.434秒独立复算，且首次重建世界；不作为正常轮提速
结论。其后584/585/586三轮分别300.995/312.358/311.792秒，中位数311.792秒，
此前578～582五轮中位数351.647秒，下降11.334%。按实际有效行为链总数除以总
墙钟的吞吐45.340→51.676链/秒，提高13.976%。不是只选最快301秒作结论。
三轮有效链数15835/15990/15983，每轮一次实际Actor step、BC1024、0次回溯。
四轮都复用完整四步KL，覆盖14322/15835/15990/15983条链，没有减少审查范围。
每卡仍400控制周期、每控制四物理子步；正常物理episode跨轮持续，优化时仿真
暂停检查通过。有限样本、不同音乐/失败重置工作量仍存在，不声称统计收敛结论。

第584轮rank0：世界摘要0.534秒、lane摘要0.079秒、参考诊断9.443秒、执行列
构造7.922秒。此前582轮分别约10/9.453/11.885/8.205秒。它们有嵌套关系，不把
分项直接加成所谓节省总量。参考CPU通路和约320MiB/卡/轮的数值传输仍存在，
Actor计算本体也没有做新数值核加速；本次收益以整轮墙钟和实际吞吐为准。
PyTorch峰值仍约32GiB，NVML含PhysX等外部分配约41GiB；显存占用率不是提速
验收指标，不能把占用更多显存等同于学习效果更好。

独立物理对照与MC
---------------

为了检验工程改动是否改变动力学，额外用原运行的GENMO a36edd4和GMT79996de
源码快照，以及优化后f72f8cf和6d58d16，分别加载完全相同的582轮Actor，在新建
PhysX世界运行同一32任务。两次共89766个控制记录、359064个真实物理子步逐项
完全一致，包括控制字段、逐步奖励、终止与累计回报；最大回报差为0。旧源码
快照逐文件与原checkpoint源码指纹相同，快照仅临时使用，验证后删除。

相同新建世界协议下，582轮与586轮的对照如下：

=========================== ================ ================
指标                        第582轮          第586轮
=========================== ================ ================
完成任务                    30/32            29/32
真实物理失败                2                3
来源平衡任务回报            116.8224         114.8794
每实际执行秒回报            2.27365          2.27589
平均实际执行秒数            56.10375         54.98938
实际活动度 rad/s            0.86686          0.86919
Tracking分数均值            0.87859          0.88008
=========================== ================ ================

这里每秒回报略升不能解释成提升，因为完成率下降、执行时间缩短。586轮三次
yaw失败分别为AIST gPO_sFM_cAll_d11_mPO0_ch08/seed42、FineDance075/seed1729、
FineDance138/seed1729；075为新增失败。138的实际执行时长157.56→140.58秒，
075为126.02→107.52秒。所有失败任务及逐步执行数据保留，没有剔除后重算均值。
两模型限位、活动度、逐任务回报和完整配对表见physical_quality_summary.json。

原长期驻留世界的582轮评估为32/32，但相同Actor新建世界为30/32，且旧、新
软件新建世界完全一致。该差异不是本次工程优化造成；重建世界与长期驻留
世界的剩余评估状态差异尚未定位，不把原0失败作为新建世界的同条件结果。
今后不同运行的收益比较应使用一致的新建世界评估协议，并保留失败。

586轮独立固定Actor完整MC验证：32个完整episode、2413个上层状态、59984个
真实控制步；EV=0.95800695、MSE=39.821685、目标方差851.05485。配对运动
初始化1903个状态的EV=0.95257，站立初始化510个状态的EV=0.98805。fit=false，
没有借验证数据更新Critic或Actor。MC墙钟438.34秒，包含独立世界建立；这是
价值标定证据，不证明优势能可靠区分邻近动作，更不证明Actor已经提高成功率。

有限训练后的整曲评估phases实际850570394字节（约0.792GiB），独立MC目录
162085794字节（约0.151GiB）；两者SQLite、raw_samples扩散链及complete_mc.pt
数量均为0。它们保留单份无损50Hz控制证据，未以丢弃真实奖励制造空间节省。
最终586轮仅一份完整controlled_end快照，评估、MC和训练会话均退出0，GMT冻结
关闭检查通过。MC-only不拟合，不产生另一份相同模型checkpoint。

恢复与使用
----------

最终完整恢复状态位于集中根目录的train/，来自continued_train/的586轮完整
checkpoint。continued_train明确是四轮有限验收父证据目录；原始train位于集中
根目录的上一层，保留582轮原模型，不覆盖。工程迁移先后两次都显式保留父
SHA和学习状态摘要，第二次只接入最终退出维护修复，不改算法或重置资源。
verification_report_v2.json绑定当前实际源码、213项CPU、八卡最终复验及32任务
物理逐控制一致性。旧verification_report.json仍保持原字节供父迁移追溯。

正式续训沿用原授权累计最多2000轮，每100轮保存里程碑，30分钟滚动容灾保留
最近2份，best保留实际权重。同轮评估/百轮/MC仅一次发布。SIGINT/SIGTERM仍在
合法边界保存，普通中断恢复通过--resume latest加载Actor、Critic、Adam、BC、
RNG与累计轮数，重建PhysX开始新episode。没有新增48小时训练时限。

登录服务器2后查看实际进度，不需遍历GB级轨迹::

    python3 /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_credit8_20261016/efficiency_20261019/show_status.py

实时日志::

    tail -f /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_credit8_20261016/efficiency_20261019/optimized_train.log

新运行checkpoint目录::

    /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_credit8_20261016/efficiency_20261019/train/checkpoints

工程速度和归档优化已通过验收，继续训练的意义仍需第600轮及后续相同协议
整曲验证来判断；目前不能宣称强化学习收益已被证明。本次不再改变学习率、
奖励或Critic，也不把增加显存占用当作下一项优化。剩余主要耗时是实际采集
和Actor计算，后续若继续优化GPU参考通路或数值核，应单独核验，不能复用本次
工程对照为尚未实施的数值核背书。

证据统一根目录：
/data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_credit8_20261016/efficiency_20261019

本机只保留小型验收结果：
/home/weili/bumi-closedloop-worktrees/GENMO/outputs/stage2_efficiency_20261019
