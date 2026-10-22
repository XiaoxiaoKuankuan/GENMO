逐去噪步自由坐标平均代理PPO
==========================

任务与执行边界
--------------

本轮在GENMO的feature/bumi-music-closedloop直接开发，GMT固定
feature/bumi-frozen-gmt-backend。只读参考origin/feature/stage2-asymmetric-denoising
的公式与掩码，不切分支、不整体合并。服务器1旧正式任务通过八worker原有StopSignal
在保存边界退出，监督器不再重启；最终完整保存累计第90轮，Actor更新计数192，
SHA256为7c0ee5fc82ee484078eca6a925f58d17167ead688628cdf1d1fae8127958c86b。
模型、Adam、Critic、日志和历史配置保留。先本地修改，确认远端无worker后才同步。

本次只进行代码与有限验证，不自动开启新的200轮正式训练。奖励、Critic、GMT、
20秒随机运动片段、超时自举、上层gamma=0.95、全部16层Transformer的训练范围和
分组学习率、BC关闭、真实20步DDIM采样核、60帧和全部物理时钟保持。

目标与流式梯度
--------------

每条行为链i仍含15、16、17、18四个随机可训练高斯转移。对每步先使用生产FP64
稳定公式计算完整自由坐标joint_log_ratio，再除该步有效坐标数D，得到mean_log_ratio。
proxy_ratio=exp(mean_log_ratio)明确是代理比值，不是真实高维联合概率比。

四步分别执行标准正负优势相关裁剪，同一上层GAE优势在四步共享；总loss是负的
四步目标之和除以4*N_global。D=0项贡献0，但不改变4*N_global分母。gamma_denoising
固定1.0，不额外添加内部折扣；前15步冻结高斯与最后确定性步不进入可训练目标。
已知前缀和padding全部排除，其余60帧自由参考完整参与，未按0.5秒裁掉远期概率。

第一遍无梯度保存[N,4,8]统计及独立裁剪分支；第二遍在同一Actor版本逐微批重算。
对joint_log_ratio的系数为-A*proxy_ratio/(4*N_global*D)，真正裁剪平坦区及D=0为0。
ConditionGraphCache保持条件图汇总反向，所有rank梯度SUM后不再除world_size，不同时
保留四份Transformer反向图。小规模完整Autograd和FP64独立Normal公式负责数学验收。

统一目标验收与优化器事务
------------------------

from_records保留[N,4]步骤轴；stepwise_surrogate、stepwise_diagnostics、
grouped_objective与流式反向共用相同公式。整批目标在相同不可变真实行为rollout上，
按四步各自裁剪后平均；objective-aware回溯只按这个新代理目标比较候选。
来源、优势正负、KL尾部分组均使用每链四步目标均值，尾部分组不影响训练样本。

所有候选（包括KL-only对照）覆盖全部有效旧链的四步。每次回溯从同一Actor、
非空多组Adam、控制器和RNG起点执行，固定模块倍率同时缩放；候选日志记录执行的
Adam尝试、候选LR、真实KL、代理目标及接受/拒绝原因。有限门限拒绝只丢弃本次候选，
保留此前已接受更新，恢复本次BC游标和RNG；真正数值异常仍严格拒绝并交外层事务。
新rollout重新从名义LR试探，Adam动量及参数组倍率保留。

KL量纲与配置
------------

新配置为configs/closedloop/stage2_stepwise_mean.yaml。clip初值0.01，可配置；
它是待真实数据校准的候选，不称为最优。

主要步长候选为mean_coordinate=0.0009、maximum_step_mean_coordinate=0.0018、
p95_coordinate=0.0021。这里coordinate_kl的P95是各内部转移“自由坐标平均KL”的
P95，不是所有标量坐标KL的P95。每步joint KL与mean-coordinate KL均单独记录；
真实chain_joint_KL是四步joint KL之和，完整记录mean/P95/P99/max，绝不除维度。

四个旧联合KL硬门限没有照搬。新配置将chain_joint_mean/p95/p99/max显式设为null，
calibration.joint_tail_status=uncalibrated_diagnostic_only。它表示联合尾部保护尚未
标定，只作独立诊断；所有真实链仍完整计算，NaN/Inf检查始终有效。若以后显式启用
联合保护，必须配置独立正值、有序门限及明确候选/实测状态，不与坐标阈值混用。

TensorBoard保留70个精简标签：显示主控坐标KL、独立链KL、proxy clip/active clip、
新目标增益和实际Transformer更新；全参数RMS转入JSON，不增加面板页数。每次候选
及各来源/各步骤的详细分布留在JSON/JSONL，指标单位在统计字段中明确标识。

新实验与旧资产
--------------

默认CLI配置切为新配置。历史联合配置仍可只读解析，用于追溯或采样/评估；当前
Actor训练入口拒绝旧目标，不会把旧配置静默改成新算法。历史checkpoint的磁盘容器
schema字符串保留，算法由identity.training中的目标/归约字段识别，不能仅凭旧
schema名称推断当前目标；普通resume仍不能跨目标混用。

从旧90轮显式继承到新算法时，应在新目录使用
--initialize-from-checkpoint和--initialize-stepwise-ppo。该入口仅允许目标、clip、
对应KL/LR控制量纲变化，Actor/Critic、两套Adam、各rank RNG和原采样游标仍继承，
真实概率核、奖励、Value、输入、物理时钟、组倍率和初始学习率继续核对。重新采集
行为链，禁止把旧rollout当成新on-policy采样；源码版本只记录，不作门禁。
本轮已用该入口重新采集有限验证数据；没有从它启动新的正式长期训练。

验证状态与限制
--------------

数学与训练链路的本地回归、服务器固定真实数据LR试探及最终验收结果见下文。
不以PPO目标增加、proxy clip较小或KL通过宣称物理跟踪已改善。
同条件完整音乐的PhysX配对收益不属于本轮已证明结论，也不自动启动长训。

已完成的接口与数值验收
----------------------

本地核心110项通过、3项CUDA专用跳过；服务器1随后独立执行这3项，全部通过，
覆盖真实CUDA非空Adam回溯、完全拒绝后的Actor/Adam/控制器/RNG恢复，以及CPU统计
与GPU流式反向的逐张量相等。新增最终整批审查确认objective-aware比较的是新目标，
正常零更新仍可进入下一物理采集轮；参数/Adam一致性沿原有周期检查。

扩大到DPPO全目录时修正了三类遗漏：旧跟踪奖励测试仍要求已取消的活动度门控为0，
仅更新测试为当前门控1，生产奖励不改；旧概率归因梯度参考仍用四步联合裁剪，改为
四步独立坐标平均目标；新算法名使GMT跳过每rank资产目录隔离，导致首次八卡采集
在接触传感器初始化失败。GMT只将资产隔离条件与算法版本名解耦，不改任何控制逻辑。
该接口覆盖旧/新/未来算法名的纯CPU配置30项通过，修复后的真实采集也已成功。

CPU Gloo测试清理曾在destroy_process_group等待，faulthandler原始堆栈保留。
测试清理改为先显式回收，再避免自动GC在Gloo销毁时重入；不改变生产通信、容差
或PPO计算。最终“多rank及空rank+回溯边界+逐步数学”组合50项通过。逐文件66个
测试及明确修正重跑的合并结果见本地cpu_final_status.json，不把重复核心测试加总。

最后全仓调用检查还补齐两个独立工具：validate_stage2_full_chain.py保留历史文件名，
独立Normal完整Autograd参考、clip、坐标KL、多组Adam及BC关闭均改为当前目标；
attribute_stage2_actor_change.py选取高KL/高ratio样本前按四步求和，避免[N,4]步骤
索引冒充链索引。所有候选最终验收复用生产_final_actor_acceptance。
TensorBoard读取历史clip_fraction时不再冒充新proxy_clip_fraction。

本地66文件最终合并为731 passed、7 skipped；服务器另通过其中3项关键CUDA测试，
其余4个跳过项没有冒充通过。最后工具/归因/面板组合58项通过，不重复计入731。
源码变化仅记录；没有新增源码相等门禁或ESS硬门槛。

服务器另用当前完整模型和同一真实行为起点，八rank各取2条、全局16条链，保留
四步完整Autograd图，以独立FP64 Normal公式核对流式梯度。微批1/3/8的全局SUM
梯度相对L2误差分别3.212007e-7、4.468299e-7、4.816544e-7；最大单坐标误差
分别8.568168e-8、9.778887e-8、8.102506e-8，全部通过原2e-5门限，未放宽。
该次--gradient-only退出码0，不执行Adam或物理仿真，完整19随机步零更新审计
也再次通过。证据为服务器analysis/real_gradient/summary.json及本地
real_gradient_verified.json。

服务器1真实数据及有限LR对照
--------------------------

修复后采集为独立目录stage2_stepwise_mean_20261022_collect_retry，退出码0。
八卡实际取得14320条有效链；0～18全部19随机步骤的零更新概率审计通过：
log-prob差和ratio-1均为0，独立Gaussian公式最大差9.094947017729282e-13。
确定性第19步不具有Gaussian密度；冻结先验、采样核及已知前缀身份保护同时保留。
采集后的Actor更新数仍192、Critic仍24416、正式轮次仍90；rollout计数额外增加1。

相同不可变case、Actor、非空Adam、控制状态和RNG，batch=8192、epoch=1、clip=.01。
14320条链形成8192和6128两个小批次；每次候选检查全部14320条链，不删除尾部。
各基础LR保持Transformer/other倍率.1、qpos输出倍率1。结果如下，目标增益是
新逐步代理单位，不能与历史联合PPO目标的数值直接比较：

=========== =========== =============== =============== =============== ===============
基础LR      接受/计划   代理目标增益    坐标平均KL      真实chainKL均值 真实chainKL最大
=========== =========== =============== =============== =============== ===============
1e-6        2/2         5.428919e-6     2.893854e-6     0.01460817      0.16877164
3e-6        2/2         1.731972e-5     2.612803e-5     0.13189430      1.52677765
1e-5        2/2         7.280530e-5     2.946899e-4     1.48759476      17.26399238
=========== =========== =============== =============== =============== ===============

三候选均普通proxy clip=0、active proxy clip=0；因此这批数据没有验证clip=.01的
饱和区，也不能说.01最优。全部候选都满足当前坐标KL候选阈值，联合尾部门限显式
未标定，仅诊断；1e-5的真实chain KL尾部明显变大，不能据此宣称物理安全。

每个候选做6次实际Adam试算（每步比较1、1/2、1/4三种LR），最终保留2次；其余
4次为选优恢复的对照候选，不是额外有效更新，也不是KL超限拒绝。本次所有试探
都数值/KL/目标合格，选中的两步都采用该候选名义LR，没有被迫降至1e-8或1e-9。
三候选流式重算误差均0，八rank Actor和Adam最终完整状态一致，两步的16个
Transformer层均有实际非零FP32参数变化；BC计算样本数及梯度均0。

96个普通/高KL/高ratio固定噪声探针显示最终关节参考RMS变化中位数依次为
0.00015487、0.00046251、0.00150774 rad，全部保护坐标逐值不变。它是有意覆盖
尾部的反事实生成探针，既不是全训练集随机均值，也不是GMT机器人实际跟踪误差。
每候选Actor优化、全rollout检查、探针及保存约183～184秒，PyTorch峰值约27.05GiB；
该时间不含模型加载和初始概率审计，更不能当成一次带PhysX训练轮的耗时。

四数据来源的整批代理增益都为正；但3e-6和1e-5的正优势组分别下降约4.40e-6和
1.2621e-4，1e-6的正负优势组均提高。没有因为更大LR整体代理值较高就修改默认：
新配置仍使用1e-6，坐标KL候选与联合尾部均未通过长期物理标定。

证据与后续边界
--------------

本地证据根为outputs/analysis/stage2_stepwise_mean_20261022，包含
cpu_final_status.json、collection_verified.json、fixed_lr_verified.json及真实
测试日志。服务器完整case位于::

    /data1/user/liwei/GENMO_outputs/closedloop_stage10/stage2_stepwise_mean_20261022_collect_retry/sessions/17d415b0-01f2-48f4-b5c8-a5ae9daf5d66/fixed_data_validation/iteration000091

服务器原始候选与日志位于同输出根的stage2_stepwise_mean_20261022_analysis/fixed_lr。
这些是本任务要求的真实数据验收证据而非临时pytest产物，保留供用户复核；所有
本轮pytest临时目录均另行清理。第一次资产初始化失败日志也保留，不抹除失败证据。
累计第90轮正式模型与旧训练目录原样保留，采集case的iteration000091不代表正式
训练已接受第91轮。不会自动启动200轮，物理跟踪收益留待同条件PhysX实验确认。
