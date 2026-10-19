当前第二阶段：完整链联合PPO
==========================

开发基线
--------

GENMO固定分支feature/bumi-music-closedloop，GMT固定分支
feature/bumi-frozen-gmt-backend。唯一配置为configs/closedloop/stage2_full_chain.yaml。
本次参考SCRIPT第4.3节、附录B.3和E：https://arxiv.org/html/2605.22894v3 。
论文的链概率比与PPO裁剪形式用于数学参考；其Flow Matching、学习噪声、Teacher
Velocity Anchor和文本奖励不引入当前系统。本文最后保留此前方案的历史结果，不能
把历史结果当作本次算法的验收结论。

概率与梯度
----------

固定20步生成：0～14用冻结Stage1，15～18为可训练高斯转移，19为冻结Stage1确定性
末步。已承诺前缀和padding从概率中排除。实际标准差仍来自原DDIM核的base std及
既有std floor，0.04/0.06/0.08/0.10是四个可训练步的下限，不能冒充全部实际std。

每条链的log ratio在FP64中对四步全部自由坐标求和，ratio=exp(sum log ratio)。
每条链对应一个上层GAE优势，只做一次PPO clip。没有除以自由坐标数，也没有平均
四个独立PPO目标；gamma_denoising=1。GAE gamma=.99、lambda=.98保持原定义。

第一遍无梯度得到完整链的裁剪分支和导数系数，第二遍逐微批重算四步联合log ratio
并反向。条件编码器梯度累计后统一反传，模型参数在两遍间禁止变化。loss只除以
全局实际有效链数，跨卡梯度SUM，不再除world size或四步。独立完整Autograd和
joint-score梯度用于检查首次未裁剪更新；后者不再作为生产训练配置入口。

高斯比值沿用旧标准化残差、log1p/expm1稳定公式，不直接减两个巨大log probability。
裁剪分支先在log space决定；不可表示的ratio、非有限值或下溢必须显式拒绝并回滚，
不截断真实ratio、不删除异常样本、不修改旧行为数据。每轮记录chain log ratio、
ratio分布、普通clip与active clip、联合KL、ESS和非有限比例。

首个候选与联合KL
----------------

* Actor AdamW统一LR=1e-6，clip=.10，epoch=1，全局minibatch2048条完整链。
* 八卡×128环境×16决策名义容量16384链，每epoch最多8次更新；实际有效链数决定
  次数。epoch=2时最多16次，需先通过有限实测。允许后续独立研究3或5个epoch。
* 每次BC全局至少256样本。epoch=1名义2048样本/轮，epoch=2为4096，尾批按实际
  计划更新次数分配；原Stage1RetentionAnchor、qpos/contact/已知前缀监督保留。
* BC/PPO共同参数梯度范数目标比例初值.1；记录全局范数、余弦、模块贡献、合成
  梯度裁剪和真实Adam参数RMS变化，不能把BC权重解释成Adam更新贡献。
* 初始候选soft chain KL=.10；整轮hard chain KL mean/P95/P99/Max=.20/.60/.90/2.0。
  这些是待真实rollout校准的实验阈值，不继承旧coordinate门限，也非物理安全证明。
* active clip达到.95或ESS比例低于.05停止继续使用本轮旧数据。每个minibatch只
  看四个随机步；整轮最终KL必须覆盖所有有效链。冻结步骤用参数/核/条件身份检查，
  启动及每100轮保留完整概率审计，不在每个minibatch重复20步网络前向。
* 整轮拒绝完整恢复Actor、Critic、两套Adam、BC采样器、BC控制器和RNG；实际物理
  交互和优化尝试预算不退回。拒绝原因、学习状态恢复证据及耗时必须保存。

configuration会自动生成优化身份摘要；用户不需要手算SHA。新训练身份为
full_chain_joint.v1，完整checkpoint为full_chain_joint.checkpoint.v1，旧优化器
运行不允许透明续训。Actor/Critic/奖励/噪声/物理时钟的既有行为仍单独绑定身份。

保持不变的任务
--------------

Stage1网络、s595000权重、GMT冻结权重、Tracking90 v2非对称Critic结构与观测、
八项Tracking及其他奖励定义和权重均保留。音乐奖励仍关闭；50Hz控制、200Hz物理、
0.5秒重规划和60帧/2秒参考保持。随机音乐位置、配对运动初态及跨轮连续物理episode
保留；完整MC预热和独立价值验证保留。新实验的整曲评估使用每次新建评估世界，
避免把驻留世界历史状态差异误认成Actor收益。

验证与运行
----------

服务器1上先执行数值/梯度测试，再用已有不可变真实rollout运行唯一标定工具
 tools/validate_stage2_full_chain.py。该工具不启动物理仿真，所有候选从相同
Actor/Adam/BC/RNG起点比较；旧rollout的GAE仅用于同数据对照，不冒充当前CriticMC。

有限闭环统一入口::

    bash scripts/train_stage10_8gpu_server1.sh \
      --config configs/closedloop/stage2_full_chain.yaml \
      --output-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/独立有限实验目录 \
      --stop-after-iteration 3

必须先完成真实联合KL及有限物理验收；不能因optimizer.step执行或KL较小就宣称
运动质量提高。固定完整音乐对照同时看总回报、各奖励项、完成率、失败、世界根部
误差、局部身体/关节误差和活动度，保留全部失败任务。长期2000轮只是配置容量，
本次有限命令不自动启动长期训练。

一个任务一个TensorBoard目录。曲线按Reward/Tracking/Motion/Termination/Policy/
Loss/Sampling/Perf分组；新增联合KL、clip/active clip、ESS、参数RMS、接受/计划更新数
及PPO/BC/Critic/KL/采集耗时。逐minibatch分位数、模块贡献、异常和输出探针留在JSON，
不递归展开成大量TensorBoard页面。root输出变化包含位置与姿态；同噪声参考变化
是生成侧诊断，不能替代GMT实际执行。

此前坐标代理PPO基线的历史验证
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
