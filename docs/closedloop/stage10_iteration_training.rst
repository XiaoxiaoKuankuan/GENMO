第二阶段正式训练：按累计 PPO 迭代停止
====================================

本说明只描述当前配置。2026-10-13的K8基线、旧预热及清理操作属于历史运行，
原记录见Git历史和根目录记录文本.md；既有模型及评估证据不因更新说明而删除。
本次实施、同条件对照和实际启动证据见stage10_formalization_20261014.rst。

运行结构和参数
--------------

配置入口为 ``configs/closedloop/stage10_server1.yaml``。Stage1 s595000 初始化
GENMO，GMT140500冻结，Isaac Lab提供真实反馈，DPPO＋BC联合更新Actor，Critic
使用独立冻结预训练条件编码器和256维、2层、4头时序Transformer。

* 8张GPU，每卡64环境。物理episode跨4秒共享世界采集窗口延续；正常环境不在
  PPO轮末重置。优化和评估期间暂停训练世界。恢复则重建PhysX并开始新episode。
* 实际生成60帧/2秒、30Hz，每0.5秒重规划；GMT50Hz、物理200Hz，四个真实子步。
* 行为扩散完整20步。独立Stage1副本冻结前16步，可训练Actor负责末4步，PPO
  使用这4步。它不是只在共享网络上选择末4步后声称前段冻结。最终安全检查仍
  覆盖全部20步，并要求冻结段KL为零。
* 噪声下限为 ``[0.001]*16+[0.08,0.10,0.12,0.12]``，eta=0.1，CFG=2.5。
  数值路径为stable_tf32x3/FP32与SDPA math，未采用BF16或compile。
* Critic学习率1e-4。PPO两epoch、全局1024链minibatch；Actor计算microbatch64，
  KL128。典型一轮8次Actor optimizer更新，BC512样本，动态梯度目标比例0.1。
* PPO clip=0.003；全20步平均坐标KL上限0.002，最大单步均值和P95上限均0.01。
  超限回滚整次Actor/Critic优化；真实采样和优化尝试仍计资源预算。
* Stage2音乐奖励关闭，音乐条件、CFG、Stage1音乐动作监督保留。tracking继续
  使用配对动作活动度门控，不能把不动换来的稳定误认为生成能力提升。

正式Actor学习率固定1e-7。两组100轮pilot及独立完整歌曲复核后选P1，自适应
学习率不采用。具体收益和动作活动度下降、参考极端越界等限制见本次验收报告。
正式阶段从P1第100轮连续恢复，不能清空已完成轮数、优化器或交互预算。

评估、Critic和归档
-----------------

每150万实际训练控制步，固定27个独立音乐组/54任务进行10秒开头与中段验证。
每卡常驻4环境评估世界，不与训练世界同时推进。配置中的eval_count是保留的
旧元数据，实际任务数由samples_per_source=all_unique生成；本次恢复不修改该
身份字段。完整test另用22组/44任务，从配对样本开头运行到music_end或真实失败。
“完整”指当前可用配对样本，不保证覆盖未裁剪的原始录音，也不保证Stage1从未
见过这些音频。验证集用于选型，test不用于反复调参。

当前Critic初始化来自新身份下48首独立训练音乐/96完整episode、32验证episode
的真实MC采集，96epoch与MC/TD比例经独立验证选择，初始化EV=0.670941。
严格绑定完整初始化路径及SHA；导出和导入都校验实际源码，不能改写旧SHA复用。
正式每1000万实际训练控制步用当前固定Actor重采完整MC验证；EV低于0.9时尝试
现有比例0/0.5/1。只有独立验证改善才接受，否则恢复原Critic与Adam，保留拒绝
候选指标。不能把被拒拟合的指标当成实际部署Critic，不能长期复用旧Actor的MC
作为当前策略无偏标签。短rollout正常bootstrap，真实终止不bootstrap。

日常完整扩散链、old statistics、条件、GAE和执行数据保留到更新与20步KL检查
结束再释放，不逐轮长期归档。严格证据每1000万控制步及异常保存，CPU/GPU奖励
首步和每1000步核对。checkpoint取得不可变CPU快照，单个在途任务异步写盘；
实际模型落盘并核验后才发布latest/best指针。latest可能落后于内存中的接受轮。

停止、恢复和使用
----------------

``--stop-after-iteration 500`` 是累计接受的PPO外层轮数上限，不是此次额外500轮。
预热与回滚不增加接受轮，仍累计实际资源。没有48小时或其他总墙钟期限；
SIGINT/SIGTERM安全停止、完整checkpoint、显存和磁盘保护、单次采集超时、参考
提交截止、推理延迟合同、物理失败终止都保留。资源上限为300万次生成、1.5亿
控制步、6亿物理子步和6000次Actor optimizer尝试，触及保护可提前受控停止。

在服务器GENMO工作树中，沿选中运行目录恢复：

::

    bash scripts/train_stage10_8gpu_server1.sh \
      --config configs/closedloop/stage10_server1.yaml \
      --output-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/formalization_20261014/pilot_P1/P1 \
      --resume latest --stop-after-iteration 500

不要把其他实验checkpoint放入该目录冒充恢复。新建运行必须用不存在的独立目录
并去掉--resume。恢复加载Actor、Critic、两套非空Adam、BC控制器、各rank BC游标
与随机状态、行为版本和累计资源；新建PhysX世界，不恢复未完成rollout。当前
iteration_horizon schema仍严格绑定实际源码、kernel、资产和完整算法配置，
同名schema也不代表不同身份的checkpoint可以混用。

部署测量64次中位0.24638秒、最大0.25851秒，采用0.30秒合同。它是单机器人
请求的独立部署预算，八卡批量训练前向不与此直接比较。GPU计算参数或数值核
改变时重新测量并使用新行为身份。启动过程仍有偶发Kit/Vulkan初始化卡住的
已知限制；600秒启动限时及原有受控重试保留，失败证据及总墙钟如实计入报告。

当前回归为988项通过、13项CUDA依赖跳过；另有真实八卡概率、非空Adam、冻结
前段、完整20步KL、实际60帧、连续物理与恢复验收。通过工程检查不等于证明
强化学习有收益，收益以同条件验证和完整歌曲物理结果判断。
