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

若后续明确授权从旧90轮继承到新算法，应在新目录显式使用
--initialize-from-checkpoint和--initialize-stepwise-ppo。该入口仅允许目标、clip、
对应KL/LR控制量纲变化，Actor/Critic、两套Adam、各rank RNG和原采样游标仍继承，
真实概率核、奖励、Value、输入、物理时钟、组倍率和初始学习率继续核对。重新采集
行为链，禁止把旧rollout当成新on-policy采样；源码版本只记录，不作门禁。

验证状态与限制
--------------

数学与训练链路的本地回归、服务器固定真实数据LR试探及最终验收结果将按实际完成
情况追加。现阶段不以PPO目标增加、proxy clip较小或KL通过宣称物理跟踪已改善。
同条件完整音乐的PhysX配对收益不属于本轮已证明结论，也不自动启动长训。
