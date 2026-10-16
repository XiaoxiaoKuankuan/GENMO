实验分支适配说明（2026-10-16）
====================================

本分支将随机起点20轮实验的训练、固定整曲评估和MC写入同一个run。
读取真实 ``trainable4_kl`` 的15～18四步，冻结前15步和确定性末步不进入KL分母。
八项Tracking误差严格对应GMT原始平方误差，新增身体线/角速度、脚高、十二腿关节RMS。
``Loss/joint_score`` 为联合score策略梯度，不画虚假的PPO实际clip曲线；原始日志仍保存
反事实坐标DPPO clipping诊断。``Policy/learning_rate`` 和 ``Policy/backtracks``
分别展示实际接受学习率和回溯次数。以下P1_500数值仅为来源分支历史说明，不能解释为
新奖励、新Critic、新采样方案的评估结果；当前最多54个指标定义，缺测不伪造零点。

第二阶段精简 TensorBoard：看训练有没有效果
==========================================

本次改动只调整监控、历史重建和说明，不修改奖励、优化器、选模规则或模型。
旧P1运行包含882个scalar标签，递归展开了预算、会话和内部诊断；原TensorBoard又
扫描整个历史实验根目录。新入口使用固定白名单，最多50个标签、8个分组，
一个任务只有一个run。当前49条实际曲线，训练和验证总回报各占一条。
当前音乐奖励关闭，Reward/music不画恒零曲线。详细原始证据仍保留在JSON/JSONL。

如何打开
--------

新训练默认写到该运行的 ``tensorboard_compact/``。只选择需要看的这一运行目录，
不要把整个 ``closedloop_stage10`` 根目录作为logdir，否则旧实验仍会出现在页面中。
旧500轮的重建命令（输出必须尚不存在）::

    python -B tools/eval/export_stage10_tensorboard.py \
      --run-dir /data1/user/liwei/GENMO_outputs/closedloop_stage10/formalization_20261014/pilot_P1/P1 \
      --output /home/user/liwei/stage10_tensorboard_compact/P1_500_20261016_single/P1_500

本地下载后的查看命令::

    /home/weili/GENMO/.venv/bin/tensorboard \
      --logdir /home/weili/bumi-closedloop-worktrees/GENMO/outputs/tensorboard_compact/P1_500_20261016_single \
      --host 127.0.0.1 --port 16008

页面只显示 ``P1_500`` 一个run，训练、固定验证和完整MC都在同一任务页面。
``Reward/train_total_per_second`` 是普通训练采集回报，
``Reward/validation_total_per_second`` 是固定验证回报，不能拼成同一条曲线。
其余指标仍按八个功能组排列，来源和采样频率见下面的逐项定义。
横轴统一为累计接受的Actor迭代，Stage1基线为0；
预热、被拒绝的更新仍留在审计记录，不冒充新增已接受训练轮次。
每条曲线都保留原始点，程序不做平滑、不删不利点；TensorBoard界面可选择轻度
平滑辅助阅读，但判断改善应查看原始固定验证点和配对区间。

先看哪些曲线
------------

1. 选择唯一的任务P1_500：看Reward/paired_gain_per_second及两个CI95边界。收益应为正，
   区间下界大于零才支持该固定验证集的明确收益。该区间不含独立重训随机性，
   反复选最好的checkpoint也会引入选模偏差。
2. 看Tracking的根位置、根朝向、关节角和根位置P95。整体漂移变差时，不能用
   相对身体姿态改善或总回报上升宣布跟踪全面改善。
3. 看Motion/activity_retention，1为Stage1水平。持续下降提示活动减少，不能把
   动得更少得到的稳定性收益直接认定为舞蹈质量提升。
4. 看Termination的物理失败和新增失败。window_limit是短验证正常行政结束，
   不是跌倒；music_end只表示数据可用曲段终点，10秒验证不等于整曲验收。
5. 有效性不明确时，再看Policy更新尺度、Loss/critic_mc_ev与critic_mc_bias。
   PPO loss在零附近、训练后Critic loss很低，都不能证明策略已经收敛。

训练轮数与还需解决的问题
------------------------

旧P1已接受500轮、约200万条上层转移，并非没有发生足够多的数值更新。固定短验证
第500轮每秒回报相对Stage1约-0.044%，最佳短验证在158轮；整曲每秒回报约+0.428%，
配对95%区间跨零。因此目前没有证据把缺少收益主要归因于轮数不够，也不能宣称
500轮已经充分收敛。普通训练的任务组成持续变化；先按真实秒数归一化并来源等权，
再对照固定验证，才能区分采样波动与学习方向。

优先级：

* 已修复随机音乐时间和配对运动状态，但旧500轮权重不含该修复；新训练需检查真实
  Actor执行覆盖，而非只看抽到的bin。曲线中的Sampling是来源时间份额，不是假装
  全局唯一覆盖；完整覆盖区间继续以checkpoint/collection审计为准。
* 奖励及选模需要单独约束全局位置漂移、限位越界尾部和动作收缩。当前身体项先
  做水平位置/yaw对齐；活动度达到配对目标一半即满门控，选模允许整体降到80%。
* 固定10秒验证不能识别约100秒后累积的错误；需同时保留固定多位置短窗与完整曲段
  验证，短窗best不直接当部署best。
* 独立完整MC仍有价值低估，必须把同批自举拟合和长期价值质量分开。不能因为同批
  EV接近1就认为优势估计准确；扩大训练前先检查有效优势和真实更新方向。
* 末4步更新较小、裁剪很少发生，只说明更新尺度保守。应在相同采集/评估条件下做
  有限预算的学习率或PPO/BC对照，不能凭loss接近零直接放大学习率。
* Isaac随机训练采样与MuJoCo确定性部署不同；部署效果需要同音乐/种子/起态的
  独立配对验证，并单列生成超期与参考保持，不让降级执行掩盖模型问题。

全部曲线含义
------------

奖励分项当前来自保存完整逐控制步证据的固定验证。旧普通采集没有将全量分项
保存到metrics，重建时不根据稀疏审计补造训练分项。Reward/train_total_per_second
为每来源总回报/实际秒数再平均，Reward/validation_total_per_second为每任务
每秒回报先来源内平均再来源等权；训练与验证任务分布不同，不比较两者绝对值判断过拟合。
固定验证的奖励分项使用相同来源/任务权重，分项和与总回报之差可含事件惩罚。

Reward
~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 标签
     - 含义与读法
   * - ``Reward/train_total_per_second``
     - 分/秒，普通训练采集各来源积分回报/实际采集秒数再来源等权；任务不断变化，只作训练监视。
   * - ``Reward/validation_total_per_second``
     - 分/秒，固定验证各任务每秒回报先来源内平均再来源等权；优先用此曲线判断训练效果。
   * - ``Reward/episode_return``
     - 分，固定验证任务积分回报的四来源等权平均；执行更久也会增加，必须一起看每秒回报和终止。
   * - ``Reward/tracking``
     - 分/秒，已乘权重和活动度门控的跟踪奖励；越高越好，但不能替代原始误差。
   * - ``Reward/stable``
     - 分/秒，根高和非 yaw 姿态稳定奖励，越高越好。
   * - ``Reward/alive``
     - 分/秒，有效存活控制区间奖励，越高越好。
   * - ``Reward/command_penalty``
     - 分/秒，控制指令变化惩罚，非正，越接近零越好。
   * - ``Reward/torque_penalty``
     - 分/秒，力矩惩罚，非正，越接近零越好。
   * - ``Reward/contact_penalty``
     - 分/秒，滑脚和异常接触惩罚，非正，越接近零越好。
   * - ``Reward/joint_limit_penalty``
     - 分/秒，实际及参考关节限位惩罚；不能当作硬限位最大越界距离。
   * - ``Reward/music``
     - 分/秒，仅音乐奖励启用时记录；奖励关闭时不画恒零曲线。
   * - ``Reward/paired_gain_per_second``
     - 分/秒，同一固定任务相对 Stage1 的来源等权回报差，正值才是收益。
   * - ``Reward/paired_gain_ci95_low``
     - 上述收益的音乐组配对 bootstrap 95%区间下界；下界大于零才支持该验证集的明确收益。
   * - ``Reward/paired_gain_ci95_high``
     - 上述收益的音乐组配对 bootstrap 95%区间上界；区间跨零表示提升不明确。

Tracking
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 标签
     - 含义与读法
   * - ``Tracking/root_position_rms_m``
     - 米，世界坐标根位置误差 RMS，越低越好；来源内按控制时间平均平方误差，再来源等权开方。
   * - ``Tracking/root_orientation_rms_rad``
     - 弧度，根部完整姿态角差 RMS，越低越好；包含 yaw。
   * - ``Tracking/body_position_rms_m``
     - 米，GMT 水平位置/yaw 对齐后的身体位置 RMS；不等于蓝色影子的全局位移。
   * - ``Tracking/body_orientation_rms_rad``
     - 弧度，GMT yaw 对齐后的身体姿态 RMS，越低越好。
   * - ``Tracking/joint_position_rms_rad``
     - 弧度，关节角误差 RMS，越低越好。
   * - ``Tracking/joint_velocity_rms_rad_s``
     - 弧度/秒，关节速度误差 RMS，越低越好；当前奖励权重为零，属于诊断。
   * - ``Tracking/root_position_p95_m``
     - 米，来源等权的逐控制步根位置误差第95百分位，越低越好，用于观察尾部漂移。

Motion
~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 标签
     - 含义与读法
   * - ``Motion/activity_rad_s``
     - 弧度/秒，实际机器人0.5秒因果窗关节速度 RMS 的来源等权均值；不是越小越好。
   * - ``Motion/activity_retention``
     - 相对同一固定任务 Stage1 实际活动度的比例；1为持平，持续下降提示动作收缩。
   * - ``Motion/music_alignment``
     - 实际动作与音乐的节拍同步诊断，越高越好；不是当前训练奖励，也不是完整舞蹈质量评分。

Termination
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 标签
     - 含义与读法
   * - ``Termination/physical_failures_per_1000s``
     - 次/1000机器人秒，普通 Actor 采集物理阈值失败次数/实际采集秒数；不把每次失败都称为摔倒。
   * - ``Termination/physical_failure_fraction``
     - 固定验证物理阈值失败任务数/全部任务数，越低越好。
   * - ``Termination/window_limit_fraction``
     - 固定验证因行政时间上限结束的任务比例；正常短窗结束，不是物理失败或整曲完成。
   * - ``Termination/music_end_fraction``
     - 固定验证真实到达可用音乐终点的任务比例；10秒短窗中的该比例不能当作整曲成功率。
   * - ``Termination/new_failure_count``
     - 与 Stage1 同任务配对后新增的失败数，越低越好；不能被另一些任务恢复成功抵消。
   * - ``Termination/reference_rejections_per_1000s``
     - 次/1000机器人秒，固定验证参考拒绝次数/实际执行秒数，越低越好。

Policy
~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 标签
     - 含义与读法
   * - ``Policy/learning_rate``
     - 实际 Actor 优化器学习率，不是配置上限。
   * - ``Policy/trainable_kl``
     - 本轮更新前后可训练去噪步骤的平均每自由坐标 KL；排除冻结步骤，大小本身不等于收益。
   * - ``Policy/max_step_kl``
     - 完整去噪链中最大的单步平均每自由坐标 KL，用于检查更新是否过大。
   * - ``Policy/clipped_objective_fraction``
     - 可训练去噪步骤最终检查中真正被 PPO 裁剪的目标比例；接近零不单独证明无效。
   * - ``Policy/gradient_norm``
     - Actor 每次更新裁剪前 PPO+BC 总梯度范数的均值；持续异常增大需检查。
   * - ``Policy/bc_to_ppo_gradient_ratio``
     - 每次更新加权 BC 梯度范数/PPO 梯度范数的均值；不是 Adam 参数更新贡献比例。

Loss
~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 标签
     - 含义与读法
   * - ``Loss/ppo``
     - Actor 本轮 PPO 代理损失按内部样本访问数加权；优势归一化后常在零附近，不能按单调下降验收效果。
   * - ``Loss/bc``
     - Actor 本轮 BC 损失按实际 BC 样本数加权；衡量监督约束拟合，不等于物理跟踪。
   * - ``Loss/critic_mse_before``
     - 本轮 Critic 更新前，对本轮自举目标的 MSE；不同轮目标会变化。
   * - ``Loss/critic_mse_after``
     - 同批 Critic 更新后的 MSE；只说明拟合程度，不能代替独立 MC。
   * - ``Loss/critic_ev_before``
     - 本轮更新前自举目标解释方差，1较好、0接近只预测均值、负值较差；不是无偏真实价值精度。
   * - ``Loss/critic_mc_ev``
     - 独立完整轨迹 MC 上最终选定 Critic 的解释方差；该集合参与候选选择，不是完全未使用的最终测试。
   * - ``Loss/critic_mc_bias``
     - 独立完整 MC 上预测均值减真实回报均值，分；接近零较好，负值表示低估。
   * - ``Loss/stage1_validation``
     - 固定 Stage1 监督验证集来源等权损失，越低表示原监督任务拟合更好。

Sampling
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 标签
     - 含义与读法
   * - ``Sampling/time_share_AIOZ-GDANCE``
     - 本轮普通 Actor 实际采集秒数中该来源的占比；不是选曲概率，也不是唯一时间覆盖率。
   * - ``Sampling/time_share_AIST++``
     - 本轮普通 Actor 实际采集秒数中该来源的占比；不是选曲概率，也不是唯一时间覆盖率。
   * - ``Sampling/time_share_FineDance``
     - 本轮普通 Actor 实际采集秒数中该来源的占比；不是选曲概率，也不是唯一时间覆盖率。
   * - ``Sampling/time_share_Mine``
     - 本轮普通 Actor 实际采集秒数中该来源的占比；不是选曲概率，也不是唯一时间覆盖率。

Perf
~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 标签
     - 含义与读法
   * - ``Perf/iteration_seconds``
     - 秒，本次接受迭代墙钟，不包含迭代之间另做的周期评估/MC/保存开销。
   * - ``Perf/controls_per_second``
     - 实际 Actor 采集控制步总数/本轮墙钟；用于判断吞吐，不是策略质量。

本次旧500轮的交付结果
---------------------

原始882个标签最终展示为49条曲线、8组、一个任务run。训练指标各有500个迭代点，
验证指标各有37个点（0至500），MC指标各有97/193/289/384/480轮五个点。
单run版本从上一版已下载并逐文件SHA核验的精简JSONL重排，逐点核对数据和横轴，
不重新评估模型、不平滑、不补零。来源摘要和delivery_verification.json保留核验结果。

本机查看地址为 http://127.0.0.1:16008 ，服务名为
``genmo-stage10-single-run-tensorboard-20261016.service``。该服务只读取上述本地
精简目录。服务器原6007历史监控保持原状态，原训练、模型和事件文件不改。

重建后的固定短验证：每秒回报3.426497到3.424977（-0.0444%）；活动保留0.959159；
根位置RMS从0.089009到0.098738米（+10.93%），根位置P95从0.164820到0.187142米
（+13.54%）；Stage1监督验证损失从4.325126到3.357998（-22.36%）。配对每秒收益
为-0.001520，2000次音乐簇bootstrap区间[-0.019248,+0.016881]，跨零。短验证
158轮最高，每秒3.449897；这个选中值不代表无偏的部署质量收益。

按100轮汇总的来源等权普通训练每秒回报为3.1831/3.2026/3.1822/3.2139/3.1902，
大体横盘；不同轮采样不同，此曲线只能监视，不能替代固定任务的配对结论。
第480轮完整MC解释方差0.799344，平均低估10.815563分。以上均由原报告重算，
未用显示平滑制造趋势，也不把本次监控改造说成模型质量已经改善。

本轮85项CPU测试通过，包括真实事件目录只发现一个run、同一轮训练/验证回报
隔离和稀疏MC保留检查。HTTP逐项确认P1_500唯一run的49条曲线均有有限数据，
训练/验证/MC分别19/28/2条，每条分别500/37/5点；新旧事件的数值、step和wall_time
逐点相同。浏览器已实际刷新验证唯一run和原空白的Loss/MC图，并关闭忽略离群值
缩放以显示完整纵轴范围。精简曲线和来源核验文件作为交付保留。

为什么上一版有空图
------------------

上一版把一个任务拆成training、validation、critic_mc三个run。实际浏览器只勾选
validation时，页面仍保留全部48张图卡，其中20项只属于另外两个run，所以没有
曲线。HTTP逐项检查确认数据并未丢失：19项训练曲线各500点、28项验证各37点、
2项MC各5点，均有有限数值。单run版本把它们汇入唯一任务，恢复训练和MC图卡。
MC只有5个点、固定验证只有37个点是原始记录频率，不是缺失500轮数据。
刷新后如地址栏仍带旧的runSelectionState，应打开无旧筛选的16008/#scalars链接。
