当前第二阶段训练
================

只保留一套配置和实现
--------------------

GENMO分支feature/stage2-asymmetric-denoising，配套GMT分支
feature/stage2-tracking-history。唯一第二阶段训练配置为
configs/closedloop/stage2_credit_server2.yaml，默认入口为
scripts/train_stage10_8gpu_server1.sh。脚本沿用原文件名，实际路径和资源配置来自
服务器2的当前配置；不会启动服务器1，也不会自动抢占已有GPU进程。

旧非对称候选、学习率扫描和随机20轮配置、坐标PPO训练路径、多线程配额采集器、
旧完整checkpoint恢复及一次性工程迁移脚本已删除。历史实现从Git获取，不在当前
工作目录保留另一套入口。Stage1和MuJoCo sim2sim是独立功能，其配置继续保留。

当前参数与执行
--------------

* Stage1 s595000初始化可训练Actor；独立冻结Stage1网络用于前15步和确定性末步。
* 第15～18步为真实高斯转移，标准差下限0.04/0.06/0.08/0.10。完整执行20步。
* 实际生成60帧、30Hz、qpos30＋contact2；每0.5秒重规划，已承诺前缀不能修改。
* 八卡，每卡128环境，一个共享PhysX世界连续采集8秒；健康episode跨轮保留。
* Actor统一学习率3e-7、一个epoch、全局有效行为链容量16384、计算微批128。
  四步joint-score求和按实际有效链数归一化，不取高维joint ratio指数。
* 原Stage1 BC每轮1024样本，动态目标梯度比例0.1；统一归约、裁剪和Adam回溯。
* Critic音乐/前缀编码器冻结，历史跟踪GRU独立训练1e-5，时序/Value层1e-4；
  完整MC训练和独立验证继续保留，gamma0.99每0.5秒，GAE lambda0.98。
* 八项GMT Tracking、配对动作活动门控、参考/实际限位0.7/0.3；音乐和独立稳定性
  奖励关闭，音乐条件、BC音乐动作损失、物理失败和事件罚分保留。
* 全部有效链的四步KL保持覆盖；门限0.0009/0.0018/0.0021，最多9次0.5倍率回溯。
  同一Actor/rollout的已接受完整检查允许验证身份后复用，失配即重算。
* GMT50Hz、物理200Hz，每控制四个真实子步；训练优化期间世界暂停。

配置使用当前continuous_joint_score身份，不再识别旧训练配置。优化参数与数据、
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

本次源码精简改变了源码身份，旧运行不能直接用新代码--resume，不能改SHA或
绕过加载检查。已有服务器2作业保持原绑定代码运行，仓库提交不会热更新该作业；
待另行切换到清理后代码时，用新运行身份完成必要的MC和物理核验。

诊断与证据
----------

唯一入口仍支持--evaluate-only、--evaluation-actor、--mc-validation-only、
--critic-preheat-only、--collect-only、--calibrate-only、--strict-evidence及
--verify-kl-reuse；这些是当前方案的检查模式，不是其他算法入口。

普通rollout在本轮优化及KL完成后释放；评估逐控制记录以单份无损压缩侧车保存，
异常/严格模式可保存完整证据。保留资源账本、显存/磁盘保护、参考截止和物理终止，
不设置48小时总训练期限。工程效率验收见docs/reports/stage2_efficiency_20261019.rst，
该报告记录对应历史实测提交；源码精简不自动继承新的物理学习收益证明。
