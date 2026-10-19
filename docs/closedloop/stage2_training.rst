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
* minibatch复用PPO前向的坐标平均KL，达到0.00075时丢弃当前未执行更新并停止。
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
