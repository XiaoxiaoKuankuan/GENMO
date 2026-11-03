# 连续12帧预览启动与执行机械限位约束

本次改动遵循用户2026-11-03的新要求：预览给定原始连续12帧，物理执行Loss增加机械
关节限位约束。这里记录输入范围、梯度、监控和断点边界；不会因为修改磁盘源码就宣称
已经改变运行中的Python训练进程，也不把数值导数通过当成真实Newton整曲通过。

## 预览启动

`preview_joint_physics_splits.py generate`显式传入原始动作的`qpos[:12]`，通用生成器新增
`startup_qpos`，只接受恰好`[prefix_frames,28]`且有限、单位四元数的输入，不能与单一
`initial_qpos`同时使用。原始第12帧之后的未知动作不传给生成器。

第0～11帧是给定动作，保持原30Hz时间轴；从第12帧开始才是学生生成。仍执行完整
DDIM20/CFG2.5/eta0和60帧窗口。启动接触logit仍是无效占位。通用工具没有动作示范时
保留原站姿启动能力；本次只改变有明确原始前缀的联合模型预览。

轨迹和manifest记录`original_continuous_first_12_frames`、给定12帧、保持时间0、首个
生成帧12，页面说明同步标注“给定连续原始12帧”。原有预览、模型及视频不覆盖。
真实Newton入口继续从同一连续参考经原ReferenceBuffer计算姿态和速度后reset一次，
不额外强制清零速度，不复制首帧填充0.4秒。现有严格硬限位、SM终止和接触检查保留。

## 执行限位公式

权威上下限来自冻结SM的`kin.q_lo/q_hi`，采用SM关节名称和顺序。对WM预测执行状态
中的关节角`q`定义：

```
hard_excess = relu(q_lo - q) + relu(q - q_hi)
margin_excess = relu(q_lo + margin - q) + relu(q - (q_hi - margin))
u = (margin_excess / scale)^2
L_joint = mean_over_joints(u) + max_over_joints(u)
L_exec_step = 原执行跟踪损失 + weight * L_joint
```

正式新增配置使用`margin=scale=pi/180`（1度）、`weight=5`。最坏关节项防止踝关节的
越界被21关节平均稀释；窗口仍按有效步数/折扣质量归一化，H32和有效短窗不因步数
变化改变名义尺度。限位项包含在同一个L_exec内，因此现有约30%全局参数梯度控制器
衡量的是“跟踪＋限位”合计梯度。原Stage1 v5限位监督继续保留。

预测参考、SM动作、WM状态均保持原值及输入梯度，不进行关节clamp、GT替换或新增
Actor/WM优化器。有限权重的可微惩罚不能数学保证零越界；严重非法状态门禁用于拒绝
不可信动力学，真实Newton机械限位仍严格检查，不因新Loss放宽。

## 监控

日志新增`execution_joint_limits`，只统计本日志区间已接受WM执行步骤：

- `window_violation_rate`：至少一步越界的窗口比例。
- `step_violation_rate`：至少一个关节越界的执行步比例。
- `joint_step_violation_rate`：关节×步的越界比例。
- `max_violation_deg`：最大超出机械限位角度；`per_joint`记录每关节次数与最大角度。
- `constraint_loss`：未乘weight和lambda_exec的限位项，按执行步平均。

计数阈值1e-4rad，与真实预览硬限位检查一致。统计经DDP按全局计数求比例，最大角度
用MAX归约；只有日志间隔执行两次紧凑归约，不逐步搬运全部状态到CPU。被拒绝窗口
继续在原拒绝统计中，不能把“已接受WM步骤”当成全部尝试或Newton实测。

## 配置与续训

新配置为`configs/closedloop/joint_physics_formal_200k_joint_limits_8gpu_server2.yaml`，
继承现有gradshare30八卡配置。旧配置默认不启用新增限位项，便于已有任务保持原目标。

从已有完整断点增加此项必须使用`--continue-from`并写全新目录，不能称作相同目标的
`--resume`。允许的合同差异只有新增`joint_limit_constraint`；模型、AdamW、scheduler、
全部rank数据/噪声游标、RNG和global_step保留。原30%课程进度保留，在首次物理更新
重新测量梯度，清除旧执行梯度EMA，避免直接把旧权重测量套到新目标。之后的新断点
可正常严格`--resume`。

旧Newton报告只在显式diagnostic模式下继续提供同资产/同H的历史支持域诊断，日志
`execution_objective_newton_certified=false`，不能宣称它认证了新增限位目标。严格认证
入口仍要求整个执行目标一致。新增限位不触发Newton全库预处理。

## 已实施验证与边界

测试覆盖真实原codec/小型原Transformer的完整DDIM20连续前缀、机械上下限的梯度方向
与有限差分、单个最坏关节、H32计数、目标迁移合同，以及完整断点的下一批次/AdamW
更新复现。冻结SM/WM接口测试必须使用正式训练对应的xi25旧bridge快照，不使用共享
目录后来升级的xi119接口；共享源码和资产都不修改。

H32接口测试单独验证新增限位梯度到达学生x0、GENMO主干和FutureEncoder，且与关闭
约束时的预测状态逐位相同；该测试明确使用张量夹具，不能当作真实学生Newton成功。
本次没有中断正在运行的八卡任务，也没有启动重复正式任务。
