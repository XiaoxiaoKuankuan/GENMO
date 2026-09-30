# Stage9 奖励 v2：实际运动、配对活动度与执行代价

本版本落实用户指定公式，配置版本为 `stage9.execution_reward.v2`。仅替换奖励与补充
必需实际诊断；Actor 条件、采样核、Critic、GMT 控制频率、PD、动作缩放及终止阈值
保持原行为。旧 v1 验收结果仍是历史结果，不能据此声明 v2 已改善策略。

## 50Hz 积分和事件

```text
r = 0.02 * (2.5*gate*r_track + 2.0*gate*r_music + r_stable + 0.5*r_alive
            -0.15*c_cmd -0.10*c_torque -0.20*c_contact -0.50*c_joint_limit)
```

每个分数/连续代价在 [0,1]；门控单独记录，只有 track/music 使用它。组件保存
raw、normalized、score、weight、gate、weighted_rate 和 integrated_reward，另保存
总 reward_rate、reward、dt_s、版本、有效性和错误。执行失败一次 -5，有限非法参考
一次 -.5，不乘 .02。正常音乐结束/行政截断为 0。无法确认的 RPC、缓存/参考构造、
一致性故障保存 invalid 转移及原始证据，不产生策略惩罚，不进入训练 Buffer。

## 跟踪、稳定与 alive

全部误差直接读取 `backend.errors`，没有重新对齐 reference：

| 项 | 权重 | scale | backend 字段 |
| --- | ---: | ---: | --- |
| 关节位置 | .45 | .22 rad | joint_position_rmse_rad |
| 关节速度 | .25 | 1.40 rad/s | joint_velocity_rmse_rad_s |
| 末端相对高度 | .15 | .07 m | end_effector_relative_height_error_m |
| yaw | .10 | .60 rad | yaw_error_rad |
| 世界根位置 | .05 | .40 m | root_position_error_m |

每项先算 `exp(-(error/scale)^2)`，再按上表加权。稳定性为根高/.05 m 和 non-yaw/.12 rad
的相同指数分数，各占 .5；没有角速度或加速度项。完整有效控制区间 alive=1；真实
执行失败步 alive=0，并另记一次性失败罚分。alive 不能抵消或掩盖执行失败的标记。

## 配对活动度与音乐

`load_paired_activity(stage9.bc_data_root, sample)` 从当前音乐对应的 train 配对动作
读取关节角。严格核对选曲与 train manifest、payload、来源 SHA、music 身份和关节
顺序；示范不会进入在线条件 builder，音乐采样器也不需要读取动作才能采音乐。

30Hz 示范位置线性插值到 50Hz 控制区间端点，以两端差/.02 得到每区间速度；最近
最多 25 个区间×21 关节的 RMS 为 A_target。A_actual 为同窗实际关节速度 RMS。
初始不足 25 个区间时，两者使用同样的已有区间，`window_complete=false`；首区间
使用示范 t=0 和 .02 秒姿态，不填假零速度。沿用音乐 N/30 秒长度约定，尾部超出
最后姿态 (N-1)/30 的不足一帧明确保持末帧，记录 tail_policy 和保持区间数。

`source_motion_sha256` 是原始转换前动作身份，核对 payload 与 manifest 的声明；
另外计算真正读取的 motion.pt 的 `motion_file_sha256`，不混同两种 SHA。若配对
动作缺失、非有限、身份不符或窗长不一致，则明确报错/invalid，绝不用生成参考替代。

```text
gate = 1                                             if A_target <= 0.10
gate = min(1, A_actual / (0.5*A_target + 1e-8))       otherwise
intensity = exp(-(log((A_actual+.05)/(A_target+.05))/log(2))^2)
music = .7*beat + .3*intensity
```

beat 继续复用 `_derive_motion_beats` 与 `_beat_alignment`，来自实际机器人运动与
EDGE35 节拍字段。窗口不足或没有音乐拍点时，beat 明确无效为 0，intensity 可独立
有效；没有用 onset 饱和值替代活动强度。所有门控/窗口/混合参数集中写入配置。

## 执行器、接触和限位

- `c_cmd`：实际提交 PhysX 的 `_joint_pos_target_sim` 在相邻控制区间的差/.02，
  除实际关节限速，平方后 clamp 至 [0,1] 再求均值。reset 后首步为 0、valid=false，
  同时保留 first_step；不是引用 q_ref，也不按 .005 把目标跳变放大。
  有站立预热时承接预热最后一条真实目标，因此首个音乐步仍计算实际目标变化；
  只有确实没有上一目标的奖励流首步才返回无效零代价。
- `c_torque`：明确使用未裁剪 `ImplicitActuator.computed_effort` 的
  `pd_torque_estimate_nm`。每关节 `u=abs(tau)/effort_limit`、
  `h=clip((u-.8)/.2,0,1)^2`；子步成本为 `.5*mean(h)+.5*max(h)`，再对四子步平均。
  同时保存整个控制区间的 max torque ratio。不称硬件实测力矩。
- 机械功率仅诊断：每子步保存 `abs(tau_pd*dq)` 的逐关节/mean/sum/max；力矩在
  physics_tick-3、速度在 physics_tick，明确 `sampling_synchronized=false`。
  这是机械功率代理，不是电功率、能耗或严格同步测量；配置权重固定为 0。
- `c_contact=.7*slide+.3*bad_contact`。原任务允许左右脚和左右肘；bad contact
  阈值来自原 undesired_contacts 配置（1 N），足接触阈值来自原 sensor（10 N）。
  四子步各自计算后平均。只在足有接触且支撑球几何靠近地面时评价滑移；优先使用
  URDF 支撑球中心附近 `v_link+omega×offset` 的切向速度代理，尺度 .15 m/s。
  只有 ankle link 水平速度时命名为 slide_proxy。冲击峰值保留，奖励权重为 0。
- `c_joint_limit=.5*max(actual_cost)+.5*max(current_reference_cost)`。优先实际
  soft limits，没有则使用硬范围内侧5%安全区；安全区内成本0，穿入边距至硬限位
  线性升至1，越界饱和1。只检查当前消费参考，不扫描整段120帧未来轨迹。
  soft limits 与硬边界在 1e-6 rad 内的浮点构造误差允许数值夹回边界，保留原始
  soft 值与校正标记；超过该容差仍视为无效限位证据。

## 一致性与版本边界

原转换器的插值、差分、FK 所得位置—速度一致性继续检查三个 RMS，默认容差均为
1e-4；不计入组件奖励。超出容差明确为构造/缓存/时间轴错误，转移无效并保留原始
误差，不能让策略用负奖励学习绕过程序问题。

配置集中在两个 stage9_dppo YAML，完整解析值写入 resolved_config；reward 字典及
源码 manifest 都绑定 checkpoint 身份，因此旧 v1 checkpoint 不能静默完整续训为
v2。只读审计器按版本分别核对历史 v1 和当前 v2，不修改旧数据或旧验收结论。

## 2026-09-30 实际验证与交付

本地完整集成230 passed（仅既有rotary autocast弃用warning），覆盖整个dppo、
协调器、条件及源码身份；预热目标初始化补修后，奖励/数据/环境定向82 passed。
GMT 136 passed、1 skipped，跳过项为本地GENMO环境缺IsaacLab的配置构建测试，
随后在服务器真实启动执行中验证。测试临时目录已清理。新审计器读取旧v1本地归档
仍为28 passed/4 not_run（仅缺远端checkpoint），不使用新公式解释旧数据。

服务器1进行了两轮collect，各64条上层转移，未调用Actor或Critic优化。初版GENMO
`c8628a0`的采集/字段校验通过，但复查发现首音乐步没有承接预热上一目标，该轮
只保留作修正前诊断。修正版GENMO `d6423b7`、GMT `4e7927c`的最终轮逐步检查
通过，首音乐步上一目标与SQLite预热末步的实际提交目标逐元素一致，跳变正常计价。

最终轮有64条上层转移、1600音乐控制步、5个episode；完整journal为386次mutation、
140次advance、1900控制/7600物理步（含预热和校准）。每步8项奖励、配对目标、
4物理子步和参考一致性有效，cmd包括首音乐步均有效。跟踪/稳定性/cmd独立重算
误差0，torque最大差5.5511e-17；积分、门控/强度、事件及执行身份检查通过。
零更新log-prob和ratio差均0。真实批次没有执行失败，故障分支由CPU测试覆盖。

| 实际分布 | 最小 | 平均 | 最大 |
| --- | ---: | ---: | ---: |
| activity_gate | .053804 | .953010 | 1.000000 |
| A_actual，rad/s | .061918 | .705889 | 1.876646 |
| A_target，rad/s | .002172 | 1.057168 | 3.064273 |
| intensity | 3.07e-9 | .722238 | .99999999 |
| 每控制步积分奖励 | .027382 | .075080 | .104938 |

这些值证明活动监督有实际区分，不能证明策略改善；本轮没有网络更新，奖励尺度
也与v1不同，不能直接跨版本比较累计分数。原模型、统计量及GMT运行指纹保持不变，
GMT正常退出；结束时8张GPU均0MiB，无本轮计算进程。两轮共享预算累计160生成、
3800控制、15200物理步、0次优化，没有重置首轮消耗。

完整原始去噪链、SQLite、rollout和报告保留在服务器：

```text
/data0/user/liwei/GENMO_outputs/closedloop_stage9/reward_v2_collect_20260930
/data0/user/liwei/GENMO_outputs/closedloop_stage9/reward_v2_collect_20260930_run02
/data0/user/liwei/GENMO_outputs/closedloop_stage9/reward_v2_20260930_budget.json
```

最终轮92文件/383055883字节，SHA清单为artifact_manifest.json。9个JSON/YAML报告
回传本地同名outputs/closedloop_stage9目录且逐SHA一致，83个原始执行文件留服务器。
独立结果为reward_v2_execution_audit.json，回传证明为local_report_verification.json。
这些是用户要求的决策记录与奖励验收证据；初版用于预热边界诊断，不与最终数据混用。

本次单独调用现有审计器的journal/rollout/probability/frozen检查，并从真实字段
独立重算公式。完整audit_closedloop_dppo.py CLI的train＋resume门禁仍要求价值更新、
Actor更新和checkpoint，不能把本次collect-only结果冒充完整训练验收。
