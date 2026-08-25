# BUMI qpos30、FK 接触与足底锁定 v3

本文档说明 `feature/bumi-music-only` 分支从 2026-08-25 起采用的 BUMI 当前训练与部署
契约。它针对旧 93D 表示中“网络同时预测 qpos 决定量和 63 维 link 位置、但最终机器人
只执行 qpos”的冲突，统一规定网络只生成真正决定 qpos28 的 30 个连续量；所有 link、
鞋底与穿透几何必须由同一份 qpos 经权威 BUMI FK 得到。文档同时给出版本边界、可靠接触
标签、foot-slide、root tilt、仅修脚滑的后处理，以及服务器 2 固定 8 卡启动命令。

## 当前表示

契约：`genmo.bumi_motion_features.qpos30.v3`

| 切片 | 维数 | 含义 |
|---|---:|---|
| `[0:2]` | 2 | 当前 heading 坐标系的逐帧 root XY 位移 |
| `[2:3]` | 1 | 相对默认 root height 的逐帧绝对高度 |
| `[3:9]` | 6 | root rotation 的 rot6d |
| `[9:30]` | 21 | MuJoCo 原生关节顺序的关节角 |

rot6d 确定一个单位根旋转，配合 3 维根位置和 21 个关节角可唯一组合 qpos28。模型输出中
不再存在 `body_link_pos_root[63]`。训练、评估和渲染若需要 21 个 feature link，流程固定为：

```text
normalized qpos30 -> physical qpos30 -> qpos28 -> BumiKinematics FK -> link/sole geometry
```

旧 93D checkpoint、旧 stats、旧单输出 ONNX 和旧 TensorRT engine 都不兼容；加载边界会
明确失败，不能按前 30 维截断后冒充新模型。

## 接触标签与脚滑

接触契约：`genmo.bumi_foot_contact.fk_sole_hysteresis.v1`。

- 左右足标签来自 GT qpos、固定 kinematics 和真实鞋底 proxy，不信任无版本的历史零标签。
- GMR 四库的 `gmr_foot_sole_ground_zero_v1` 严格使用世界 `Z=0`。
- 自建历史库的 `legacy_body_origin_min_zero` 使用当前 GT 序列足底高度 2% 低分位估计等效
  地面；混合 batch 会逐样本选择，不把两种地面语义混在一起。
- 进入接触阈值为高度 `0.035 m`、水平速度 `0.15 m/s`；退出阈值放宽到 `0.055 m`、
  `0.25 m/s`，并删除少于 2 帧的脉冲。
- 两维 contact head 用 BCE 监督。foot-slide loss 只在 GT 连续接触且两帧有效时惩罚预测
  FK 鞋底水平速度；门控不使用预测接触，模型不能通过把接触概率降为零逃避脚滑。

部署足底锁定契约：`genmo.bumi_fk_foot_lock_xy.v1`。它根据 contact head 的迟滞状态维护
左右鞋底 FK 锚点，只修改 floating root 世界 XY。root Z、root quaternion 和 21 个关节角
逐值不变。原先用于遮掩躺倒的强制 root 直立/抬升接口已从正式运行时移除。

## Root rotation 与 tilt

当前损失契约：`physical_qpos30_contact_v2`。根姿态由三层监督共同约束：

- normalized rot6d 表示损失 `repr_root_rot=2.0`；
- 完整 SO(3) 测地误差 `root_rot=1.0`；
- 从 ZYX 根旋转显式提取 roll/pitch 环绕角，并叠加过大倾角安全项
  `root_tilt=1.0`。

root tilt 在机器人已倾斜时仍对左乘 yaw 不敏感，不会把舞蹈转向压回固定方向。单位旋转处
的角度公式带可微数值下限，已验证 forward/backward 不产生 NaN。防躺倒属于模型训练
目标，不由后处理改四元数。

主要权重为：

```text
repr_root_pos=1.0  repr_root_rot=2.0  repr_joint=1.0
root_pos=0.2       root_rot=1.0       root_tilt=1.0
joint_dof=0.2      fk_body_pos=1.0
joint_velocity=0.05  joint_acceleration=0.005  joint_jerk=0.001
joint_limit=0.1    contact_bce=1.0   foot_slide=0.05
penetration=0.05   root_height=0.1
```

foot-slide 权重保持温和，并在 5k step 内渐进启用，避免以“所有动作少动”换低脚速；root
rotation、root tilt、接触 BCE 和三个直接表示损失从第一步生效。

## 归一化

stats 契约为 `genmo.bumi_qpos30_stats.v3`，运行环境变量改为
`BUMI_MUSIC_QPOS30_STATS_PATH`。现有五库 93D stats 显示 root XY 位移标准差约为
`0.0055–0.0061 m/帧`；若机械照搬 SMPL 的 `std<1 -> 1`，根运动监督会缩小约 160 倍。
因此仍采用“给 std 设下限”的 main 思想，但 BUMI 专用下限为 `0.01`。stats 文件会绑定
该值、kinematics SHA、表示版本和五库 manifest 指纹。

服务器 2 上重算统计量：

```bash
$GENMO_PYTHON tools/data/bumi/compute_bumi_30d_stats.py \
  --kinematics "$BUMI_KINEMATICS_PATH" \
  --dataset "aistpp_bumi=$AISTPP_BUMI_ROOT" \
  --dataset "aioz_gdance_bumi=$AIOZ_GDANCE_BUMI_ROOT" \
  --dataset "finedance_bumi=$FINEDANCE_BUMI_ROOT" \
  --dataset "compas3d_bumi=$COMPAS3D_BUMI_ROOT" \
  --dataset "mine_bumi=$MINE_BUMI_ROOT" \
  --dataset-joint-limit-tolerance mine_bumi=0.25 \
  --output "$BUMI_MUSIC_QPOS30_STATS_PATH"
```

## 8 卡 350k 完全从零训练

实验入口：`gem_bumi_music_only_5set_manual_q1_v3_qpos30_contact_scratch_350k`。每卡
batch=192、8 卡全局 batch=1536，训练 350k step；网络 qpos30 输入列、30 维输出层、两维
contact head 与 Transformer 主干都从随机初始化开始。配置把 `pretrain_ckpt`、`ckpt_path`、
`resume_mode` 和 `checkpoint_adapter` 全部固定为 null，因此不会加载 main/SMPL、旧 BUMI
模型、optimizer 或 global step。

```bash
cd /home/user/liwei/GENMO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export TORCH_NCCL_BLOCKING_WAIT=1

$GENMO_PYTHON -u scripts/train.py \
  exp=gem_bumi_music_only_5set_manual_q1_v3_qpos30_contact_scratch_350k \
  output_dir="$BUMI_QPOS30_OUTPUT" \
  pl_trainer.devices=8 \
  pl_trainer.strategy=ddp
```

正式启动命令仍应显式追加 `pretrain_ckpt=null model.model_cfg.checkpoint_adapter=null`，作为
配置之外的第二道防护。学习率里程碑为 210k/315k，每 5k step 保存 checkpoint。

原生 qpos30 checkpoint 保存表示版本。之后即使配置仍保留 adapter，加载该 checkpoint 时
也会优先按原生权重完整加载，不会再次误走 SMPL adapter。

## 验收边界

- 测试必须检查 30D round-trip、link 全 FK、GT 接触地面语义、contact/slide、root tilt
  backward、足底锁定只改 XY，以及 ONNX/TensorRT 两个输出的 parity。
- demo 默认启用足底锁定并同时保存 `qpos_raw`、`qpos`、contact logits、活动接触和逐帧
  XY 修正；`--no-foot-lock` 可查看纯模型输出。
- foot-slide 指标按 GT 或 contact head 声明的接触区统计，不能用“脚已经很慢”反推接触后
  再测速度，否则滑动脚会被排除并得到虚假零分。
- 这些仍是运动学验收；GMT 动力学跟踪、扭矩和真实稳定性需要单独仿真验证。
