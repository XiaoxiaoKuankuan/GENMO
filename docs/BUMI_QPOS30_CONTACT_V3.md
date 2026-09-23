# BUMI qpos30、FK 接触与足底锁定 v3

本文档说明仓库当前采用的 BUMI qpos30/v5 训练与部署契约。网络只生成真正决定 qpos28
的 30 个连续量；所有 link、
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

维度、表示版本、输出头或统计契约不匹配的 artifact 会在加载边界明确失败，不能截断后
冒充当前 qpos30 模型。

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

## Root rotation、tilt 与 v5 动态长尾

当前正式损失契约：`physical_qpos30_contact_v5`。根姿态仍由三层监督共同约束：

- normalized rot6d 表示损失 `repr_root_rot=2.0`；
- 完整 SO(3) 测地误差 `root_rot=1.0`；
- 从 ZYX 根旋转显式提取 roll/pitch 环绕角，并叠加过大倾角安全项
  `root_tilt=1.0`。

root tilt 在机器人已倾斜时仍对左乘 yaw 不敏感，不会把舞蹈转向压回固定方向。单位旋转处
的角度公式带可微数值下限，已验证 forward/backward 不产生 NaN。防躺倒属于模型训练
目标，不由后处理改四元数。

v5 在 qpos30、contact head 和 FK 语义不变的前提下，继续约束关节速度、加速度、jerk、
限位、脚滑、穿地和根高，并新增根平移/旋转动态、root-frame FK 动态和长尾 top-k/max 项。
正式 s350000 scratch 配置在 v5 基础上冻结的主要动态与安全权重为：

```text
joint_velocity=0.15  joint_acceleration=0.04  joint_jerk=0.006
joint_acceleration_excess=0.20  joint_jerk_excess=0.006
joint_limit=0.40  joint_limit_margin=0.80
joint_limit_topk=2.00  joint_limit_max=0.20
foot_slide=0.10  penetration=0.10  root_height=0.15
root_velocity=0.05  root_angular_acceleration=0.02  fk_acceleration=0.02
```

超额项只惩罚预测导数超过同帧 GT 幅值的部分；top-k/max 项避免短时尖峰被全序列均值
稀释。v5 的 robust joint-limit 和 advanced-physics 项分别在 10000 step 内 warmup；直接
表示、root rotation/root tilt 与 contact BCE 从第一步生效。它们仍是运动学与有限差分
代理，不等价于控制器闭环、接触力、扭矩或实机安全验证。

## 归一化

新 stats 契约为 `genmo.bumi_qpos30_stats.v4`，写入 `root_height_reference_m`；
默认根高为 **0.48120910 m**，兼容加载旧 v3 stats 并补偿高度均值。
详见 [根高与旧模型兼容](BUMI_ROOT_HEIGHT.md)。运行环境变量为
`BUMI_MUSIC_QPOS30_STATS_PATH`。构建五库 qpos30 stats 时，root XY 位移标准差约为
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
  --output "$BUMI_MUSIC_QPOS30_STATS_PATH"
```

## 8 卡 350k 完全从零训练

冻结入口：
`gem_bumi_music_only_5set_robot_retargeter_pass_v2_qpos30_contact_v5_scratch_s350000`。
每卡 batch=256、8 卡全局 batch=2048，五库共 2578 条 train 序列，训练 350k step；网络
qpos30 输入列、30 维输出层、两维 contact head 与 Transformer 主干都从随机初始化开始。
配置把 `pretrain_ckpt`、`ckpt_path` 和 `resume_mode` 全部固定为 null，
因此不会加载 SMPL、旧 BUMI 模型、optimizer 或 global step。

```bash
cd /home/user/liwei/GENMO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export TORCH_NCCL_BLOCKING_WAIT=1

$GENMO_PYTHON -u scripts/train.py \
  exp=gem_bumi_music_only_5set_robot_retargeter_pass_v2_qpos30_contact_v5_scratch_s350000 \
  output_dir="$BUMI_QPOS30_OUTPUT" \
  pl_trainer.devices=8 \
  pl_trainer.strategy=ddp
```

正式启动命令仍可显式追加 `pretrain_ckpt=null`，作为
配置之外的第二道防护。学习率为 `1e-4`，里程碑为 210k/315k，每 5k step 保存 checkpoint。
该版本化入口来自正式运行目录的 Hydra config/overrides；除将服务器绝对 `output_dir`
规范化为仓库相对路径外，模型、数据、损失、优化器、scheduler 和 trainer 字段逐项一致。

## 验收边界

- 测试必须检查 30D round-trip、link 全 FK、GT 接触地面语义、contact/slide、root tilt
  backward、足底锁定只改 XY，以及 ONNX/TensorRT 两个输出的 parity。
- demo 默认启用足底锁定并同时保存 `qpos_raw`、`qpos`、contact logits、活动接触和逐帧
  XY 修正；`--no-foot-lock` 可查看纯模型输出。
- foot-slide 指标按 GT 或 contact head 声明的接触区统计，不能用“脚已经很慢”反推接触后
  再测速度，否则滑动脚会被排除并得到虚假零分。
- 这些仍是运动学验收；GMT 动力学跟踪、扭矩和真实稳定性需要单独仿真验证。
