# BUMI 默认根高与统计量兼容

适用分支：`feature/bumi-text-only`、`feature/bumi-music-only`、
`deploy/bumi-music-only-gmt`。

## 统一值与物理含义

自 2026-09-20 起，三个分支的活动默认根高均为 **0.48120910 m**。
唯一代码定义是 `gem/robots/bumi/kinematics.py` 的
`BUMI_DEFAULT_ROOT_HEIGHT_M`；`BumiKinematics.default_qpos[2]`、qpos30 编解码、
默认世界 anchor、canonical 地面与损失计算共同使用它。

该值来自部署分支 fe934 机器人在零关节角姿态下的 FK 贴地结果：
最低脚底代理点位于 z=0.002 m，可见脚底网格约离地 1 mm。
它表示世界地面 z=0 时根坐标原点的参考高度，不是机器人身高，也不是每帧的强制高度。
四元数 wxyz、21 关节顺序、轴、限位和 FK 几何均沿用原有资产。

待机调用 `make_standing_qpos(joint_positions)`：先设置实际关节姿态，再用 FK 将
最低脚底代理点放到 2 mm。当前 GMT 屈膝待机约为 **0.47546181 m**，继续由实际
策略默认角度计算，不硬设为零位高度。生成动作不逐帧贴地，跳跃、下蹲等保留自身高度。

## 新统计量和旧模型

特征的高度通道仍是 `h = root_z_world - z_ref`，30D → qpos28 仍为固定数学转换。
新统计量升级为 `genmo.bumi_qpos30_stats.v4`，必须包含有限、正数的
`root_height_reference_m`。文本完整序列统计和音乐 qpos30 统计均写入当前基准。
qpos30 表示版本、网络维度和 checkpoint 权重结构不变。

加载旧 `genmo.bumi_qpos30_stats.v3` 时，其高度基准取自经过 SHA 核验的原运动学
资产 `default_qpos[2]`；运行时将它单独保存为 `source_default_qpos[2]`。
随后只对内存中的高度均值做补偿：

```text
mean_new[2] = mean_old[2] + z_ref_old - 0.48120910
std_new     = std_old
```

其余 29 维不变，因此相同世界动作的归一化网络输入及相同网络输出解码后的世界高度
保持一致（允许浮点舍入误差）。新 v4 文件按显式基准换算，当前基准不会重复补偿。
给 v3 混入显式根高会被拒绝，避免旧程序误读新语义；旧代码也会拒绝 v4，应同步升级
相应运行时代码。标准差仍执行既有的最小值裁剪。

原 MJCF/kinematics JSON 中的 0.55 m（或旧资产的 0.65 m）保留为来源 qpos0，
不再决定活动默认高度。原始文件、生产数据、stats、checkpoint、ONNX、engine、
资源清单及 SHA 绑定不改写；直接打开原 MJCF 的 MuJoCo qpos0 仍反映该来源值。
仓库内预览/待机须经 `BumiKinematics` 初始化，不能直接读 JSON qpos0 作为活动站姿。
已有 checkpoint 搭配原 stats 即可通过兼容路径读取，无需仅因基准变更重新训练或导出。

显式传入世界 `anchor_z` 仍表示覆盖当前参考原点；历史脚本若自行硬编码旧 anchor_z，
需按其摆放目的调整。该兼容承诺针对默认世界 anchor，不替调用者改写显式参数。

## 验证入口

`tests/bumi/test_bumi_root_height.py` 覆盖旧基准 0.55/0.65/1.0 的归一化输入、
非 GT 高度预测的世界输出及梯度一致，新统计量不重复补偿、非法契约拒绝、
资产字节保留和零位/屈膝 FK 贴地。分支既有编解码、损失、文本生成或音乐部署测试
同时验证受影响链路。单元测试与 CPU FK 不等同于模型生成质量或实机跟踪验收。
