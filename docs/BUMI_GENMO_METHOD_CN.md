# BUMI-GENMO：面向 BUMI 机器人的音乐条件动作扩散方法

## 文档定位

本文说明 BUMI-GENMO 的**方法、表示、训练目标和整体框架**。它回答的是“模型在学什么、为什么这样设计、输入怎样变成机器人动作”，而不是逐个解释源码文件、类或命令。

本文按当前 `genmo.bumi_motion_features.v2` 表示说明。旧 `s430000` 使用绝对
`root_pos_local` v1，虽然同为 93D，也不能与本文的数据语义混用。

工程实现、数据契约、部署命令和当前训练状态另见 [BUMI-native Music-only GENMO 工程说明](bumi_native_music_genmo.md)。

## 1. 方法目标

BUMI-GENMO 要学习的条件分布是：

```text
p(BUMI 动作 | 音乐)
```

给定一段 30 Hz 的音乐特征序列

```text
M = [m1, m2, ..., mT],  mt ∈ R^35
```

模型生成同样长度的 BUMI 机器人状态序列

```text
Q = [q1, q2, ..., qT],  qt ∈ R^28
```

每帧 `qpos28` 由以下三部分组成：

```text
root XYZ         3 维，米，Z-up
root quaternion  4 维，wxyz
21 个关节角     21 维，弧度
```

它不是“先生成一个人的舞蹈，再实时调用 GMR 把人变成机器人”。人体到 BUMI 的重定向只发生在**训练数据准备阶段**。完成训练后，运行模型所需的主路径是：

```text
音乐 → BUMI-GENMO → BUMI qpos28
```

因此它是 BUMI-native 生成模型；但它学习的动作分布仍继承自离线 GMR 产生的训练目标。

## 2. 一张图理解整体框架

### 训练阶段

```text
四个人体音乐舞蹈数据集
          │
          ├── 音频 ───────────────→ 30 Hz EDGE35 ──────────────┐
          │                                                     │
          └── 人体动作 → 离线 GMR → BUMI qpos28 → 自动质量筛选 │
                                                 │              │
                                                 ▼              ▼
                                   first-frame canonicalization
                                                 │
                                      qpos + Torch FK
                                                 │
                                      physical 93D → 标准化 x0
                                                 │
                               随机扩散时刻 t，加噪得到 xt
                                                 │
                         xt + t + EDGE35 → RoPE Transformer
                                                 │
                                      预测干净动作 x0_hat
                                                 │
                       表示损失 + qpos/FK/时序/限位辅助损失
```

### 推理阶段

```text
WAV
 │
 ▼
30 Hz EDGE35
 │
 ▼
随机噪声 xN ──DDIM 多步去噪 + music CFG──→ normalized 93D
                                                │
                                                ▼
                                           physical 93D
                                                │
                              取 root/rotation/joint 前 30D
                                                │
                                                ▼
                                  canonical BUMI qpos28
                                                │
                           ┌────────────────────┴───────────────┐
                           ▼                                    ▼
                  可选放回世界坐标                       Torch FK 重算
                           │                                    │
                           └────────→ qpos28 + link positions ←─┘
```

## 3. 如何理解五段式链路

```text
BumiMusicDanceDataset
  → BumiEndecoder (qpos/FK → normalized 93D)
  → diffusion_lg_bumi93 (RoPE Transformer + music CFG)
  → BumiMusicPipeline
  → canonical/world qpos28 + Torch FK
```

这五段并不是五个串联的神经网络。

| 环节 | 方法含义 | 输入 | 输出 | 是否学习参数 |
|---|---|---|---|---|
| Dataset | 提供时间严格对齐的训练对 | BUMI 动作、EDGE35 | `qpos[120,28]`、音乐、有效帧掩码 | 否 |
| Endecoder 编码侧 | 建立统一坐标系，执行 FK，构造并标准化 93D 学习目标 | qpos28 | normalized 93D | 否，属于确定性几何变换 |
| diffusion 模型 | 学习在音乐条件下把带噪 93D 恢复为干净 93D | `xt`、扩散时刻、音乐 | `x0_hat[120,93]` | 是，核心生成网络 |
| Pipeline | 组织训练/推理，解码预测并计算各项损失 | 模型预测和 GT | 总损失或生成结果 | 无额外学习模块；它容纳并调用扩散子模型 |
| Endecoder 解码侧 + FK | 把 93D 变回机器人状态，并重新计算可信几何 | predicted 93D | qpos28、link positions | 否，属于确定性机器人运动学 |

最容易产生的误解是把 `BumiEndecoder` 当成可学习的 autoencoder。当前方法中它不是 VAE，也没有学习一个潜空间；它只是统一承担 canonicalization、93D 编解码、统计量归一化和 Torch FK。

另一个关键点是：93D 的最后 63 维不会直接成为机器人控制量。最终 qpos 只由 root 位置、root 旋转和 21 个关节角确定，身体位置会通过 FK 再算一遍。

## 4. 音乐条件表示

音乐使用与原 GENMO music-only 路径一致的 EDGE baseline35，每帧 35 维、30 Hz：

| 通道 | 内容 |
|---|---|
| 0 | onset strength |
| 1–20 | 20 维 MFCC |
| 21–32 | 12 维 chroma CENS |
| 33 | onset peak |
| 34 | beat peak |

这些特征同时提供音色、频谱包络、和声以及节拍脉冲。动作与音乐逐帧对齐，因此第 `t` 帧动作始终对应第 `t` 帧音乐特征。

模型没有接收文本、相机、图像或人体姿态条件；当前正式实验是严格的 Music-only 条件生成。

## 5. 为什么不直接扩散 qpos28

直接学习 qpos28 有三个问题：

1. 四元数有单位范数约束，并且 `q` 与 `-q` 表示同一个旋转；普通欧氏回归不友好。
2. 单独监督关节角并不能直接告诉网络手、脚和躯干在空间中的几何结果。
3. 数据中的世界平移和绝对朝向与舞蹈内容无关，会浪费模型容量。

因此采用 v2 93D 冗余表示：

```text
x_t = [Δp_t,xy^heading, z_t-z_default, r_t^6D, θ_t,
       b_t,1^root, ..., b_t,21^root]
```

| 分量 | 维度 | 作用 |
|---|---:|---|
| Root heading-local 水平增量 | 2 | 表示逐帧水平运动，整段只积分一次 |
| Root 高度偏移 | 1 | 每帧直接表示 `z-z_default`，不积分 Z |
| Root 局部旋转 rot6d | 6 | 连续地表示完整 root 姿态 |
| 21 个关节角 | 21 | 决定机器人的可执行构型 |
| 21 个 root-relative link 位置 | 63 | 直接提供姿态几何监督，不重复携带轨迹 |
| 合计 | 93 | 扩散模型的输入和输出维度 |

rot6d 解码为旋转矩阵后再转为单位四元数，所以最终 qpos 仍是 28 维：

```text
3 + 6 + 21 = 30D 学习分量
        ↓ rot6d → quaternion
3 + 4 + 21 = 28D qpos
```

后 63D 是“冗余但有用”的辅助表示。它让网络不仅拟合角度，还显式理解角度造成的身体
几何；逐帧 root-relative 坐标去除了整段平移和朝向，避免辅助通道与水平积分轨迹重复。
FK 一致性损失约束它不能与前 30D 描述出两副互相矛盾的身体。

## 6. First-frame canonicalization

### 6.1 要消除什么

同一个舞蹈放在场地左边或右边、朝东或朝西，本质内容没有变化。若直接学习世界坐标，模型还要记忆这些无关自由度。

每个 120 帧训练窗口因此都以自己的第一帧建立局部坐标系。

设：

- `p_t` 为世界 root 位置；
- `R_t` 为世界 root 旋转；
- `H_0` 为第一帧 root 旋转中只保留 yaw 的 heading；
- `z_default` 为 BUMI 默认 root 高度；
- `a = [p_0.x, p_0.y, z_default]` 为 anchor。

局部旋转、水平增量、高度和 link 位置定义为：

```text
R_t^local   = H_0^-1 R_t
Δp_t,xy^heading = Heading(R_t^local)^-1 (p_t+1.xy - p_t.xy)
h_t         = p_t.z - z_default
b_t,j^root  = R_t^-1 (b_t,j - p_t)
```

这里的 `b_t,j` 是从 qpos 经过 BUMI FK 得到的第 `j` 个驱动链节位置。

### 6.2 第一帧究竟被归零了什么

- 第一帧 root 的水平 `X/Y` 被归零；
- 第一帧 yaw 被归零；
- root 高度相对于默认站姿高度保留；
- root roll、pitch 保留；
- 后续水平运动以当前 heading 下的米/帧增量保留；
- link 辅助几何相对于每帧 root 表示，不携带世界轨迹。

所以 canonicalization 不是把机器人强行摆成标准站姿，而是删除与舞蹈内容无关的“场地位置和初始朝向”。

### 6.3 canonical 与 world qpos

模型先生成 canonical motion。解码令 `p_0.xy=[0,0]`，只对水平增量积分，Z 每帧直接取
`h_t`。若指定世界 anchor `(x, y, yaw)`，则可确定性地把局部轨迹旋转和平移回世界坐标：

```text
p_t+1^local.xy = p_t^local.xy
                 + Heading(R_t^local) Δp_t,xy^heading
p_t^local.z = h_t
p_t^world = H_world p_t^local + [x, y, z_default]
R_t^world = H_world R_t^local
```

这一步只改变整段舞蹈放在哪里、朝向哪里，不改变关节动作本身。GMT 不需要提供地图全局
XY；默认局部原点就是 `[0,0]`。只积分水平运动避免分块位置重置，而保留绝对高度避免完整
XYZ 速度积分的长期上浮/下沉。

## 7. 归一化与统计量

93 个通道的量纲差异很大：位置用米、关节用弧度、rot6d 为无量纲。扩散前逐通道标准化：

```text
x0 = (x_physical - μ_train) / σ_train
```

`μ_train` 和 `σ_train` 只由 train split 计算，并与数据版本、关节顺序和运动学资产绑定。这样可避免 validation/test 信息泄漏，也防止换了数据或机器人资产却继续使用旧统计量。

扩散模型实际生成的是 normalized 93D；推理后必须先反归一化，才能得到具有米和弧度物理意义的表示。

## 8. 音乐条件扩散模型

### 8.1 训练任务

对于干净动作 `x0`，在 1000 个训练扩散时刻中均匀抽取 `t`，使用 cosine noise schedule 构造：

```text
xt = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) ε,
ε ~ N(0, I)
```

网络不是预测噪声 `ε`，而是直接预测干净动作：

```text
x0_hat = fθ(xt, t, M)
```

这种 `x0` prediction 与 93D 的分组重建和物理辅助损失可以直接连接。

### 8.2 Transformer 如何融合信息

每个时间位置包含三类信息：

```text
音乐帧 M_t → music embedding
扩散时刻 t → timestep embedding
带噪动作 xt → motion projection
```

音乐 embedding 与 timestep embedding 融合后，再与带噪 93D 拼接并投影到 1024 维 latent。随后经过 16 层、8 头的 RoPE Transformer。RoPE 给自注意力提供相对时间位置信息，使模型可以学习：

- 音乐起拍与动作启动的关系；
- 前后动作之间的时序依赖；
- 乐句内动作的准备、高潮和收束；
- 跨多个关节的全身协调。

训练窗口为 120 帧，即 4 秒。网络在一个窗口内使用双向时序上下文，不是逐帧自回归控制器。

### 8.3 Music classifier-free guidance

训练时以 10% 概率把整条样本的音乐条件替换为空条件，使同一个网络同时学会：

```text
x0_cond   = fθ(xt, t, M)
x0_uncond = fθ(xt, t, ∅)
```

推理时用 CFG 调整音乐约束强度：

```text
x0_cfg = x0_uncond + s (x0_cond - x0_uncond)
```

当前默认 `s=2.5`。较大的 `s` 通常会让动作更追随音乐，但过大也可能减少多样性并放大不自然姿态。CFG 改变的是生成偏好，不是物理安全保证。

### 8.4 DDIM 生成

推理从高斯噪声开始，通过 DDIM 逐步得到 93D 动作。当前离线质量基线默认 50 步、`eta=0`：

- 固定音乐、seed、长度和参数时结果可复现；
- 改变 seed 可得到同一音乐下的不同舞蹈；
- DDIM 步数减少会更快，但可能降低动作质量；
- `eta=0` 表示采样过程本身使用确定性 DDIM 路径。

## 9. 训练损失在约束什么

### 9.1 主去噪目标

93D 被分成四组，每组都在 normalized 空间使用 MSE，权重均为 1：

```text
L_repr = L_root-motion + L_root-rot6d + L_joint + L_body-position
```

分组而不是整段 93D 一次平均，是为了避免 63 维 link position 仅凭维度数量压过 root 和关节分量。

### 9.2 物理与运动学辅助目标

| 目标 | 含义 | 归一化尺度 | 权重 |
|---|---|---:|---:|
| Root 位置 | 对水平增量积分后匹配 GT 整体轨迹，并直接匹配高度 | 1 m | 0.1 |
| Root SO(3) | 匹配完整 root 姿态的测地角 | π | 0.1 |
| 关节角 | 匹配 21 个机器人关节 | 1 rad | 0.1 |
| FK body position | 由预测 qpos 做 FK 后匹配 GT 几何 | 1 m | 0.5 |
| 93D/FK 一致性 | 预测的 63D 与预测 qpos 的 FK 结果一致 | 1 m | 0.1 |
| 关节速度 | 匹配 GT，而不是压向静止 | 6 rad/s | 0.01 |
| 关节加速度 | 匹配 GT，而不是一味平滑 | 180 rad/s² | 0.002 |
| 关节 jerk | v1 暂不启用 | 600 rad/s³ | 0 |
| 软关节限位 | 惩罚超出 MJCF range 的部分 | 0.1 rad | 0.01 |
| Root 高度 | 匹配 GT 高度变化 | 1 m | 0.05 |

表示损失从 step 0 就启用；其余辅助项在前 10,000 step 从 0 线性升到目标权重，避免随机初始化初期由几何项主导优化。

速度和加速度按 30 FPS 换算为物理单位。只有参与差分的所有帧都是真实有效帧时才监督，因此短序列末尾的补帧不会制造虚假的零速度、零加速度。

### 9.3 为什么 v1 不训练接触、滑步和穿地

当前离线重定向数据沿用 `legacy_body_origin_min_zero` 地面语义：对齐的是历史 body origin，而不是真实鞋底接触平面。若据此伪造脚接触或把地面强设为零，会把错误标签写进模型。

所以正式 v1 明确关闭：

- contact BCE；
- foot sliding；
- sole penetration。

这不是认为这些问题不重要，而是遵守“错误物理标签不如不监督”的原则。未来只有在动作重新贴地、足底代理和接触标签完成审计后，才应启用这些损失。

## 10. 数据如何形成训练分布

### 10.1 从人体数据到机器人教师数据

四个音乐舞蹈数据集首先提供人体动作和音乐。人体动作离线经过固定版本 GMR 转成 BUMI qpos；随后通过速度、加速度、jerk、root 运动和异常贴地风格等规则自动筛选。

最终正式语料为：

| 数据集 | 接受动作数 | train / val / test |
|---|---:|---:|
| AIST++ | 824 | 790 / 15 / 19 |
| AIOZ-GDANCE | 5,608 | 4,614 / 513 / 481 |
| FineDance | 111 | 99 / 1 / 11 |
| CoMPAS3D | 67 | 34 / 18 / 15 |
| 合计 | 6,610 | 5,537 / 547 / 526 |

因此训练可理解为一种**离线重定向教师数据蒸馏**：GMR 不在神经网络内部，也不参与反向传播，但它产生的 BUMI 轨迹构成模型监督信号。

### 10.2 为什么不能直接按文件数采样

AIOZ 中同一音乐可能对应多位舞者或多个变体。如果把每个动作文件都当成独立音乐时长，大数据集和多舞者音乐会被重复放大，模型容易偏向少数数据来源。

当前采用四层采样：

1. 按各数据集的**去重音乐时长开方**选择数据集，并把概率限制在 5%–50%；
2. 数据集内部按每个音乐组的去重时长选择音乐；
3. 在同一音乐组内均匀选择舞者、编舞、take 或角色；
4. 在选中的动作内均匀选择 120 帧窗口。

目标数据集概率约为：

```text
AIST++       11.3993%
AIOZ         50.0000%
FineDance    33.6007%
CoMPAS3D      5.0000%
```

这一策略不是让四个数据集各占 25%，而是在“真实去重时长”和“防止单集垄断”之间折中。

## 11. 预测结果为什么要再次做 FK

模型会同时预测：

```text
A. root + rot6d + joint DOF
B. 21 个 link position
```

由于扩散输出存在误差，A 和 B 不可能永远完全一致。机器人只有一套真实状态，因此必须定义权威路径：

```text
预测前 30D
  → rot6d 转合法 root quaternion
  → 组合 qpos28
  → 绑定版本的 BUMI Torch FK
  → 权威 link positions
```

预测的原始 63D 只用于训练监督、诊断和一致性比较，不用于 IK，也不覆盖 FK 结果。这样可以保证最终几何与最终关节角属于同一个机器人构型。

这也解释了链路末尾的：

```text
canonical/world qpos28 + Torch FK
```

其中 qpos28 是最终机器人运动状态，Torch FK 是它的确定性几何解释，而不是又生成一次动作。

## 12. 与原 SMPL→GMR 链路的区别

| 方面 | 原 SMPL GENMO | BUMI-GENMO |
|---|---|---|
| 模型输出 | 人体 SMPL 151D | BUMI 93D |
| 最终状态 | SMPL 参数 | BUMI qpos28 |
| 在线是否需要 GMR | 需要 | 不需要 |
| 机器人关节限位 | 生成后由重定向间接处理 | 可直接写入训练损失 |
| 机器人 FK | 在线重定向后得到 | 训练和推理都显式使用 |
| 人体动作细节 | 保留更多人体自由度 | 被压缩到 BUMI 的 21 个关节 |
| 误差来源 | 人体生成误差 + 在线 GMR 误差 | 离线 GMR 教师偏差 + 机器人生成误差 |

BUMI-native 的主要意义不是“模型更大”，而是把生成目标从人体空间换到了机器人本体空间，使训练目标、推理输出和实际机器人关节定义一致。

## 13. 如何评价模型是否学好

不能只看训练总 loss。至少要分四层评价：

### 13.1 去噪和拟合

- validation representation MSE；
- joint angle error；
- root trajectory error；
- FK body position error。

这些指标说明模型是否恢复训练分布，但不直接代表舞蹈观感。

### 13.2 时序质量

- joint velocity、acceleration、jerk 的分布和 P95；
- 相邻帧 root linear/angular velocity；
- 视频中是否存在高频抖动和瞬时姿态跳变。

### 13.3 音乐一致性与多样性

- motion beat 与 music beat 的时间距离；
- 相同音乐不同 seed 是否产生不同但合理的舞蹈；
- 固定 seed 的回归样例是否稳定；
- 动作是否只在拍点抖动，而缺少乐句级结构。

### 13.4 机器人可执行性

- 关节是否越限；
- root 高度和倾斜是否异常；
- MuJoCo 运动学渲染是否合理；
- GMT/Gazebo 中能否跟踪；
- 实物上的平衡、扭矩、足地接触和安全性。

最后一层不能由 GENMO 的运动学 loss 替代，必须通过控制器和动力学验证。

## 14. 当前方法明确没有解决的问题

当前模型已经学习音乐到 BUMI 运动学轨迹，但尚不能据此宣称解决了：

- 动力学可行性、质心稳定、ZMP 或捕获点稳定；
- 电机速度、扭矩、功率和热限制；
- 精确脚接触、零滑步和鞋底不穿地；
- 自碰撞和环境碰撞；
- GMT 对所有生成动作的稳定跟踪；
- 任意长度运行时已经从结构上消除水平根位置分块重置，但 v2 重训模型的窗口边界速度、
  根旋转和关节质量仍需用新 checkpoint 实测；
- 对未见音乐风格的强泛化。

此外，模型会学习离线 GMR 中系统性的优点和缺陷。若教师轨迹包含扭曲姿态、异常 root 或不自然动作，数据筛选只能减少问题，不能从理论上完全消除它们。

## 15. 最简心智模型

可以把整个方法记成四句话：

1. **数据层**：先把人体舞蹈离线变成经过筛选的 BUMI 教师轨迹，并与音乐逐帧配对。
2. **表示层**：把 qpos 变成 heading-local 水平增量、绝对高度偏移、局部旋转、关节和
   root-relative FK 几何组成的 normalized 93D。
3. **生成层**：用音乐条件 RoPE 扩散 Transformer 从噪声恢复完整 93D 动作窗口。
4. **机器人层**：只用预测 root 和关节组成 qpos28，再通过绑定 BUMI 资产的 FK 得到唯一可信的机器人几何。

因此，这条链路的核心不是“93D 最后直接控制 93 个机器人量”，而是：

```text
93D 是更适合生成学习的冗余动作语言；
qpos28 才是最终机器人状态；
Torch FK 保证最终状态与机器人几何一致。
```
