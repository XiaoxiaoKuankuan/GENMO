# 服务器1 UMR 前70%与自建音乐库训练交付

本文件记录 2026-09-18 实际完成的数据查找、服务器2到服务器1传输、统一质量筛选、
训练格式发布、统计量计算和八卡验证。适用分支为 `feature/bumi-music-only`，
代码验证提交为 `8aaa57e33c87b7f9534d5877b0c4e2ab3b6558b3`。
最终数据已就绪；正式 350000 optimizer steps 训练尚未启动。

## 1. 已发布位置与启动方法

| 用途 | 服务器1绝对路径 |
|---|---|
| BUMI代码工作树 | `/home/user/liwei/GENMO-bumi-music` |
| 训练Python | `/home/user/liwei/GENMO/.venv/bin/python` |
| 正式训练数据 | `/data0/user/liwei/datasets/bumi_music_umr70_mine_pass_v1` |
| 训练集统计量 | 上述数据目录的 `stats/qpos30_train_stats.json` |
| 完整准备报告 | 上述数据目录的 `training_preparation_report.json` |
| 格式发布报告 | 上述数据目录的 `conversion_report.json` |
| 全量质量报告 | `/data0/user/liwei/datasets/bumi_music_umr70_mine_quality_v1/quality_report.jsonl` |
| 质量统计与配置快照 | `/data0/user/liwei/datasets/bumi_music_umr70_mine_quality_v1` |
| 自建原始数据、参考版本和重建结果 | `/data0/user/liwei/datasets/bumi_music_umr70_mine_sources_v1` |

2026-09-18按用户要求将原工作树目录重命名为 `GENMO-bumi-music`，
使用 `git worktree move` 同步更新Git工作树登记。已有远程终端请重新 `cd` 到新路径。

登录服务器1后执行：

```bash
cd /home/user/liwei/GENMO-bumi-music
BUMI_OUTPUT_BASE=/data0/user/liwei/GENMO_outputs/bumi_music_umr70_mine \
  bash scripts/train_bumi_music.sh
```

需要断开SSH后继续运行时，用独立tmux会话启动同一入口：

```bash
tmux new-session -d -s bumi_umr70_350k \
  'cd /home/user/liwei/GENMO-bumi-music && BUMI_OUTPUT_BASE=/data0/user/liwei/GENMO_outputs/bumi_music_umr70_mine bash scripts/train_bumi_music.sh'
tmux attach -t bumi_umr70_350k
```

二者选择一种执行，避免重复启动。训练入口为
[scripts/train_bumi_music.sh](../scripts/train_bumi_music.sh)，实验配置为
[gem_bumi_music_only_umr70_mine_scratch_350k.yaml](../configs/exp/gem_bumi_music_only_umr70_mine_scratch_350k.yaml)。
脚本读取本数据版本的统计量并严格校验来源指纹，自动设置四个数据根目录、8卡及NCCL环境。
上述命令通过已有的 `BUMI_OUTPUT_BASE` 参数指定独立日志父目录，
每次正式运行创建
`/data0/user/liwei/GENMO_outputs/bumi_music_umr70_mine/bumi_umr70_mine_s350k_<时间>_<提交短SHA>`，
启动时打印完整路径。以下用 `<运行目录>` 指代该路径：

| 文件用途 | 实际保存位置 |
|---|---|
| 控制台训练日志 | `<运行目录>/launch.log` |
| TensorBoard事件 | `<运行目录>/version_0/events.out.tfevents.*` |
| 训练checkpoint | `<运行目录>/version_0/checkpoints/` |
| 启动时Hydra配置 | `<运行目录>/hydra/.hydra/` |

独立的新运行从 `version_0` 开始。TensorBoardLogger代码在
[scripts/train.py](../scripts/train.py) 中明确将logger目录设为运行目录下的版本目录，
保存回调随后使用同一个 `cfg.output_dir`。

训练启动后，在服务器1查看最新一次控制台日志：

```bash
BUMI_RUN_DIR=$(ls -dt /data0/user/liwei/GENMO_outputs/bumi_music_umr70_mine/bumi_umr70_mine_s350k_* | head -n 1)
tail -n 100 -f "$BUMI_RUN_DIR/launch.log"
```

在服务器1的另一个终端启动TensorBoard，读取固定的BUMI日志父目录：

```bash
/home/user/liwei/GENMO/.venv/bin/tensorboard \
  --logdir /data0/user/liwei/GENMO_outputs/bumi_music_umr70_mine \
  --host 127.0.0.1 \
  --port 6006
```

在本地电脑终端建立转发，并保持该命令运行：

```bash
ssh -p 50030 -N -L 16006:127.0.0.1:6006 user@112.65.216.193
```

浏览器打开 `http://127.0.0.1:16006`。核验时服务器6006和本地16006均空闲，
服务器TensorBoard可执行文件版本为2.21.0；本次只提供启动命令，未启动训练或新的TensorBoard服务。
日志父目录和事件文件会在用户启动正式训练后生成。

`bash scripts/train_bumi_music.sh --smoke` 只运行两步八卡训练和三来源各一个验证batch，
输出使用 `/tmp/genmo-bumi-eight-gpu.*` 的独立目录并在退出时清理。
本次已执行成功，无需为启动正式训练重新生成数据。

## 2. 来源与70%口径

UMR动作根目录：`/home/user/music7286_umr/out_umr/bumi3`。
使用的明确选择清单为：

```text
/home/user/music7286_umr/out_umr/bumi3/manifests/motion_track_foot_spread/umr_bumi3_motion_track_foot_spread_best70pct.json
```

UMR当前全集6915条：AIOZ-GDANCE 5890、AIST++ 899、FineDance 126。
该清单取其中4840条，实际为69.9928%，总50.762463小时；
它按脚底高度P90-P10波动排序。这里的70%是UMR当前6915条的70%，
也不是在后续质量门禁之后再次抽70%。最初7286条人体源数据与6915条UMR全集口径不同。
本清单没有CoMPAS3D，因此训练不配置这一来源。

服务器2找到的自建原始库为
`/data0/user/liwei/datasets/bumi_music_mine_raw_v1`，
其中 `dance_2_csv` 48组、`dance_3_csv` 52组，共100组CSV/WAV。
原始动作总1.432150小时。已传输到服务器1来源目录的同名子目录。
传输同时包括旧99条参考版本和缺失的AIST音频，共561文件、2820676485字节；
两端逐文件大小与SHA256一致，记录在来源目录的 `transfer_manifest.json`。

自建库在服务器1通过原有CSV构建器重新转换：按具名关节映射到同一机器人，
统一四元数与30Hz时间线，配对音频并生成EDGE35与接触标签。
APT啦啦操因动作约108.03秒、音频约168.02秒不匹配而排除，剩99条、1.401787小时。
没有用静默尾裁把该错配样本纳入训练。

## 3. 筛选结果和数据占比

复用现有自动门禁，UMR先完成qpos来源、关节、坐标、帧率、时间线和机器人资产契约验证，
再通过同一运动学转成筛选器输入。自建99条也接受相同门禁；原质量阈值没有放宽。
只发布PASS；REVIEW和REJECT均不进入训练数据。

| 来源 | UMR全集 | 本次门禁输入 | PASS | REVIEW | REJECT | 门禁剔除率 | 最终动作时长/h | 最终条数占比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AIOZ-GDANCE | 5890 | 4282 | 4161 | 114 | 7 | 2.83% | 44.026704 | 87.3242% |
| AIST++ | 899 | 432 | 391 | 20 | 21 | 9.49% | 1.122454 | 8.2057% |
| FineDance | 126 | 126 | 114 | 7 | 5 | 9.52% | 3.865157 | 2.3924% |
| 自建Mine | — | 99 | 99 | 0 | 0 | 0.00% | 1.401787 | 2.0776% |
| 合计 | 6915 | 4939 | 4765 | 141 | 33 | 3.5230% | 50.416102 | 100% |

从4840条UMR加100条原始自建计算：4940条 → 排除1条音画错配 →
4939条统一质量筛选 → 再排除174条 → 最终4765条，整体保留96.4575%。
质量门禁自身保留96.4770%。UMR部分最终4666条，占UMR全集67.4765%。
最终共有5444939帧，30Hz，动作时长按样本累加；多人同曲样本不在这里去重。

| 来源 | train | val | test | train自然条数占比 | 实际期望采样占比 |
|---|---:|---:|---:|---:|---:|
| AIOZ-GDANCE | 3585 | 283 | 293 | 86.4064% | 59.0674% |
| AIST++ | 365 | 12 | 14 | 8.7973% | 29.0155% |
| FineDance | 100 | 1 | 13 | 2.4102% | 6.7358% |
| 自建Mine | 99 | 0 | 0 | 2.3861% | 5.1813% |
| 合计 | 4149 | 296 | 320 | 100% | 100% |

沿用原始公开库split。此次FineDance有1条有效val，训练监控使用val，13条test保留用于独立评估。
自建99条沿用原来全部train的设置，没有独立自建验证/测试集。

采样沿用旧350k配置的相对权重：AIOZ=0.57、AIST=0.28、FineDance=0.065、Mine=0.05；
删去不存在的CoMPAS后，按总和0.965归一化得到上表比例。
真实八卡采样器日志已确认这些概率。
实现为“来源 → 按时长加权的音乐组 → 同组动作变体 → 均匀时间窗口”，
避免多人舞者的重复音乐条数直接控制音乐组概率。
比例为长期期望值，并不要求每个batch刚好满足。

## 4. 大小与训练格式

下表区分原始机器人动作、自建原始媒体和含音频/特征的训练版本，不能直接用字节比计算筛选率。
GB使用十进制，GiB使用1024进制。

| 对象 | 字节 | GB | 口径 |
|---|---:|---:|---|
| 本次4840条UMR输入 | 592513047 | 0.592513 | 只计重定向NPZ，不含引用音乐 |
| 自建原始100组 | 1124461817 | 1.124462 | 100个CSV及100个WAV |
| 最终4765条训练数据载荷 | 13700742459 | 13.700742 | motions、audio、musicfeat_v2，约12.7598GiB |

最后一行不含少量manifest、meta、统计量和报告。
公共音频与特征尽量硬链接复用，因此逻辑载荷大小不代表本次新增物理占盘。

每个来源保留 `motions/`、`audio/`、`musicfeat_v2/`、`manifests/` 和 `meta/`。
存储运动为 `qpos[T,28]`、接触为 `[T,2]`，音乐特征为EDGE35 `[T,35]`；
训练reader输出30维qpos特征，窗口120帧（4秒）。
UMR保留世界地面0和原始root高度，地面语义为 `umr_foot_sole_ground_zero_v1`；
自建保持其既有body-origin地面语义，接触通过同一资产重算。

## 5. 与350k配置的对应关系

对照服务器2实际保存的旧实验配置：

```text
/data0/user/liwei/experiments/genmo/gem_bumi_music_only_5set_robot_retargeter_pass_v2_qpos30_contact_v5_scratch_b256_s350k_20260909/hydra/.hydra/config.yaml
```

新配置的optimizer、scheduler、network、model、pipeline、endecoder与旧配置对齐，
保持qpos30/contact v5和原物理损失权重。主要设置：

- 350000个optimizer steps，8卡DDP，每卡batch 256，全局batch 2048，梯度累积1。
- 从零训练，`pretrain_ckpt`、`ckpt_path`、`resume_mode`、checkpoint adapter均为空。
- FP16混合精度，AdamW学习率0.0001，在210000/315000步乘0.5，梯度裁剪0.5。
- 每5000步验证和保存；采样器每epoch抽53248个窗口；seed=42。
- 固定 `NCCL_CUMEM_HOST_ENABLE=0`、`NCCL_IB_DISABLE=1`、`NCCL_SOCKET_IFNAME=lo`、`TORCH_NCCL_BLOCKING_WAIT=1`。

因此“35万”指优化器更新步数，不是遍历全部数据35万轮。
统计量只用4149条train计算，本次41313个统计窗口、4957282个特征帧；
窗口包含最后一个合法窗口，统计帧数不是去重后的动作总帧数。
统计量不是placeholder，绑定每库train manifest、dataset_info与kinematics指纹。

| 资产 | SHA256 |
|---|---|
| BUMI3 MJCF | `fe93472dd764704fe8389b0f82052ae84ed8bc90f6d71b1467872f86e08a9ad3` |
| kinematics JSON | `c08731704dccece11351b6fa877e30bac5ca2a8d363de30af7ca2ea1398f4029` |
| 质量配置 | `cda54be059e79ac1cf5c15e8a21003148baa84a94781063c2130dcf98eef72bd` |
| 全量质量JSONL | `324750f42e1c7d5df162ad19a2cabaf85cf8eab8fc9de22d12299060c6e0aa86` |
| 新训练集stats | `a695a1bb09bb9a936869aefb87672e4895bc4f35ab23efb40047bf335fa4d8fc` |

## 6. 实际验证与质量边界

- 4765条发布数据均通过正式reader逐条验收，来源文件与音乐配对SHA已检查；通过后原子发布。
- 本地定向回归30通过、1条件跳过；Ruff、Bash语法和Git差异检查通过。
- 服务器1八张RTX6000D矩阵运算及32/256MiB的all_reduce、broadcast均通过。
- 真实8卡、每卡256的两步训练通过，stdout四舍五入loss分别4.73/4.68，epoch loss为4.7，
  三个公开来源各完成1个验证batch，达到max_steps=2，退出码0。日志报告最大GPU占用32.8GB。
- 临时目录 `/tmp/genmo-bumi-eight-gpu.lsLaJE` 已删除，检查无本次残留训练进程或CUDA计算进程。
  两步验证未触发5000步checkpoint saver，也不能证明长期稳定性或模型收敛质量。

当前质量规则检查运动学和数值代理指标，**没有把悬空比例设为硬门禁**。
已知 `aioz_gdance/EjV9ZVSpOds_06_0_1020_dancer_00` 仍为PASS，
其 `both_off_5cm=0.9549019607843138`。
因此本版本不能称为“已排除全部悬空动作”，脚底波动前70%也不等于绝对贴地。
本任务按既有筛选器阈值完成构建，未把离线PASS或训练smoke当作动力学、部署或实机验收。

本次修改复用了原筛选入口和构建入口，公共UMR适配位于
[umr_qpos_adapter.py](../tools/data/bumi/umr_qpos_adapter.py)；
reader、endecoder和loss同步识别新增地面语义，旧robot-retargeter默认入口保持原契约。
实现和验证记录按时间追加在根目录 [记录文本.md](../记录文本.md)。
