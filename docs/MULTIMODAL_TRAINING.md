# BUMI 音乐／英文文本两阶段训练与评价

本说明对应 `feature/bumi-music-text-adapter`，以 `33d02cb` 为基线。一个 Stage1 从头混合训练两种条件，每次推理只选择一种；同一个 Stage2 Adapter 接收两类生成参考。所有命令从本分支仓库根目录执行。输出目录必须独立，准备工具拒绝覆盖已有目录。以下正式训练命令供用户按需执行，本次代码交付没有自动运行它们。

## 1. 资产与数据合同

Stage1 使用原始 30 Hz 发布：音乐四库的动作／EDGE35 配对，以及文本四库的机器人动作／英文 caption／标注区间。当前 SuperMimic 的 50 Hz 控制缓存不能反推为这套数据。Server-2 已现场确认以下原始发布与本地 T5 资产存在；完整全库审计及新统计量需要按下面命令单独生成。

```bash
cd /home/weili/bumi-closedloop-worktrees/multimodal/GENMO
# 服务器上先 cd 到同步后的本分支仓库；PY 指向已有 GENMO Python 环境。
export PY=/home/weili/GENMO/.venv/bin/python
# Server-2 可使用：export PY=/data0/user/liwei/envs/GENMO-cu128/bin/python
export PYTHONDONTWRITEBYTECODE=1
export BUMI_MUSIC_ROOT=/data0/user/liwei/datasets/bumi_music_umr70_mine_pass_90505_v1
export BUMI_TEXT_ROOT=/data0/user/liwei/datasets/bumi_text_crop120_v2
export BUMI_T5_MODEL_PATH=/data0/user/liwei/models/t5-3b_bed96aab
export BUMI_MULTIMODAL_ASSETS=/data0/user/liwei/GENMO_assets/bumi_music_text_v1
export BUMI_KINEMATICS_PATH="$PWD/configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json"
```

这些是服务器路径；本机验证须指向已经存在的本机资产，不能假定它们被自动下载。T5 必须包含 `model.safetensors`、`config.json`、`spiece.model` 和本地 tokenizer 配置。内容 SHA 绑定完整训练 checkpoint，路径可迁移，内容改变会拒绝恢复和文本推理。新 checkpoint 不把 T5 权重重复保存进去。

```bash
"$PY" -B tools/data/bumi/prepare_multimodal_stage1.py \
  --data-config configs/multimodal/data.yaml \
  --output-root "$BUMI_MULTIMODAL_ASSETS"
```

工具只读原发布，输出 `split_allowlist.json`、`split_audit.json`、`qpos30_train_stats.json`。也可加 `--audit-only` 只审计，使用另一个新输出目录。训练读取 allowlist，不重写源 manifest。

| 分配层级 | 固定比例 |
|---|---|
| 条件 | music 50%，text 50% |
| 音乐内部 | AIST++ 20%，AIOZ-GDANCE 35%，FineDance 25%，Mine 20% |
| 文本内部 | MotionMillion 5%，HumanML3D 25%，KIT-ML 10%，BONES 60% |

文本先选母动作，再选记录／caption／窗口，避免多 caption 放大母动作权重。caption 只监督对应区间，因果历史来自完整源动作的过去；根 XY 的右侧 halo 可读取区间边界之后一个真实动作帧，但它不变成额外监督帧。动作有效轴独立于音乐有效轴。地面语义通过 FK 适配，不平移原始根高度。

去重使用源动作、镜像、切片／group、内容摘要和已提供 alias，冲突优先保留 test，再 val，再 train。不同库缺少可核对的共同血缘时，`unverified_cross_dataset_lineage` 明确报告未知，不能把这些条目解释为已经证明无重复。需要确认的跨库关系可通过数据配置的 `source_identity_aliases` 补充后，在新目录重新准备。

统计量仅使用 allowlist 中的 train，按照实际模态、来源、母动作、record、caption 和窗口的概率加权。每个可选窗口区间使用确定性代表窗口做数值估计，**没有枚举全部随机 crop**；统计文件记录该估计方法及身份。两种条件共享这一个 mean/std，验证／测试不参与计算。修改数据或采样分布后应生成新的统计资产和独立实验。

## 2. Stage1：从零训练、恢复与验证

默认结构为 1024 维、16 层、8 heads、120 帧、30 Hz、H50。新主干及所有可训练条件模块从零初始化；冻结预训练 T5 是外部条件编码器。联合配置拒绝旧音乐／文本 checkpoint 的 warm start。语义 dropout 仅 Actor 入口一次，概率 0.1；缺失文本 sentinel 加整个残差门控，保证纯音乐、纯文本和全语义 dropout 均保留 DDP 固定计算图。CFG 两分支共用历史和逐坐标前缀。

显式有界验收（默认 8 步，仍是完整网络，显存需求未在空闲 GPU 上实测）：

```bash
export STAGE1_RUN=/data0/user/liwei/GENMO_outputs/multimodal/stage1_run01
CUDA_VISIBLE_DEVICES=0 "$PY" -B tools/train_closedloop_stage1.py \
  --config configs/multimodal/stage1.yaml --output-dir "$STAGE1_RUN"
```

正式预算需要明确覆盖；下面的 batch／累积值是启动示例，按已验证显存预算设置，并在恢复时保持一致：

```bash
# 复用本仓库 Server-2 单机八卡脚本的 NCCL 设置；本次仅验证 CPU/Gloo。
export NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PY" -B -m torch.distributed.run \
  --standalone --nproc_per_node=8 tools/train_closedloop_stage1.py \
  --config configs/multimodal/stage1.yaml \
  --output-dir /data0/user/liwei/GENMO_outputs/multimodal/stage1_formal01 \
  --set train.max_steps=600000 --set data_loader.batch_size=8 \
  --set trainer.gradient_accumulation=8 --set data_loader.num_workers=4
```

完整恢复使用原配置、原 world size／batch／累积和匹配资产，加载 `latest.json` 所指向的完整编号 checkpoint。`--output-dir` 可以是没有更新 checkpoint 的原运行目录，或新的恢复目录；不能回滚覆盖已经存在的更新文件。

```bash
"$PY" -B tools/train_closedloop_stage1.py \
  --config configs/multimodal/stage1.yaml --output-dir "$STAGE1_RUN" \
  --resume-checkpoint "$STAGE1_RUN/checkpoints/s000008.pt" \
  --set train.max_steps=16
```

上例对应默认单卡 8 步实验；多卡正式恢复继续使用相同 torchrun 参数和训练覆盖项。checkpoint 保存 optimizer、scheduler、AMP、每 rank RNG／采样游标、共享 stats／kinematics 身份、T5 身份和 best 元数据。源码版本只用于追溯。

验证每来源固定样本与 seed，先按来源平均，再对 music/text 各 50% 计算 `raw_reconstruction_loss`，以它选择 `best.json` 的 `checkpoint`；`latest.json` 只是最后保存位置。总 loss 和模态误差分别记录，重建 best 不代表语义或执行 best。

```bash
export BUMI_MULTIMODAL_STAGE1="$("$PY" -c 'import json,os; print(json.load(open(os.environ["STAGE1_RUN"]+"/best.json"))["checkpoint"])')"
"$PY" -B tools/train_closedloop_stage1.py \
  --config configs/multimodal/stage1.yaml \
  --output-dir /data0/user/liwei/GENMO_outputs/multimodal/stage1_fixed_validation01 \
  --set mode=validate --set stage1_checkpoint="$BUMI_MULTIMODAL_STAGE1"
```

## 3. 音乐／文本生成

```bash
"$PY" -B tools/generate_bumi_multimodal.py \
  --checkpoint "$BUMI_MULTIMODAL_STAGE1" \
  --stats "$BUMI_MULTIMODAL_ASSETS/qpos30_train_stats.json" \
  --kinematics "$BUMI_KINEMATICS_PATH" --device cuda:0 \
  --audio /absolute/path/song.wav --output /absolute/new/music_generation01

"$PY" -B tools/generate_bumi_multimodal.py \
  --checkpoint "$BUMI_MULTIMODAL_STAGE1" \
  --stats "$BUMI_MULTIMODAL_ASSETS/qpos30_train_stats.json" \
  --kinematics "$BUMI_KINEMATICS_PATH" --device cuda:0 \
  --text-encoder-path "$BUMI_T5_MODEL_PATH" \
  --text "Walk forward and raise your right hand." --duration-s 8 \
  --output /absolute/new/text_generation01
```

音乐默认整首时长，文本必须提供时长，同时传两种条件会报错。超过 120 帧用已生成动作构造因果历史和 12 帧承诺前缀，起始采用显式站姿或 Stage2 任务初态。`reference.pt` 与 `motion.npz` 包含 30 Hz 世界 qpos28、接触概率及其有效性、具名关节；`generation.json` 记录可执行时长、采样参数、种子和资产身份。十帧控制预览／导数所需尾部由模型继续生成，不复制末帧；尾部不计入任务时长。输出是运动学参考，不是物理成功证明。

## 4. Stage2：新任务 → 新缓存 → 新标定 → 共享 Adapter

配套 SuperMimic 必须包含已有 bridge／WM 和 IsaacLab3 Newton 环境，Actor checkpoint 与 RobotSpec／MJCF 内容必须匹配。

```bash
export SUPERMIMIC_REPO=/home/user/liwei/bumi-closedloop-worktrees/adapter/SuperMimic
export SM_BRIDGE_CHECKPOINT=/data0/user/liwei/GENMO_assets/adapter_20261026/ckpt_024000.pt
export SM_ROBOT_ASSET_DIR=/data0/user/liwei/GENMO_assets/adapter_20261026/sm/data/bumi3_4340/robot

"$PY" -B tools/prepare_stage2_multimodal_assets.py \
  --data-config configs/multimodal/data.yaml --config configs/multimodal/stage2.yaml \
  --output "$BUMI_MULTIMODAL_ASSETS/stage2_tasks" \
  --train-per-source 8 --heldout-per-source 2

CUDA_VISIBLE_DEVICES=0 "$PY" -B tools/prepare_stage2_multimodal_reference.py \
  --config "$BUMI_MULTIMODAL_ASSETS/stage2_tasks/runtime.yaml" \
  --output "$BUMI_MULTIMODAL_ASSETS/stage2_cache" --device cuda:0

CUDA_VISIBLE_DEVICES=0 "$PY" -B tools/calibrate_stage2_multimodal.py \
  --config "$BUMI_MULTIMODAL_ASSETS/stage2_cache/runtime.yaml" \
  --output "$BUMI_MULTIMODAL_ASSETS/stage2_calibration" \
  --steps 1024 --num-envs 64 --visualizer none --deterministic
```

逐级使用上一步的 `runtime.yaml`，不要在后一步重新用未准备的模板。每来源默认 8 条 train、2 条 val、2 条 test，总计 96 个任务；数据不足会明确报错，不复制任务凑数。Stage1 val/test 分别映射 Stage2 validation/evaluation。文本时长限定在有效 caption 区间，最短默认 2 秒；任务初态只从区间首姿态构造，余下真值绝不充当名义缓存。所有缓存由选中的新 Stage1 实际生成，记录条件、初态、时长、seed、资产与 halo 来源。

标定只使用新 train 任务的零 Adapter **真实物理转移**，输出两模态等权关节速度 s0、精度误差标尺、零策略 Replay，并从新名义缓存重算六项参考连续性标尺。模板的 `music_baseline`、`alternating.calibration` 故意为空；只有标定输出 runtime 才具备正式训练资产。H64 奖励权重沿现有基线，不在准备时搜索参数。

```bash
export STAGE2_RUN=/data0/user/liwei/GENMO_outputs/multimodal/stage2_run01
CUDA_VISIBLE_DEVICES=0 "$PY" -B tools/train_stage2_alternating.py \
  --config "$BUMI_MULTIMODAL_ASSETS/stage2_calibration/runtime.yaml" \
  --output "$STAGE2_RUN" --visualizer none --deterministic
```

该入口是单 GPU 交替训练，不使用 DDP。Adapter 零输出初始化，WM 从匹配 Actor bundle 的预训练 WM 独立初始化；Actor 与 normalizer 冻结，Critic 默认关闭。保留 50 Hz 控制、200 Hz 物理、十帧残差、H8 真实 WM 标签、H64 Adapter BPTT、提交锚点与导数重算。环境 reset、WM Replay、Adapter 起点按两模态各 50% 采样；缺少某模态合法 H64 支持时不会悄悄缩短 H 或只抽另一种。

文本任务无需音频文件；音乐特征、有效标志与音乐时间均屏蔽，节拍奖励为零，真实／名义活动历史仍存在。静态指令的相对活动保持目标随名义参考降低，不要求产生舞蹈。跟踪奖励始终对**原始名义参考**计算。

```bash
export RESUME_STAGE2="$("$PY" -c 'import json,os; print(json.load(open(os.environ["STAGE2_RUN"]+"/latest_checkpoint.json"))["path"])')"
CUDA_VISIBLE_DEVICES=0 "$PY" -B tools/train_stage2_alternating.py \
  --config "$BUMI_MULTIMODAL_ASSETS/stage2_calibration/runtime.yaml" \
  --resume "$RESUME_STAGE2" \
  --output /data0/user/liwei/GENMO_outputs/multimodal/stage2_resume01 \
  --visualizer none --deterministic
```

Stage2 恢复必须用新的输出目录。完整 checkpoint 恢复网络／优化器／Replay／RNG，真实物理从新 episode 开始，不冒充恢复了求解器隐藏状态。`latest_checkpoint.json`、`zero_checkpoint.json`、`best_checkpoint.json` 分别指向最新、真实零策略和完整 validation-best；不能把最新模型当最佳。best 可仍是零 Adapter。

## 5. 固定质量评价

Stage1 的窗口重建监控与完整轨迹质量分开。使用固定 held-out 任务／seed，分别生成共享模型和原独立音乐／文本模型轨迹。旧模型沿各自原入口生成，导出相同 30 Hz qpos28／具名关节；不可把旧权重加载进新的联合主干冒充基线。配对 manifest 格式和离线评价命令如下：

```json
{"tasks": [{"task_id": "text_walk_01", "split": "test", "condition_kind": "text", "seed": 42, "duration_s": 8, "text": "Walk forward.", "models": {"joint": "/absolute/joint/reference.pt", "independent": "/absolute/baseline/motion.npz"}}]}
```

```bash
"$PY" -B tools/evaluate_bumi_multimodal_generation.py \
  --manifest /absolute/paired_tasks.json --kinematics "$BUMI_KINEMATICS_PATH" \
  --output /absolute/new/stage1_quality01 --blind-seed 42
```

音乐任务改用 `condition_kind=music` 和真实 `music_features`。报告分模型／模态列动作活动、速度／jerk、关节限位和音乐节拍；缺模型或非法轨迹明确计入无效项，报告照常落盘、CLI 返回 2。只有完整有效配对进入公平聚合。`reference.pt` 内的生成条件、seed／时长合同逐项核对；旧 NPZ 只有 qpos、joint_names、fps 时，缺失生成身份明确标为未核验，不能把 manifest 声明当证据。运动学足滑代理不等于真实求解器接触脚滑。文本没有可靠 BUMI 语义评测器时，使用 `blind_tasks.json/csv` 与匿名 `blind_assets` 开展类别、方向和符合度盲评；`metrics.json`、`reveal_key.json` 含模型身份，由组织者保管。人工分数未填写时为待评价，不生成伪造语义分数。

Stage2 自动在相同新缓存、初态、seed 和物理配置下比较零 Adapter 与候选。完整验证按两模态等权完成率优先、根位置 RMSE 其次、完整预算 Tracking 再次排序；各模态退化单独展示。节拍只对音乐适用，文本为 null／不适用。可显式重新运行完整配对评价：

```bash
export BUMI_MULTIMODAL_ADAPTER="$("$PY" -c 'import json,os; print(json.load(open(os.environ["STAGE2_RUN"]+"/best_checkpoint.json"))["path"])')"
export ADAPTER_UPDATES="$("$PY" -c 'import json,os; print(json.load(open(os.environ["STAGE2_RUN"]+"/best_checkpoint.json"))["adapter_updates"])')"
CUDA_VISIBLE_DEVICES=0 "$PY" -B tools/evaluate_stage2_dedicated.py \
  --config "$BUMI_MULTIMODAL_ASSETS/stage2_calibration/runtime.yaml" \
  --snapshot "$BUMI_MULTIMODAL_ADAPTER" --adapter-updates "$ADAPTER_UPDATES" \
  --num-envs 4 --output /absolute/new/stage2_paired01
```

不传 `--max-steps` 执行完整任务。任务数可超过 4，按批次完整覆盖。正式训练的早期短前缀报告不用于冒充完整任务 best。新生成分布可能超出冻结 Actor 能力，应分动作类别记录失败，扩大 Actor 能力不是本分支的隐含训练步骤。

## 6. 音乐／文本 MuJoCo sim2sim

```bash
export BUMI_MULTIMODAL_BASELINE="$BUMI_MULTIMODAL_ASSETS/stage2_calibration/music_baseline.pt"
"$PY" -B scripts/demo/run_genmo_sm_adapter_sim2sim.py \
  --config configs/multimodal/sim2sim.yaml --audio /absolute/path/song.wav

"$PY" -B scripts/demo/run_genmo_sm_adapter_sim2sim.py \
  --config configs/multimodal/sim2sim.yaml \
  --text "Raise both arms slowly." --duration-s 8
```

支持 `--headless --mute` 和既有零 Adapter 切换。文本自动静音，任务结束由显式时长决定。缓存绑定模型／统计／kinematics／条件／采样参数，文本另绑定 T5 内容。演示先生成完整名义参考再执行，实际物理反馈不回传生成器；不代表实时改写指令、摄像头连续输入或 Stage2-B 在线生成。

## 7. 验证与边界

回归覆盖纯音乐／纯文本／全语义 dropout、缺失条件占位不变性、caption／padding／halo／因果历史、前缀逐坐标保持、train-only 统计、双 rank 梯度同步、完整恢复、无音频时钟、50/30 Hz 对齐、Replay reset 边界、真实 WM 标签和冻结 Actor 梯度链。

本次完成 367 项集成回归和 8 项离线质量评价回归。临时真实 24 条源数据的 reader／联合统计通过；在 Server-2 使用真实冻结 T5、CPU FP32、32 维一层小主干、正式 120 帧／H50 窗口训练 2 步，再完整恢复到第 4 步。13 次实际 T5 编码调用覆盖 7 条独立描述，两模态支路梯度非零且参数更新；音乐 150 帧、文本 180 帧均完成两个窗口生成，前缀误差为 0。两条真实生成产物又通过离线 CLI，文本语义分保持空白；仅一个小模型，不是旧模型对比。双 rank 同步另用 CPU/Gloo 验证。

Stage2 的 CPU MuJoCo 验收使用合成参考，分别采集音乐／文本 16 个控制步，执行一次真实 H8 WM 更新和一次 H64 Adapter 更新；它只证明有界更新链路。完整最终执行结果见根目录 `记录文本.md`，其中失败原因与未执行项单独保留。

没有执行全数据正式长训练、新联合模型整曲／长文本质量排名、新模型完整 Newton 配对收益或硬件测试；没有联合 ONNX 导出。摄像头／视频需同步视觉条件与重定向机器人目标、因果延迟处理和独立验收，此版没有暴露这类未经训练的输入。
