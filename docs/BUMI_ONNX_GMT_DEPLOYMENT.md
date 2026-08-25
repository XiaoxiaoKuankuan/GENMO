# BUMI ONNX 到 GMT 部署链路

> 当前 ONNX 边界已升级为 `pred_motion[1,120,30]` 与
> `pred_foot_contact_logits[1,120,2]` 双输出；link 全由 qpos FK 得到，正式后处理只允许
> contact-gated root XY 足底锁定。新契约见
> [BUMI qpos30、FK 接触与足底锁定 v3](BUMI_QPOS30_CONTACT_V3.md)。下文 93D 单输出内容
> 仅作旧版部署记录，不能用于新模型。

本文主体记录 `feature/bumi-music-only` 分支的历史 93D 部署链。该版本输出机器人原生
93D 特征和 MuJoCo 顺序 `qpos28`，链路中不再经过 SMPL/GMR：

> 重要：下文 `s430000`、旧 stats、ONNX、TensorRT 和视频是 v1 `root_pos_local`
> 历史审计结果，只能用于对比，不能由当前 v2 代码加载或继续部署。当前代码要求
> `genmo.bumi_motion_features.v2`，必须重算 stats、重新训练并导出新 ONNX/engine；这是
> 有意的失败保护，不能因为张量同为 93D 而绕过。

```text
WAV
  → EDGE35 @ 30 Hz
  → BUMI ONNX/TensorRT 单步去噪 [1,120,93]
  → DDIM 20 步 + 120/30 独立窗口
  → 世界水平增量/绝对根高/关节 overlap-add + 根四元数 SLERP
  → 全序列只积分一次水平根位移
  → 连续世界系 qpos28 @ 30 Hz
  → CRC/revision/模型指纹 + 实时安全门
  → 线性插值/四元数 SLERP @ 50 Hz
  → GMT trajectory_v1 110×55
  → Redis + GMT ACK
```

## 两个术语

多 checkpoint 批量评测、排序、选优，是让多个训练存档在同一套音乐、目标动作、seed、
CFG 和 DDIM 步数下生成动作，汇总关节限位、穿地、滑脚、平滑性、根稳定性、节拍和可选
GT 误差，先按硬安全门槛淘汰，再按归一化综合分排序。最优模型不保证是 step 最新、训练
loss 最低的存档。

Parity 是数值等价检查。本项目固定相同 checkpoint、输入和噪声，比较 PyTorch、ONNX
Runtime、TensorRT 的单步 93D，以及完整 DDIM 后的 93D、qpos28 和 Torch FK；浮点算子
实现不同，所以要求在明确容差内一致，不要求逐 bit 相同。

## 历史 v2 s350000 导出与 50 首网页验证（2026-08-24）

当前人工 q1 五库基线使用以下本地资产；`meshes/` 必须与 `bumi3.xml` 同目录，否则 MuJoCo
会在推理产物已经生成后统一渲染失败：

```bash
cd /home/weili/GENMO

BUMI_TASK_CKPT=inputs/checkpoints/bumi_5set_manual_q1_v3/s350000/last.ckpt
BUMI_TASK_KIN=inputs/assets/bumi_manual_q1_v3/bumi_kinematics_482138_v1.json
BUMI_TASK_STATS=inputs/assets/bumi_manual_q1_v3/bumi_93d_stats_train_manual_q1_v3.json
BUMI_TASK_MJCF=inputs/assets/bumi_manual_q1_v3/bumi3.xml
BUMI_TASK_ONNX=outputs/onnx/bumi_music/s350000_hq50/bumi_music_denoiser_t120.onnx
BUMI_TASK_AUDIO=data/server_music_wav_4set_all_20260818/aistpp/wav/mJS5.wav
BUMI_TASK_OUT=outputs/onnx/bumi_music/s350000_hq50
```

导出 120 帧、opset 18 的 guided denoiser：

```bash
.venv/bin/python tools/export/export_bumi_music_onnx.py \
  --ckpt "$BUMI_TASK_CKPT" \
  --output "$BUMI_TASK_ONNX" \
  --exp gem_bumi_music_only_5set_manual_q1_v3 \
  --kinematics "$BUMI_TASK_KIN" \
  --stats "$BUMI_TASK_STATS" \
  --seq-len 120 \
  --opset 18 \
  --device cuda:0 \
  --overwrite
```

严格基准使用 CPU provider；CUDA provider 另跑一份诊断，不用 CUDA 中间 93D 的严格失败
覆盖 CPU 的可移植导出结论：

```bash
.venv/bin/python tools/eval/validate_bumi_music_onnx.py \
  --audio "$BUMI_TASK_AUDIO" \
  --ckpt "$BUMI_TASK_CKPT" \
  --onnx "$BUMI_TASK_ONNX" \
  --exp gem_bumi_music_only_5set_manual_q1_v3 \
  --kinematics "$BUMI_TASK_KIN" \
  --stats "$BUMI_TASK_STATS" \
  --seq-len 120 \
  --provider cpu \
  --device cpu \
  --full-ddim-steps 20 \
  --output-dir "$BUMI_TASK_OUT/parity_cpu"

# 将 --provider/--device 改为 cuda/cuda:0，并把输出目录改为 $BUMI_TASK_OUT/parity，
# 可复现 CUDA 诊断。
```

50 首验证固定取音乐起点的前 8 秒，以 `FineDance=30、CoMPAS3D=5、AIOZ-GDance=10、
AIST++=5` 的配额从人工 `score=1` 音频全集均匀抽样；相同命令可恢复正式报告，也会复用
已原子落盘但尚未成功渲染的动作产物：

```bash
MUJOCO_GL=egl .venv/bin/python scripts/validate_bumi_hq_music_full.py \
  --audio-root data/server_music_wav_4set_all_20260818 \
  --ratings-root data/motions_npz_bumi3_smooth_q1/rate \
  --checkpoint "$BUMI_TASK_CKPT" \
  --onnx "$BUMI_TASK_ONNX" \
  --kinematics "$BUMI_TASK_KIN" \
  --stats "$BUMI_TASK_STATS" \
  --mjcf "$BUMI_TASK_MJCF" \
  --output-root "$BUMI_TASK_OUT/validation_hq50_8s_20260824" \
  --finedance-count 30 \
  --compas3d-count 5 \
  --aioz-gdance-count 10 \
  --aistpp-count 5 \
  --max-duration-sec 8 \
  --onnx-provider cuda \
  --device cuda:0 \
  --ddim-steps 20 \
  --cfg-scale 2.5 \
  --seed 42 \
  --joint-limit-tolerance-rad 0.25 \
  --width 640 \
  --height 480

MUJOCO_GL=egl .venv/bin/python scripts/build_bumi_hq_original_comparison.py \
  --validation-root "$BUMI_TASK_OUT/validation_hq50_8s_20260824" \
  --source-motion-root data/motions_npz_bumi3_smooth_q1 \
  --quality-config configs/bumi/quality_filter_sonic_npz_50hz_auto025_v1.yaml \
  --kinematics "$BUMI_TASK_KIN" \
  --mjcf "$BUMI_TASK_MJCF" \
  --output-root "$BUMI_TASK_OUT/validation_hq50_8s_20260824/comparison_original_vs_generated" \
  --joint-limit-tolerance-rad 0.25 \
  --width 640 \
  --height 480
```

实际导出 ONNX 为 `864271583` bytes，SHA256 是
`1e6c1a73469785be646665fd3df8ef11bc9b40beacbdf0f0e5fd7e1a8897e670`。CPU 单步、完整
93D、qpos 和 FK 全部通过；CUDA 的完整 qpos/FK 仍在 `0.02` 容差内，但单步和完整 93D
未通过严格 allclose。50 项均完成生成与并排渲染，总对比 399.4 秒；AIST++ `mWA5` 的原
动作只有 7.4 秒，按真实长度保留，没有循环或拉伸。

## 历史 v1 模型和资产（仅审计）

- 服务器 2 checkpoint：
  `/data0/user/liwei/experiments/genmo/gem_bumi_music_only_4set_random_v1/version_0/checkpoints/s430000.ckpt`
- 本地 checkpoint：
  `/home/weili/GENMO/outputs/checkpoints/server2_bumi_random_v1/s430000.ckpt`
- 本地训练统计：
  `/home/weili/GENMO/outputs/checkpoints/server2_bumi_random_v1/bumi_93d_stats_train_v1.json`
- 运动学：`/home/weili/GENMO/configs/bumi/bumi_kinematics_482138_v1.json`
- GMT policy：
  `/home/weili/docker_projects/bumi_GMT_deployment_listao/bumi_GMT_deployment_listao/src/legged_rl/rl_controller/rl_controllers/policy/bumi/0724_lab_148500.onnx`

已核对运动学 SHA256 为
`226306c217a39643a32ccc641b63e72c8ce0bf307af1fd6781b739fb694bcda0`，训练统计 SHA256
为 `3c642a9b7fa07e5639a508ce7bf1fca9705615f6646097d81c9911c3b80c646a`，`s430000`
checkpoint 的服务器端 SHA256 为
`a61e4e4a7629304ed06e00f5479fcd059337bc7a9478b05bb8f7a04d5f501b58`。GMT policy
中的 21 关节集合与 BUMI MuJoCo 顺序完全一致，发布边界使用显式 permutation，不在模型
内部改变关节顺序。

本次真实导出/验证结果：ONNX 大小 `864271583` bytes，SHA256 为
`95bf637da38311d53150d8ba51d39ede9848859c81d59fcc08f83d5652ae1a56`；RTX 4090、
TensorRT/libnvinfer 10.13.3 的 FP16 engine 大小 `433320476` bytes，SHA256 为
`41dea596316e2d85f346ad3cb349efaff7083f8c297e15cd649e1543d54f1466`。PyTorch↔ONNX
单步/20 步 DDIM 通过，三后端报告 `final_pass=true`。4 秒 MuJoCo+音乐验证视频保存在
`outputs/onnx/bumi_music/s430000/demo_mJS3.mp4`。

下面命令保留为历史复现记录；当前分支会明确拒绝旧 checkpoint/stats，因此不能直接执行：

```bash
cd /home/weili/GENMO

BUMI_TASK_CKPT=/home/weili/GENMO/outputs/checkpoints/server2_bumi_random_v1/s430000.ckpt
BUMI_TASK_KIN=/home/weili/GENMO/configs/bumi/bumi_kinematics_482138_v1.json
BUMI_TASK_STATS=/home/weili/GENMO/outputs/checkpoints/server2_bumi_random_v1/bumi_93d_stats_train_v1.json
BUMI_TASK_ONNX=/home/weili/GENMO/outputs/onnx/bumi_music/s430000/bumi_music_denoiser_t120.onnx
BUMI_TASK_AUDIO=/home/weili/datasets/AISTPP_official/music/wav/mJS3.wav
BUMI_TASK_GMT=/home/weili/docker_projects/bumi_GMT_deployment_listao/bumi_GMT_deployment_listao/src/legged_rl/rl_controller/rl_controllers/policy/bumi/0724_lab_148500.onnx
BUMI_TASK_ENGINE=/home/weili/GENMO/outputs/tensorrt/bumi/s430000/1d4dda3f78c62d99c2ffcfae71cdd1f64c8f176aafd5d86cae658d2768e815ab/bumi_music_denoiser.engine

export BUMI_KINEMATICS_PATH="$BUMI_TASK_KIN"
export BUMI_MUSIC_STATS_PATH="$BUMI_TASK_STATS"
```

## 导出 ONNX

```bash
.venv/bin/python tools/export/export_bumi_music_onnx.py \
  --ckpt "$BUMI_TASK_CKPT" \
  --output "$BUMI_TASK_ONNX" \
  --exp gem_bumi_music_only_4set_random_v1 \
  --kinematics "$BUMI_TASK_KIN" \
  --stats "$BUMI_TASK_STATS" \
  --seq-len 120 \
  --opset 18 \
  --device cuda:0 \
  --overwrite
```

导出物包括 ONNX、可能的外部权重文件，以及
`bumi_music_denoiser_t120.onnx.json`。JSON 记录 checkpoint/运动学/统计 SHA 和固定输入输出
合约；后续 GMT demo 会强制核验，不能把不匹配的 stats 或 checkpoint 混进来。

## ONNX 验证和渲染 demo

先做单步与完整 20 步 DDIM parity：

```bash
.venv/bin/python tools/eval/validate_bumi_music_onnx.py \
  --audio "$BUMI_TASK_AUDIO" \
  --ckpt "$BUMI_TASK_CKPT" \
  --onnx "$BUMI_TASK_ONNX" \
  --exp gem_bumi_music_only_4set_random_v1 \
  --kinematics "$BUMI_TASK_KIN" \
  --stats "$BUMI_TASK_STATS" \
  --seq-len 120 \
  --provider cuda \
  --device cuda:0 \
  --full-ddim-steps 20 \
  --output-dir outputs/onnx/bumi_music/s430000/validation
```

再用 checkpoint 路径运行正式生成、运动学指标和 MuJoCo 视频；这是可视化验证，不代表
GMT 动力学已经通过：

```bash
.venv/bin/python scripts/demo/demo_music_bumi.py \
  --wav "$BUMI_TASK_AUDIO" \
  --checkpoint "$BUMI_TASK_CKPT" \
  --output outputs/onnx/bumi_music/s430000/demo_mJS3.pt \
  --report outputs/onnx/bumi_music/s430000/demo_mJS3.json \
  --exp gem_bumi_music_only_4set_random_v1 \
  --kinematics "$BUMI_TASK_KIN" \
  --stats "$BUMI_TASK_STATS" \
  --duration-sec 4 \
  --ddim-steps 20 \
  --cfg-scale 2.5 \
  --seed 42 \
  --device cuda:0 \
  --render-mjcf /home/weili/GMR_minimal_robots_smplx.tar.gz/GMR-master/assets/bumi3/bumi3.xml \
  --video outputs/onnx/bumi_music/s430000/demo_mJS3.mp4 \
  --width 640 \
  --height 480 \
  --world-root-x 0 \
  --world-root-y 0 \
  --world-root-yaw 0 \
  --mux-audio
```

BUMI 93D 训练表示是首帧 XY/yaw、默认根高归一化后的 canonical 坐标，不能直接当作 MJCF
世界坐标渲染。Demo 在启用 `--render-mjcf` 且没有显式给出世界位置时，会自动使用
`root_xy=[0,0]、anchor_z=0.65、yaw=0`；上面的命令仍显式写出 XY/yaw，便于复现实验。

四数据集人工高质量完整音乐批量验证使用评分表中的 `score=1`，按共享音频键去重并在每库
候选全集上均匀取样。工具复用一次 ONNX 会话执行 120/30 长音乐滑窗，生成带声音视频、
严格 XML/容差后逐关节限位报告和按 FineDance、CoMPAS3D、AIOZ-GDance、AIST++ 排序的
网页；中断后用同一命令可复用已完成项：

```bash
MUJOCO_GL=egl .venv/bin/python scripts/validate_bumi_hq_music_full.py \
  --audio-root outputs/server_music_wav_4set_all_20260818 \
  --ratings-root data/motions_npz_bumi3_smooth_q1/rate \
  --checkpoint "$BUMI_TASK_CKPT" \
  --onnx "$BUMI_TASK_ONNX" \
  --kinematics "$BUMI_TASK_KIN" \
  --stats "$BUMI_TASK_STATS" \
  --mjcf /home/weili/GMR_minimal_robots_smplx.tar.gz/GMR-master/assets/bumi3/bumi3.xml \
  --output-root outputs/onnx/bumi_music/s430000/validation_hq_4set_full_20260821 \
  --per-dataset 10 \
  --onnx-provider cuda \
  --device cuda:0 \
  --ddim-steps 20 \
  --cfg-scale 2.5 \
  --seed 42 \
  --joint-limit-tolerance-rad 0.25 \
  --width 640 \
  --height 480
```

输出根目录的 `index.html` 可直接打开，也可在该目录启动 `python -m http.server` 供浏览器
访问。`quality_summary.json` 同时保留 0、0.05、0.10、0.15、0.20、0.25 rad 的样本级
限位敏感性；页面中的关节编号为 1-based，JSON 同时保存 0-based 编号。

在上面的 40 首模型视频完成后，可将每首 ``score=1`` 代表动作的原始 50 Hz GMR BUMI
轨迹与模型结果做成同步左右对比。转换严格复用正式数据构建器的 50→30 Hz 插值、SLERP、
关节重排和地面规范；AIST++ 原始动作只有 7–12 秒时按真实片段截断，不循环原动作：

```bash
MUJOCO_GL=egl .venv/bin/python scripts/build_bumi_hq_original_comparison.py \
  --validation-root outputs/onnx/bumi_music/s430000/validation_hq_4set_full_20260821 \
  --source-motion-root data/motions_npz_bumi3_smooth_q1 \
  --quality-config configs/bumi/quality_filter_sonic_npz_50hz_auto025_v1.yaml \
  --kinematics "$BUMI_TASK_KIN" \
  --mjcf /home/weili/GMR_minimal_robots_smplx.tar.gz/GMR-master/assets/bumi3/bumi3.xml \
  --output-root outputs/onnx/bumi_music/s430000/validation_hq_4set_full_20260821/comparison_original_vs_generated \
  --joint-limit-tolerance-rad 0.25 \
  --width 640 \
  --height 480
```

输出对比视频为 1280×480：左侧 ``Original GMR BUMI``，右侧
``s430000 Generated``。目录自包含原动作视频、模型视频和合成对比视频；同盘模型视频优先
硬链接，跨盘才复制。`index.html` 提供合成播放、单独原动作、完整模型视频和同区间指标。

## TensorRT 构建与 parity

TensorRT plan 与 GPU 型号、TensorRT/libnvinfer 主次版本绑定，应在最终部署机器上构建：

```bash
.venv/bin/python tools/export/build_bumi_music_tensorrt.py \
  --onnx "$BUMI_TASK_ONNX" \
  --checkpoint "$BUMI_TASK_CKPT" \
  --output-dir outputs/tensorrt/bumi/s430000 \
  --device cuda:0 \
  --precision fp16
```

构建命令最后一行会打印实际 `.engine` 路径。上面的 `BUMI_TASK_ENGINE` 是本次 RTX 4090
构建产物；换 GPU、TensorRT/libnvinfer 主次版本或 ONNX/checkpoint 后必须使用新路径。
三后端 parity 命令：

```bash
.venv/bin/python tools/eval/validate_bumi_music_tensorrt.py \
  --audio "$BUMI_TASK_AUDIO" \
  --ckpt "$BUMI_TASK_CKPT" \
  --onnx "$BUMI_TASK_ONNX" \
  --engine "$BUMI_TASK_ENGINE" \
  --exp gem_bumi_music_only_4set_random_v1 \
  --kinematics "$BUMI_TASK_KIN" \
  --stats "$BUMI_TASK_STATS" \
  --device cuda:0 \
  --onnx-provider cuda \
  --ddim-steps 20 \
  --output outputs/tensorrt/bumi/s430000/parity.json
```

只有 `final_pass=true` 的引擎才进入机器人链路。

## 长音乐 ONNX + GMT

先 dry-run。它会走完整安全流和 30→50/GMT 转换，只保存计划，不写 Redis：

```bash
.venv/bin/python scripts/demo/demo_bumi_onnx_gmt.py \
  --audio "$BUMI_TASK_AUDIO" \
  --duration-sec 20 \
  --checkpoint "$BUMI_TASK_CKPT" \
  --onnx "$BUMI_TASK_ONNX" \
  --backend onnx \
  --onnx-provider cuda \
  --kinematics "$BUMI_TASK_KIN" \
  --stats "$BUMI_TASK_STATS" \
  --gmt-policy "$BUMI_TASK_GMT" \
  --device cuda:0 \
  --ddim-steps 20 \
  --cfg-scale 2.5 \
  --seed 42 \
  --output outputs/bumi_onnx_gmt/s430000_mJS3_20s_plan.npz
```

`NPZ` 包含 `qpos_30hz`、`qpos_50hz`、`gmt_frames_50hz` 和 `native_to_gmt`；相邻 JSON
记录所有模型/资产 SHA、安全阈值、分块数、执行状态和
`sliding_qpos_contract_version=genmo.bumi_sliding_motion_overlap_add.v3`。v3 不再使用
30 帧 DDIM 硬历史；每窗独立生成后在完整重叠区融合物理水平增量、绝对根高、根旋转和
关节，最后只积分一次水平轨迹。旧 v1/v2 artifact/report 不会被当作当前结果复用。

以下数值是旧硬拼接运行时的历史诊断，不能代表 overlap-add v2；必须用当前代码重新生成
后再决定安全门结果。旧 `s430000 + mJS3 + seed=42` 曾被严格默认安全门拒绝：最大关节限位样例是
`r_arm_roll_joint=0.206278`，相对 XML 上限 0.14 在加 0.05 rad 容差后仍超 0.016278；
按此前数据筛选约定显式改为 0.25 rad 后，最大手臂速度 28.781752 rad/s 仍超过默认
18 rad/s，根角速度 15.373582 rad/s 也超过默认 8 rad/s。因此该样例不能进入实物执行。

下面命令只用于证明 20 秒、7 个 120/30 滑窗、安全流编解码、600 帧 30 Hz→999 个动作
采样点 50 Hz、过渡/未来上下文和 GMT 55D 数据契约完整贯通。它会产生 1200 帧含过渡的
离线诊断计划；这些放宽值不得复制到 `--execute`：

```bash
.venv/bin/python scripts/demo/demo_bumi_onnx_gmt.py \
  --audio "$BUMI_TASK_AUDIO" \
  --duration-sec 20 \
  --checkpoint "$BUMI_TASK_CKPT" \
  --onnx "$BUMI_TASK_ONNX" \
  --backend tensorrt \
  --engine "$BUMI_TASK_ENGINE" \
  --kinematics "$BUMI_TASK_KIN" \
  --stats "$BUMI_TASK_STATS" \
  --gmt-policy "$BUMI_TASK_GMT" \
  --device cuda:0 \
  --ddim-steps 20 \
  --cfg-scale 2.5 \
  --seed 42 \
  --joint-limit-tolerance-rad 0.25 \
  --max-joint-velocity-radps 35 \
  --max-root-angular-velocity-radps 20 \
  --output outputs/bumi_onnx_gmt/s430000_mJS3_20s_tensorrt_diagnostic_plan.npz
```

连接已经实现 `trajectory_v1` 和 ACK 的 GMT 进程后，实际发布必须显式双确认。以下命令
保留默认安全阈值；当前 `s430000/mJS3/seed=42` 会在写 Redis 前被拒绝，必须先由多
checkpoint 选优或模型改进得到通过默认安全门的候选：

```bash
.venv/bin/python scripts/demo/demo_bumi_onnx_gmt.py \
  --audio "$BUMI_TASK_AUDIO" \
  --duration-sec 20 \
  --checkpoint "$BUMI_TASK_CKPT" \
  --onnx "$BUMI_TASK_ONNX" \
  --backend onnx \
  --onnx-provider cuda \
  --kinematics "$BUMI_TASK_KIN" \
  --stats "$BUMI_TASK_STATS" \
  --gmt-policy "$BUMI_TASK_GMT" \
  --device cuda:0 \
  --output outputs/bumi_onnx_gmt/s430000_mJS3_20s_execute.npz \
  --redis-host 127.0.0.1 \
  --redis-key gmt_online_frame_bumi \
  --revision 1 \
  --execute \
  --confirm-robot-motion
```

TensorRT 版本只需改为：

```bash
--backend tensorrt --engine "$BUMI_TASK_ENGINE"
```

## 多 checkpoint 批量选优

套件 JSON 固定评测样本。路径支持绝对路径、相对套件文件路径或环境变量：

```json
{
  "contract_version": "genmo.bumi_checkpoint_suite.v1",
  "samples": [
    {
      "id": "aist_mJS3_0s",
      "dataset": "AIST++",
      "wav": "$AISTPP_ROOT/music/wav/mJS3.wav",
      "start_sec": 0.0,
      "num_frames": 120
    }
  ]
}
```

实际选优应从 FineDance、CoMPAS3D、AIOZ-GDance、AIST++ 各取多段固定样本，而不是只用
上面的单条示例。运行命令：

```bash
export AISTPP_ROOT=/home/weili/datasets/AISTPP_official

.venv/bin/python tools/eval/select_bumi_checkpoints.py \
  --checkpoints outputs/checkpoints/server2_bumi_random_v1/s380000.ckpt \
                outputs/checkpoints/server2_bumi_random_v1/s400000.ckpt \
                outputs/checkpoints/server2_bumi_random_v1/s420000.ckpt \
                outputs/checkpoints/server2_bumi_random_v1/s430000.ckpt \
  --suite /absolute/path/bumi_checkpoint_suite.json \
  --output-dir outputs/eval/bumi_checkpoint_selection_v1 \
  --exp gem_bumi_music_only_4set_random_v1 \
  --kinematics "$BUMI_TASK_KIN" \
  --stats "$BUMI_TASK_STATS" \
  --device cuda:0 \
  --ddim-steps 20
```

脚本默认删除中间 motion `.pt`，保留逐样本 JSON、`selection.json` 和
`best_checkpoint.txt`。如果没有候选通过关节限位、最大穿地和根倾角硬门槛，脚本会保留
完整报告并以失败退出，不会在不安全候选中强行选一个。

## 安全边界

- 运行时拒绝 NaN/Inf、非单位 wxyz 四元数、XML 关节限位超界、根高度越界、根/关节速度
  越界，并覆盖分块边界。
- qpos 包拒绝 CRC 错误、帧号 gap/duplicate、旧 revision 及中途改变模型/引擎/运动学身份。
- GMT 发布前检查 policy 的三个输入形状、关节 metadata 和显式 reorder；运动前必须收到
  同一 stream/revision/plan 的 ACK，播放期间 ACK 过期立即终止。
- dry-run、ONNX/TensorRT parity 和 MuJoCo 视频通过，只能证明数值/运动学/协议链路；真实
  机器人仍必须先做仿真闭环、限速场地和急停值守测试。
