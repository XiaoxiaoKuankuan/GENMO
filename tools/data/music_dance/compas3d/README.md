# CoMPAS3D → GENMO 数据转换

这套工具只处理本地已经完整下载的数据，不执行 Git LFS、Hugging Face 或其他网络下载。

## 实际数据格式

当前发布 NPZ 的真实 key 为：

```text
gender                  scalar string
surface_model_type      scalar string: smplx_locked_head
mocap_frame_rate        scalar float64: 30.0
betas                   [300] float64
poses                   [T,165] float64
trans                   [T,3] float64
markers_obs             [T,53,3] object（实际内容为 finite 数值）
markers_sim             [T,53,3] object（实际内容为 finite 数值）
v_template              object scalar: None
```

`poses` 是 55 个 SMPL-X 关节的 axis-angle：

```text
0:3       global_orient
3:66      body_pose（21 × 3）
66:69     jaw_pose
69:72     left_eye_pose
72:75     right_eye_pose
75:120    left_hand_pose（15 × 3）
120:165   right_hand_pose（15 × 3）
```

GENMO 只保留前 66D，不改变 body joint 顺序；face、eyes、hands 不进入 GENMO pose。
原始 300D shape betas 保留在输出 metadata 的 `source_smplx.betas_300`，训练接口继续使用
neutral zero betas `[T,10]`，不修改 GENMO 网络契约。

数据集 README 中的 Vicon 采集频率是 120 FPS，但下载后的每个 MoSh NPZ 都明确保存
`mocap_frame_rate=30.0`。因此当前文件保持原帧，不允许再次 4:1 下采样。转换器仍兼容以后
可能获得的 120 FPS NPZ：使用同一组 `0,4,8,...` 索引同时选 pose 和 translation；其他
FPS 使用现有 quaternion SLERP helper，绝不对 axis-angle 做普通线性插值。

源 SMPL-X/MoSh 坐标是 Z-up、米制，GENMO 是 Y-up、米制。转换使用：

```text
(x, y, z)_source -> (x, z, -y)_GENMO
```

SMPL-X 根旋转围绕 shaped pelvis，而不是世界原点。因此 global orientation 与 translation
变换还包含 source gender/300D shape pelvis 到 GENMO neutral/10D zero shape pelvis 的 offset
补偿。SMPL-X forward 数值验证表明，带补偿的坐标变换与整个人体网格刚体旋转一致；不做
补偿会产生约 0.3～0.44 m 的错误平移。

## 检查原始数据

```bash
cd /home/user/liwei/GENMO

.venv/bin/python tools/data/music_dance/compas3d/inspect_compas3d.py \
  --root /data0/user/liwei/datasets/music_dance_raw/CoMPAS3D \
  --reference-root /data0/user/liwei/datasets/music_dance_raw/compas3d_repair \
  --output /data0/user/liwei/datasets/music_dance_genmo/CoMPAS3D/reports/inspection.json \
  --summary-output /data0/user/liwei/datasets/music_dance_genmo/CoMPAS3D/reports/inspection_summary.txt \
  --npz-sample-count 20 \
  --media-sample-count 12 \
  --strict
```

文件角色通过 stem 是否包含 `leader` / `follower` 判断。这样官方异常文件
`Pair7_song2_take1_leaderi.npz` 会被正确识别，不能用严格 `_leader.npz` 后缀硬拼路径。

## 默认 split

官方 README 给出了面向交互动作 benchmark 的逐 Pair 验证/测试序列，但同一 `songX` 会
同时出现在 train、val、test。对 music-conditioned generation，这会泄漏完全相同的音乐。

默认使用 `music_identity`：

```text
train: song1 + song2
val:   song3
test:  song4
```

同一歌曲的所有 Pair/take 以及一个 sequence 的 leader/follower 永远属于同一 split。官方
交互划分仍可通过 `--split-strategy official_interaction` 复现，但报告会明确标注音乐泄漏。

## 5-sequence smoke

请使用独立、空的 smoke 输出目录：

```bash
.venv/bin/python tools/data/music_dance/compas3d/convert_compas3d_to_genmo.py \
  --root /data0/user/liwei/datasets/music_dance_raw/CoMPAS3D \
  --reference-root /data0/user/liwei/datasets/music_dance_raw/compas3d_repair \
  --output-root /data0/user/liwei/datasets/music_dance_genmo/CoMPAS3D-smoke5 \
  --sequence-id Pair1_song1_take1 \
  --sequence-id Pair2_song2_take1 \
  --sequence-id Pair3_song3_take1 \
  --sequence-id Pair4_song4_take1 \
  --sequence-id Pair5_song2_take1 \
  --split-strategy music_identity \
  --strict

.venv/bin/python tools/data/music_dance/compas3d/validate_compas3d_genmo.py \
  --root /data0/user/liwei/datasets/music_dance_genmo/CoMPAS3D-smoke5 \
  --smpl-forward-samples 10 \
  --strict
```

## 当前完整子集全量转换

```bash
.venv/bin/python tools/data/music_dance/compas3d/convert_compas3d_to_genmo.py \
  --root /data0/user/liwei/datasets/music_dance_raw/CoMPAS3D \
  --reference-root /data0/user/liwei/datasets/music_dance_raw/compas3d_repair \
  --output-root /data0/user/liwei/datasets/music_dance_genmo/CoMPAS3D \
  --split-strategy music_identity \
  --strict

.venv/bin/python tools/data/music_dance/compas3d/validate_compas3d_genmo.py \
  --root /data0/user/liwei/datasets/music_dance_genmo/CoMPAS3D \
  --smpl-forward-samples 10 \
  --strict
```

每个 MP4 只提取一次音频并只计算/保存一份 EDGE baseline35。leader 与 follower manifest
引用同一个 `music_feature_path`。MP4/AAC 比 motion 一致地多约 3.5～5.6 个 30 Hz 尾帧，
转换器先显式截取 motion 对应时长；EDGE35 的 STFT 仍多出的 1 帧经过记录后裁掉。明显短音频
或超过审计上限的长音频会报错，不会静默对齐。

## 缺失数据补齐后的增量转换

补齐文件后重复运行同一全量转换命令即可。转换器会：

1. 重新扫描真实 MP4 和可加载 NPZ；
2. 重新严格验证已经导出的 sequence；
3. 跳过验证通过的输出；
4. 只转换新近变完整的 sequence；
5. 更新 manifests、split 和 incomplete reports。

只有明确希望重算所有 EDGE35 和 motion 时才使用 `--overwrite`。

## 输出

```text
CoMPAS3D/
├── motions/          # 每个角色一个 canonical SMPL body .pt
├── musicfeat_v2/     # 每个 sequence 一份 [T,35] EDGE baseline35
├── manifests/        # train.jsonl / val.jsonl / test.jsonl / groups.jsonl
├── reports/          # inspection/conversion/validation/split/incomplete reports
├── renders/          # 人工验收视频
└── audio_cache/      # 从 MP4 截取到 motion 时长的单声道 WAV cache
```
