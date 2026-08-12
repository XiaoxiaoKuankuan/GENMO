# Music-only GEM-SMPL 训练与验证

本文档描述独立的 `gem_smpl_music_only` specialist。它不是把 generalist 的图像、
相机、音频和文本条件填零，而是在配置和网络入口层只保留一个条件：

```text
condition: music_embed [B, T, 35]
pipeline.args.in_attr: ["encoded_music"]
target: target_x [B, T, 151]
```

`music_embed` 是 `gem/utils/music_features.py` 定义的 EDGE baseline35，不接收 raw
WAV。WAV 必须先离线提取为 `[T,35]`。训练采用 diffusion-only；AIST++ 的相机、
2D、图像零占位仍可用于统一 batch 和几何监督，但绝不会进入 denoiser 的
`f_cond`。

## 1. 模型和数据流

```text
WAV
  ↓ tools/data/aistpp/extract_musicfeat_v2.py
EDGE baseline35
  ↓
music_embed [B,T,35]
  ↓ music_embedder
latent condition
  +
noisy motion [B,T,151]
  ↓ diffusion transformer
predicted motion [B,T,151]
  ↓ unchanged EnDecoder
SMPL motion
```

151D motion representation 沿用现有 EnDecoder 和 normalization statistics：

| 切片 | 维数 | 含义 |
|---|---:|---|
| `0:126` | 126 | 21 个 SMPL body joints × 6D rotation |
| `126:136` | 10 | SMPL betas |
| `136:142` | 6 | camera-space root orientation |
| `142:148` | 6 | gravity-view root orientation |
| `148:151` | 3 | local root translation velocity |

模型保留 151D simple diffusion loss、3D joint/vertex、2D projection、global
translation rollout 和 static-contact loss。这里的 music-only 仅指 condition
architecture，不表示只训练一个 MSE。

CFG 训练只有一套 dropout：`music_mask_prob=0.1` 按 sample 将整段音乐置零，约
90% sample 使用真实音乐、约 10% sample 训练 unconditional branch。
`disable_random_null_condition=true` 禁止再叠加逐帧 `uncond_prob` dropout。

## 2. 数据目录

训练前目录至少应为：

```text
inputs/AIST++/
├── annot_aist_30fps.pt
├── train.pt
├── val.pt
├── test.pt
├── minitrain.pt
└── musicfeat_v2/
    ├── gXX_..._musicfeat_fps30.pt
    └── ...
```

official crossmodal split 应为 train/val/test = 980/20/20，三者不重叠。每个
music feature 必须是 float、finite 的 `[T,35]`，其中通道 33/34 满足既有
baseline35 contract。

激活仓库环境后执行下列命令；如果不激活，也可把每条命令的 `python` 换成
`.venv/bin/python`。

## 3. 从 WAV 提取 baseline35

单个 WAV：

```bash
python tools/data/aistpp/extract_musicfeat_v2.py \
  --audio /path/to/music.wav \
  --output outputs/music_embed.pt
```

若输出已存在并确认需要覆盖，再加 `--overwrite`。

AIST++ batch 模式要求 `--aligned-wav-dir` 中存在与每个 motion sequence 同 stem
的 WAV：

```bash
python tools/data/aistpp/extract_musicfeat_v2.py \
  --annotations-root /home/weili/datasets/AISTPP_fullset/aist_plusplus_final \
  --aligned-wav-dir /home/weili/datasets/AISTPP_fullset/music_prepare/aligned_wav_official \
  --output-dir inputs/AIST++/musicfeat_v2 \
  --overwrite
```

默认缺任何 aligned WAV 都以非零状态退出。开发阶段确实接受缺失文件时才使用
`--allow-missing`。

## 4. 构建 official AIST++ 30 FPS artifact

先 dry-run；它检查官方 split、motion、keypoints、camera、music feature、帧数、
bbox 和 camera-space SMPL，但不发布标准 artifact：

```bash
python tools/data/aistpp/build_annot_aist_official_30fps.py \
  --annotations-root /home/weili/datasets/AISTPP_fullset/aist_plusplus_final \
  --musicfeat-dir inputs/AIST++/musicfeat_v2 \
  --output-root inputs/AIST++ \
  --view c01 \
  --strict \
  --dry-run \
  --report-dir outputs/aistpp_official_dryrun
```

若官方 split 中的 sequence 同时出现在上游 `ignore_list.txt`，builder 默认拒绝
发布。只有检查报告并明确接受这些 official ID 后，才在 dry-run 和正式命令中加
`--allow-ignored-official`。

dry-run 通过后正式发布：

```bash
python tools/data/aistpp/build_annot_aist_official_30fps.py \
  --annotations-root /home/weili/datasets/AISTPP_fullset/aist_plusplus_final \
  --musicfeat-dir inputs/AIST++/musicfeat_v2 \
  --output-root inputs/AIST++ \
  --view c01 \
  --strict \
  --overwrite \
  --report-dir outputs/aistpp_official_build_report
```

## 5. Music-only preflight

正式训练前必须运行：

```bash
python tools/data/aistpp/preflight_music_only.py \
  --root inputs/AIST++ \
  --strict
```

报告写入 `outputs/aistpp_music_only_preflight.json`。strict 模式要求 980/20/20，
并在 artifact 缺失、split 重叠、sequence 缺失、motion/music 非法、非 finite，或
music-motion 帧差大于 2 时返回非零 exit code。

只用于小型开发数据时：

```bash
python tools/data/aistpp/preflight_music_only.py \
  --root /path/to/subset \
  --strict \
  --allow-subset \
  --output outputs/aistpp_music_only_subset_preflight.json
```

`--allow-subset` 只放宽 split 数量，不放宽 feature、motion、finite、重叠或时间
对齐检查。

## 6. Smoke train

以下命令执行真实单卡 diffusion training，并保留正式 validation dataset，只将
训练规模和运行步数压小：

```bash
python scripts/train.py \
  exp=gem_smpl_music_only \
  data.limit_each_trainset=16 \
  data.loader_opts.train.batch_size=2 \
  data.loader_opts.train.num_workers=0 \
  data.loader_opts.val.num_workers=0 \
  pl_trainer.devices=1 \
  pl_trainer.max_steps=10 \
  pl_trainer.val_check_interval=100 \
  use_wandb=false
```

Lightning 的 sanity validation 由现有全局配置设为 0；正式 validation 并未删除。
结果独立写到 `outputs/gem_smpl_music_only/version_N/`。

## 7. Full train

正式单卡训练使用 experiment 默认超参数：

```bash
python scripts/train.py \
  exp=gem_smpl_music_only \
  pl_trainer.devices=1 \
  use_wandb=true
```

GPU 数、batch size、workers、steps 和 validation interval 都可继续通过 Hydra CLI
覆盖，没有硬编码进模型。默认 `pretrain_ckpt=null`，可从零训练。

若将 generalist checkpoint 用作 warm start：

```bash
python scripts/train.py \
  exp=gem_smpl_music_only \
  pretrain_ckpt=/path/to/generalist.ckpt \
  pl_trainer.devices=1
```

这是权重 warm start，不恢复 optimizer/global step。loader 会明确打印
missing/unexpected keys；Transformer shape mismatch 不会被静默吞掉。

## 8. Resume

恢复最近一个 `gem_smpl_music_only` version 的 `last.ckpt`，包括 optimizer 和
训练进度：

```bash
python scripts/train.py \
  exp=gem_smpl_music_only \
  resume_mode=last \
  pl_trainer.devices=1 \
  use_wandb=true
```

也可给出 checkpoint 的绝对或相对路径：

```bash
python scripts/train.py \
  exp=gem_smpl_music_only \
  resume_mode=/path/to/version_N/checkpoints/last.ckpt \
  pl_trainer.devices=1
```

## 9. Validation/test generation export

validation 使用固定中心 120 帧，不做随机裁剪，也不读取 `audio_array` 或 MP3：

```bash
python tools/eval/eval_music_only.py \
  --ckpt outputs/gem_smpl_music_only/version_N/checkpoints/last.ckpt \
  --split val \
  --num-samples 20 \
  --seed 42 \
  --output-dir outputs/gem_smpl_music_only/eval_val
```

将 `--split val` 改成 `--split test` 可导出 test。每个样本保存：

- `generated_motion_151d.pt`
- `source_motion_151d.pt`
- `pred_body_params_global.pt`
- `metadata.json`（source sequence、music path、中心帧区间、长度、finite 与统计）

总览保存在 `summary.json`。发现 NaN/Inf 或 shape mismatch 会直接失败。

仓库没有 bundled 的、与论文完全一致的 AIST++ pretrained FIDk/FIDm evaluator；
因此该工具不声称 FIDk、FIDm、Divk、Divm、PFC 或 BAS 是论文等价指标，也不会用
临时实现伪造这些名字。目前实现的是 generation export、source 对照、finite 检查
和 basic statistics。

## 10. Music-only inference

输入必须是已提取的 float、finite `[T,35]` tensor；当前最小 CLI 明确支持
`1 <= T <= 120`：

```bash
python scripts/demo_music_only.py \
  --music-embed outputs/music_embed.pt \
  --ckpt outputs/gem_smpl_music_only/version_N/checkpoints/last.ckpt \
  --output-dir outputs/gem_smpl_music_only/demo \
  --cfg-scale 2.5 \
  --seed 42
```

`--cfg-scale` 直接映射到仓库已有 diffusion guidance 参数，没有实现另一套
sampler。若要启用现有锁脚后处理，可加 `--postproc`。脚本构造静态 camera、零
图像/2D placeholder 以兼容 `GEM.predict` contract，但模型会先断言
`pipeline.args.in_attr == ["encoded_music"]`；这些 placeholder 不进入 `f_cond`。

输出包括 `generated_motion_151d.pt`、`pred_body_params_global.pt` 和
`metadata.json`，并验证生成 motion 与所有 global body parameter 都为 finite。
