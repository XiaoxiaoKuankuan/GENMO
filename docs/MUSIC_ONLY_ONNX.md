# AIST++ music-only checkpoint：音乐验证与 ONNX

## 本次模型

服务器训练产物：

```text
/data0/user/liwei/GENMO_outputs/gem_smpl_music_only/version_1/checkpoints/last.ckpt
```

本地副本：

```text
inputs/checkpoints/music_only_aistpp/version_1/last.ckpt
```

该 checkpoint 的 `global_step=260000`，实验是
`exp=gem_smpl_music_only`，网络条件列表必须严格为：

```python
["encoded_music"]
```

输入音乐不是 raw WAV，而是仓库现有
`extract_edge_baseline35()` 从 WAV 提取的 `[T,35] @ 30 Hz` 特征。音乐验证脚本会在
内部完成这一步，不会使用 OMG 特征。

## 1. 用真实音乐验证 PyTorch checkpoint

以下命令选取 WAV 的前 4 秒，从 STFT 边界产生的 121 帧中明确选择前 120 帧
EDGE35，运行 50-step DDIM/CFG，检查
151D motion、SMPL body 参数和接触置信度均存在且 finite：

```bash
cd /home/weili/GENMO

.venv/bin/python tools/eval/validate_music_only_checkpoint.py \
  --audio /home/weili/datasets/AISTPP_official/music/wav/mHO1.wav \
  --audio-start-sec 0 \
  --audio-duration-sec 4 \
  --num-frames 120 \
  --ckpt inputs/checkpoints/music_only_aistpp/version_1/last.ckpt \
  --cfg-scale 2.5 \
  --ddim-steps 50 \
  --seed 42 \
  --output-dir outputs/validation/music_only_aistpp_mHO1
```

输出包括：

```text
music_features.pt
generated_motion_151d.pt
pred_body_params_global.pt
validation_report.json
```

如果要验证已提取的 EDGE35，改用：

```bash
.venv/bin/python tools/eval/validate_music_only_checkpoint.py \
  --music-embed inputs/AIST++/musicfeat_v2/gHO_sBM_cAll_d20_mHO4_ch09_musicfeat_fps30.pt \
  --feature-start-frame 0 \
  --num-frames 120 \
  --ckpt inputs/checkpoints/music_only_aistpp/version_1/last.ckpt \
  --output-dir outputs/validation/music_only_aistpp_feature
```

`--postproc` 是可选的推理后处理；不传时验证网络原始生成，传入时再启用接触置信度
驱动的根轨迹修正和 IK 锁脚。

## 2. ONNX 的准确边界

导出的 ONNX 是“一个 CFG-guided diffusion denoising step”，包含：

```text
music [B,120,35]
  -> music_embedder
  -> conditional / unconditional music branches
  -> CFG
noisy_motion [B,120,151] + timestep
  -> 16-layer diffusion Transformer
  -> pred_motion [B,120,151]
     pred_camera [B,120,3]
     static_conf_logits [B,120,6]
```

它不把 50 次 DDIM 循环展开成一个超大静态图，也不包含 Python 版 EnDecoder、SMPL
forward 或 IK 后处理。这样同一份 Transformer 权重只保存一次，ONNX Runtime 每个
DDIM step 重复调用它。完整 ONNX 验证仍复用仓库已有 DDIM scheduler，未重新发明扩散
公式。

ONNX 输入为：

| 名称 | dtype | shape | 含义 |
|---|---|---|---|
| `noisy_motion` | float32 | `[B,120,151]` | 当前扩散状态 `x_t` |
| `diffusion_timestep` | int64 | `[B]` | 原始 0..999 timestep |
| `music` | float32 | `[B,120,35]` | EDGE baseline35 |
| `length` | int64 | `[B]` | 有效帧数 |
| `guidance_scale` | float32 | `[1]` | CFG scale |

序列长度固定为 120，batch 维动态。这与该模型的 120 帧训练窗口一致。

## 3. 导出 ONNX

环境需要 `onnx`；运行验证还需要 `onnxruntime-gpu`（或 CPU 版
`onnxruntime`）：

```bash
.venv/bin/pip install onnx
```

导出命令：

```bash
.venv/bin/python tools/export/export_music_only_onnx.py \
  --ckpt inputs/checkpoints/music_only_aistpp/version_1/last.ckpt \
  --exp gem_smpl_music_only \
  --seq-len 120 \
  --opset 18 \
  --device cuda \
  --output outputs/onnx/music_only_aistpp_s260000/music_only_denoiser.onnx
```

模型较大时，ONNX 可能生成主 `.onnx` 文件和外部权重 `.onnx.data`，两者必须一起
复制。旁边的 `music_only_denoiser.onnx.json` 保存图合约、checkpoint step、文件大小
和 SHA256。

## 4. ONNX Runtime 数值验证

先做真实音乐的单步严格对齐：

```bash
.venv/bin/python tools/eval/validate_music_only_onnx.py \
  --audio /home/weili/datasets/AISTPP_official/music/wav/mHO1.wav \
  --audio-duration-sec 4 \
  --seq-len 120 \
  --ckpt inputs/checkpoints/music_only_aistpp/version_1/last.ckpt \
  --onnx outputs/onnx/music_only_aistpp_s260000/music_only_denoiser.onnx \
  --provider cuda \
  --output-dir outputs/onnx/music_only_aistpp_s260000/validation_single_step
```

再用同一份初始噪声分别跑 PyTorch 和 ONNX Runtime 的完整 50-step DDIM：

```bash
.venv/bin/python tools/eval/validate_music_only_onnx.py \
  --audio /home/weili/datasets/AISTPP_official/music/wav/mHO1.wav \
  --audio-duration-sec 4 \
  --seq-len 120 \
  --ckpt inputs/checkpoints/music_only_aistpp/version_1/last.ckpt \
  --onnx outputs/onnx/music_only_aistpp_s260000/music_only_denoiser.onnx \
  --provider cuda \
  --full-ddim-steps 50 \
  --output-dir outputs/onnx/music_only_aistpp_s260000/validation_full_ddim
```

报告会给出每个输出的 shape、finite、最大绝对误差、平均绝对误差、RMSE 和
`allclose` 结果。完整 DDIM 结果另外保存为 `onnx_generated_motion_151d.pt`。

## 5. 本机实际验收结果

2026-08-14 使用 `mHO1.wav` 前 4 秒、seed 42、CFG 2.5、50-step DDIM 实测：

- checkpoint 本地/远端 SHA256 均为
  `4ffd053b6bd53405e56198ce3c89762a0b972b6ee4eb5356544f1c10f02ca5fd`；
- checkpoint 大小 2,570,404,846 bytes，`global_step=260000`；
- 模型参数量 217,580,704，条件列表为 `["encoded_music"]`，没有文本权重；
- PyTorch 生成 `[1,120,151]`，SMPL body 参数全部 finite；
- ONNX 大小 874,074,147 bytes，SHA256 为
  `1dcf2afeb30a488522c38946a6ba6fbdc6eccfab6d43c8a98a9b039fdfd69356`；
- ONNX 单步 151D 最大绝对误差 0.002948、平均绝对误差 0.000181；
- 完整 50-step DDIM 最大绝对误差 0.009118、平均绝对误差 0.000251，finite 且通过；
- 完整报告位于
  `outputs/onnx/music_only_aistpp_s260000/validation_full_ddim/validation_report.json`。
