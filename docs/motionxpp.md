# Motion-X++ 三维动作与文本训练支持

本工具链把 Motion-X++ 的 `smplx322` 三维动作和 `semantic_label` 转成
GENMO 可延迟加载的动作分片与 T5-3B 文本分片。它不解压全部 ZIP，不会修改原始
数据，也不会在内存中一次加载全部动作或 embedding。

## 数据边界

服务器原始数据路径：

```text
/home/liwei/datasets/Motion-Xplusplus
```

仓库内使用软链接：

```bash
cd /home/liwei/GENMO

if [ ! -e inputs/Motion-Xplusplus ]; then
  ln -s /home/liwei/datasets/Motion-Xplusplus inputs/Motion-Xplusplus
fi

readlink -f inputs/Motion-Xplusplus
du -sh inputs/Motion-Xplusplus
```

本地真实归档审计结果为：

| subset | SMPL-X | semantic text | keypoints |
|---|---:|---:|---:|
| `animation` | 559 | 559 | 559 |
| `haa500` | 6,944 | 6,944 | 6,944 |
| `humman` | 971 | 971 | 971 |
| `idea400` | 12,040 | 12,040 | 12,040 |
| `kungfu` | 1,031 | 1,031 | 1,032 |
| `music` | 3,394 | 3,394 | 3,394 |
| `perform` | 922 | 922 | 922 |
| **合计** | **25,861** | **25,861** | **25,862** |

归档内部带有很长的绝对式前缀，工具只用最终相对 stem 配对，所以 ZIP 内部顶层目录
变化不会破坏配对。

官方 Motion-X 代码和 README 明确：

- 发布动作统一为 30 FPS；
- `0:3` 是 root/global orientation；
- `3:66` 是 21 个身体关节轴角；
- `309:312` 是世界平移；
- `312:322` 是 10 维 body shape。

官方预处理将原始 Z-up mocap 用固定 `R_x(-90°)` 变到 Y-up；本地真实 `smplx322`
根平移统计也显示第二轴是身体高度。因此服务器构建命令显式使用：

```text
--source-up-axis y
```

构建器仍不会把未知坐标系静默当成 Y-up。非官方变体必须显式给出 `x`、`y` 或 `z`。

### 当前关键点限制

真实 keypoint JSON 是 COCO-WholeBody 格式，包含 17 个 body keypoint、手、脚、脸
以及 0/1 置信度，但当前归档的 `images` 和顶层对象没有图像宽高，也没有校准的相机
内参/外参。第一版因此只训练：

```text
SMPL-X 3D motion + semantic text
```

正式配置固定 `condition_on_keypoints=false`，`has_2d_mask` 全 False，
`2d_only=false`。代码不会伪造相机 K，也不声称复现论文中的纯 2D-only 训练。

## 1. 完整数据审计

```bash
cd /home/liwei/GENMO
source .venv/bin/activate

python tools/data/motionxpp/inspect_motionxpp.py \
  --root inputs/Motion-Xplusplus \
  --output-dir outputs/motionxpp_inspect \
  --sample-count 3
```

生成：

```text
outputs/motionxpp_inspect/
├── inventory.json
├── pairing_report.json
├── schema_report.json
├── overlap_report.json
└── recommended_subsets.txt
```

当前 7 个归档都能一一配对，并且来源不是当前 AMASS、HumanML3D 或 AIST++ 的明确
重复来源，因此默认推荐：

```text
animation
haa500
humman
idea400
kungfu
music
perform
```

`music` 归档实际是乐器演奏视频（例如 `Play_Flute`、`Play_Guitar`），不是 AIST++
的 `gXX_sXX_cXX_dXX_mXX_chXX` 舞蹈序列。若以后出现名为 `aist`、`humanml`
或 `amass` 的归档，审计会依据明确 provenance 将其标记为与现有训练集重叠。构建器
还会对归一化后的动作内容计算 hash，报告真正的跨 subset 重复，而不是凭模糊名字删数据。

## 2. 8 条 smoke build

先使用一个非重复 subset：

```bash
python tools/data/motionxpp/build_motionxpp_genmo.py \
  --root inputs/Motion-Xplusplus \
  --output-root outputs/motionxpp_smoke/genmo_support \
  --subsets animation \
  --limit 8 \
  --records-per-shard 4 \
  --source-up-axis y \
  --target-fps 30 \
  --strict
```

输出采用：

```text
genmo_support/
├── manifests/{train,val,test}.jsonl
├── shards/{train,val,test}/*.pth
└── reports/
    ├── build_summary.json
    ├── rejected_samples.jsonl
    ├── duplicates.jsonl
    ├── coordinate_audit.json
    └── split_summary.json
```

每个 motion shard 是 `dict[motion_id, record]`，record 中的 `pose` 为 `[F,66]`，
`trans` 为 `[F,3]`，`beta` 为 `[10]` 或 `[F,10]`。旋转用 shortest-path
quaternion SLERP 重采样，平移线性重采样；root orientation 和 translation 使用同一个
固定坐标旋转。手、下颌、表情和脸部 shape 暂未进入 GEM-SMPL 训练目标，忽略范围会
写入 `build_summary.json`。

只审计、不发布主分片：

```bash
python tools/data/motionxpp/build_motionxpp_genmo.py \
  --root inputs/Motion-Xplusplus \
  --output-root outputs/motionxpp_dryrun \
  --subsets-file outputs/motionxpp_inspect/recommended_subsets.txt \
  --source-up-axis y \
  --dry-run
```

## 3. 8 条 T5-3B smoke

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/data/motionxpp/extract_t5_embeddings.py \
  --manifest outputs/motionxpp_smoke/genmo_support/manifests/train.jsonl \
  --output-root outputs/motionxpp_smoke/t5_embeddings_v1_half \
  --batch-size 8 \
  --motions-per-shard 4 \
  --model-name-or-path t5-3b \
  --local-files-only \
  --device cuda:0 \
  --model-dtype float16 \
  --strict
```

如果服务器的 T5-3B 不在 Hugging Face 默认缓存，将 `--model-name-or-path` 换成
服务器本地 T5-3B 目录。工具不会用零 embedding 替代缺失模型。

每个 motion 得到 `[caption数,50,1024]` 的 CPU FP16 tensor；不会生成一个需要整体
载入内存的 `all_text_embed.pth`。

## 4. smoke preflight

```bash
python tools/data/motionxpp/preflight_motionxpp.py \
  --root inputs/Motion-Xplusplus \
  --motion-manifest outputs/motionxpp_smoke/genmo_support/manifests/train.jsonl \
  --embedding-manifest outputs/motionxpp_smoke/t5_embeddings_v1_half/manifests/train.json \
  --sample-records 8 \
  --dataset-samples 8 \
  --report outputs/motionxpp_smoke/preflight_report.json
```

预检会重载动作和 embedding 分片、检查 key/shape/finite/caption 一致性，实例化
`MotionXppDataset`，读取 8 个样本，并用现有 `collate_fn` 组成 batch_size=2。

## 5. 完整转换

确认审计报告后运行：

```bash
python tools/data/motionxpp/build_motionxpp_genmo.py \
  --root inputs/Motion-Xplusplus \
  --output-root inputs/Motion-Xplusplus/genmo_support \
  --subsets-file outputs/motionxpp_inspect/recommended_subsets.txt \
  --records-per-shard 512 \
  --source-up-axis y \
  --target-fps 30 \
  --split-seed 20260724 \
  --strict
```

中断后用相同参数恢复：

```bash
python tools/data/motionxpp/build_motionxpp_genmo.py \
  --root inputs/Motion-Xplusplus \
  --output-root inputs/Motion-Xplusplus/genmo_support \
  --subsets-file outputs/motionxpp_inspect/recommended_subsets.txt \
  --records-per-shard 512 \
  --source-up-axis y \
  --target-fps 30 \
  --split-seed 20260724 \
  --strict \
  --resume
```

resume 只复用 fingerprint、motion ID 和重载验证都一致的分片；配置不同会拒绝混用。

## 6. 完整 T5 提取

依次处理三个 split：

```bash
for split in train val test; do
  CUDA_VISIBLE_DEVICES=0 \
  python tools/data/motionxpp/extract_t5_embeddings.py \
    --manifest inputs/Motion-Xplusplus/genmo_support/manifests/${split}.jsonl \
    --output-root inputs/Motion-Xplusplus/t5_embeddings_v1_half \
    --batch-size 16 \
    --motions-per-shard 256 \
    --model-name-or-path t5-3b \
    --local-files-only \
    --device cuda:0 \
    --model-dtype float16 \
    --strict
done
```

中断恢复只需在相同命令中增加：

```text
--resume
```

## 7. 完整训练前预检

```bash
python tools/data/motionxpp/preflight_motionxpp.py \
  --root inputs/Motion-Xplusplus \
  --motion-manifest inputs/Motion-Xplusplus/genmo_support/manifests/train.jsonl \
  --embedding-manifest inputs/Motion-Xplusplus/t5_embeddings_v1_half/manifests/train.json \
  --sample-records 64 \
  --dataset-samples 8 \
  --report outputs/motionxpp_preflight/report.json
```

## 8. 单卡 20 步训练 smoke

这一步只在数据预检通过后手工启动：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/train.py \
  exp=gem_smpl_motionxpp \
  ckpt_path=inputs/pretrained/gem_smpl.ckpt \
  pl_trainer.devices=1 \
  pl_trainer.max_steps=20 \
  data.loader_opts.train.batch_size=2 \
  data.loader_opts.train.num_workers=0 \
  use_wandb=false
```

这里的 `ckpt_path` 调用仓库现有 `load_pretrained_model()`，只加载官方完整 GEM-SMPL
权重，不把 checkpoint 中的 `global_step` 当作 Lightning optimizer resume。

## 9. 四卡正式微调

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python scripts/train.py \
  exp=gem_smpl_motionxpp \
  ckpt_path=inputs/pretrained/gem_smpl.ckpt \
  pl_trainer.devices=4 \
  data.loader_opts.train.batch_size=4 \
  data.loader_opts.train.num_workers=4
```

配置默认：

- 完整 GEM 网络，不删除 audio/music 参数；
- 不加入 BEAT2；
- 不加入缺失的 `3dpw_occ_v1`；
- AdamW 学习率 `2e-5`；
- `max_steps=20000`；
- 每 epoch 验证；
- 每卡 batch size 4、4 个 worker。

真正恢复 optimizer、scheduler 和 global step 时使用项目现有 `resume_mode=last`，不要
同时把旧的官方 checkpoint 当成 resume checkpoint。

## 与 Motion-X V1 的关系

当前输入是 2025 年重新整理的 Motion-X++ Hugging Face 版本，不等同于论文最早发布的
Motion-X V1。这里实现的是可审计的 Motion-X++ `motion_generation/smplx322` +
`semantic_label` 支持；没有声称严格复现 Motion-X V1 的全部 subset、face/hand
目标或论文纯 2D-only 训练。

