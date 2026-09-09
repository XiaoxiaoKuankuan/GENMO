# MotionMillion → GENMO SMPL 纯文本动作模型

本文记录 `feature/smpl-text-only` 分支的数据、模型、训练与评测契约。
分支从 `main@966e4a2968c263593d7597b61d60fcf277ec622a` 创建，不包含 BUMI 或
music-only 功能分支的改动。首版只接入 MotionMillion 官方发布数据，把 272D 动作
恢复成 GENMO 151D SMPL 连续表示，并从随机初始化训练 GEM DDIM 扩散器；不复刻官方
FSQ/VQ + LLaMA 模型。

官方资料：

- [项目页](https://vankouf.github.io/MotionMillion/)
- [ICCV 2025 论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Fan_Go_to_Zero_Towards_Zero-shot_Motion_Generation_with_Million-scale_Data_ICCV_2025_paper.pdf)
- [Hugging Face 数据页](https://huggingface.co/datasets/InternRobotics/MotionMillion)
- [官方代码](https://github.com/VankouF/MotionMillion-Codes)
- [官方恢复函数](https://github.com/VankouF/MotionMillion-Codes/blob/main/utils/motion_process.py)

## 1. 授权边界与目录

MotionMillion 是 gated release，数据页声明 CC BY-NC-SA 4.0，并说明部分来源因各自
许可证不能再发布。用户必须亲自在 Hugging Face 接受协议，然后在服务器 1 执行
`hf auth login`，或仅在当前 shell 提供 `HF_TOKEN`。token、密码、私钥不能写入命令
参数、配置、仓库或日志。

固定目录：

```text
/data0/user/liwei/datasets/MotionMillion/
├── raw_hf/
├── genmo_smpl_v1/
├── t5_3b_v1_fp16/
├── official_evaluator/
└── work/
```

仓库只在同名路径不存在时建立软链接；`inputs` 已整体忽略：

```bash
ln -s /data0/user/liwei/datasets/MotionMillion inputs/MotionMillion
```

原始包、转换数据、embedding、checkpoint、评测视频和 HTML 均不进 Git。

## 2. 阶段 A：metadata、split 与空间审计

首次只下载 metadata。`main` 只用来解析一次不可变 revision；成功后从
`download_manifest_metadata.json` 复制 `resolved_revision`，后续均使用完整 SHA：

```bash
source .venv/bin/activate
python tools/data/motionmillion/download_motionmillion.py \
  --stage metadata \
  --revision main \
  --output-root /data0/user/liwei/datasets/MotionMillion/raw_hf

python tools/data/motionmillion/build_motionmillion_genmo.py \
  --metadata-only \
  --raw-root /data0/user/liwei/datasets/MotionMillion/raw_hf \
  --output-root /data0/user/liwei/datasets/MotionMillion/genmo_smpl_v1
```

下载 manifest 记录完整远端文件树的文件数和压缩大小；metadata audit 记录
`version1/t2m_60_300` 的 train/val/test 数量、caption 数、空行、重复文本、每动作
caption 范围、缺文本数和镜像 base ID 泄漏。若官方把原动作及其镜像分到不同 split，
以非镜像原动作的官方 split 为 canonical：保留该 split 中的原动作/变体，把其他 split
变体写入 `mirror_cross_split_exclusions.jsonl` 并排除，不重分配到别的 split。空间门为
至少 1.5 TiB。约 200 万动作、每动作只取一条 caption 的 150-token 全 padding FP16
参考值约为 572 GiB，但不能把它误当成全部 caption 的上限；本 release 隔离后有
15,041,259 条 caption，全部保存 padding 的理论上限约 4.20 TiB。10,000 条 pilot 实测
每条 caption 平均 32.166 个有效 token，紧凑格式按 caption 数线性外推约 923.4 GiB；
正式格式只保存有效 token + offset，全量完成后以实际 shard 总大小为准。

## 3. 阶段 B：MotionGV 10,000 条 pilot

用阶段 A 的完整 `<DATASET_COMMIT_SHA>`，根据文件树填写真实 MotionGV glob：

```bash
python tools/data/motionmillion/download_motionmillion.py \
  --stage full \
  --revision <DATASET_COMMIT_SHA> \
  --motion-pattern 'motion_272rpr/MotionGV/*.tar.gz' \
  --output-root /data0/user/liwei/datasets/MotionMillion/raw_hf

python tools/data/motionmillion/build_motionmillion_genmo.py \
  --raw-root /data0/user/liwei/datasets/MotionMillion/raw_hf \
  --output-root /data0/user/liwei/datasets/MotionMillion/work/genmo_smpl_pilot_10k \
  --archive-pattern 'motion_272rpr/MotionGV/*.tar.gz' \
  --only-split train --limit 10000 \
  --records-per-shard 512 --motion-frames 120 --strict
```

转换器不解压完整数据集。每条数组直接从 tar member 读取，官方布局为：

```text
[0:2]      root local X/Z velocity
[2:8]      root heading delta rotation 6D
[8:74]     22 x local joint position
[74:140]   22 x local joint velocity
[140:272]  22 x local joint rotation 6D
```

heading 按 `R_rel @ R_previous` 累积；root rotation 为
`R_heading^-1 @ R_local_root`；第 `t>0` 帧根速度用前一帧逆 heading 旋回世界坐标后
累加；translation Y 取局部 root 高度。输出固定 30 FPS、Y-up、米制
`pose[F,66]`、`trans[F,3]` 和共享零 `beta[10]`。小于 60、大于 300、shape 错误、
NaN/Inf 或旋转退化都持久化到拒绝表，不静默跳过，也不误记为未发布。
每个 split manifest 和顶层 `dataset_release.json` 还会累加实际接收动作的
`total_frames / duration_seconds / duration_hours`；该时长按原始有效帧和 30 FPS
计算，不把短动作训练时补齐到 120 帧的 padding 计入统计。
MotionGV、MotionLLAMA、MotionUnion 及镜像分卷的 tar member 会省略来源顶层目录；
转换器依据受控归档相对路径补回该命名空间，再要求候选唯一命中官方 split 与文本索引。
该规则不适用于已在 member 中携带完整名称的根目录 PhantomDance 归档，也不会仅按
basename 猜测并接收动作。
pilot 因 `archive_pattern/only_split/limit` 尚未观察到的 ID 写入
`unresolved_by_scope.jsonl`；只有 metadata 已证明缺文本的记录会进入
`unavailable_by_release.jsonl`，不能把 pilot 范围外数据误报为官方缺失。

T5 pilot 使用不可变模型 revision；工具会对快照内所有模型/分词器文件计算 SHA256：
Hugging Face `local-dir` 自动生成的 `.cache` 下载时间戳不属于模型身份并明确排除，避免
同一官方 commit 因重新下载时间不同而产生伪 fingerprint 漂移。

```bash
python tools/data/motionmillion/extract_t5_embeddings.py \
  --motion-root /data0/user/liwei/datasets/MotionMillion/work/genmo_smpl_pilot_10k \
  --output-root /data0/user/liwei/datasets/MotionMillion/work/t5_pilot_10k \
  --model-name-or-path t5-3b \
  --model-revision <T5_COMMIT_SHA> \
  --device cuda:0 --batch-size 16

python tools/data/motionmillion/preflight_motionmillion.py \
  --motion-root /data0/user/liwei/datasets/MotionMillion/work/genmo_smpl_pilot_10k \
  --embedding-root /data0/user/liwei/datasets/MotionMillion/work/t5_pilot_10k \
  --normalized-stats-samples 1000

python tools/data/motionmillion/render_motionmillion_pilot.py \
  --motion-root /data0/user/liwei/datasets/MotionMillion/work/genmo_smpl_pilot_10k \
  --output-root /data0/user/liwei/datasets/MotionMillion/work/pilot_render_32 \
  --device cuda:0
```

全量 T5 建议先用单进程 `--limit-shards 1` 建立三个 split 的共同 contract 和首批
shard，再启动 8 个进程，各自设置 `CUDA_VISIBLE_DEVICES=<rank>`、
`--device cuda:0 --batch-size 64 --resume --worker-rank <rank> --worker-world-size 8`。
worker 只写 `shard/meta` 和独立的 `workers/rank_NNN.json`，不会竞争最终 manifest；
8 个 worker 全部成功后，必须再运行一次不带 worker 参数的单进程 `--resume`，逐 shard
核验 motion/embedding SHA256、caption fingerprint、record 顺序并发布完整 release。
任一 worker 失败时只恢复失败 rank，不能在 worker 未齐时把部分 manifest 用于训练。
每个 worker 会先把一个 motion shard 的全部 caption 按原顺序展平，以真实 batch 64
连续编码，再利用 token offset 拆回逐动作记录；该优化不改变 caption 顺序、紧凑存储
schema 或数值，只避免平均约 21 条 caption/动作导致配置 batch 长期吃不满。

渲染器固定选择 32 条并强制覆盖走跑、舞蹈、武术、地面、镜像和其他动作；类别不足
即失败。preflight 重验 shard SHA/顺序/split，统计根高度、根速度、旋转幅度，并实际
走现有 151D EnDecoder，报告每通道 mean/std、`|z|>8` 比例；任一通道离群比例超过
25% 默认阻断。阈值可以有记录地调整，但不能为放过坐标/单位错误临时换统计契约。

pilot 单卡 forward/backward、DataLoader 吞吐和显存检查属于独立运行授权；代码就绪
不自动执行。执行后需记录环境、峰值显存、samples/s、loss finite 和临时目录清理。

## 4. 阶段 C：全量 release

不设置 `--motion-pattern` 即逐归档恢复下载。每个文件完成后立即校验远端大小、
LFS/Xet SHA256 和 tar 完整性，并显示速度与 ETA。full 阶段同时包含官方 `assets/**`，
使本地文件集合与该 revision 的完整仓库文件树一致：

```bash
python tools/data/motionmillion/download_motionmillion.py \
  --stage full --revision <DATASET_COMMIT_SHA> \
  --output-root /data0/user/liwei/datasets/MotionMillion/raw_hf

python tools/data/motionmillion/build_motionmillion_genmo.py \
  --raw-root /data0/user/liwei/datasets/MotionMillion/raw_hf \
  --output-root /data0/user/liwei/datasets/MotionMillion/genmo_smpl_v1 \
  --records-per-shard 512 --motion-frames 120

python tools/data/motionmillion/extract_t5_embeddings.py \
  --motion-root /data0/user/liwei/datasets/MotionMillion/genmo_smpl_v1 \
  --output-root /data0/user/liwei/datasets/MotionMillion/t5_3b_v1_fp16 \
  --model-name-or-path t5-3b --model-revision <T5_COMMIT_SHA> \
  --device cuda:0 --batch-size 16

python tools/data/motionmillion/preflight_motionmillion.py \
  --motion-root /data0/user/liwei/datasets/MotionMillion/genmo_smpl_v1 \
  --embedding-root /data0/user/liwei/datasets/MotionMillion/t5_3b_v1_fp16 \
  --normalized-stats-samples 1000
```

full 下载恢复会读取 `download_progress_full.json`：只有 repo ID、不可变 revision、
motion pattern、远端文件顺序/大小/blob/LFS 身份、本地大小、既有 SHA256 和 tar 完成
状态全部一致时，才直接复用已验证归档前缀，避免重新读取数百 GB 做重复哈希；任一字段
漂移都会阻断，不会降级成静默重下或混用。

motion 与 embedding shard 一一对齐。训练索引是 mmap 结构化 NumPy，只含
`shard_id/record_index/frames/window_index`；v1 每个 motion 只有一行且
`window_index=0`，连续 120 帧窗口在 Dataset 读取时随机裁取。sampler 先打乱 shard、
再打乱 shard 内样本，最后无重复分给 DDP rank；训练固定 `drop_last=True` 并报告丢弃数。最终
release、三个 split manifest 与 preflight 报告共同构成数据身份链。训练规模只能引用
真实 closed-loop record_count，不能用官方约 200 万宣传数替代。

## 5. 模型与 batch 契约

实验入口是 `exp=gem_smpl_motionmillion_text_only`：

- 训练/验证只有 `motionmillion_text_train` / `motionmillion_text_val`。
- `train_modes=[diffusion]`，`pipeline.args.in_attr=[]`。
- `f_cond/f_uncond/f_empty` 是 `[B,L,1024]` 零张量；文本只走 cross-attention。
- `encode_text=true`、1024D、150 token，训练不加载 T5。
- 只有 denoiser 的 `text_mask_prob=0.1`；pipeline 文本 mask 及其他模态 dropout 为零。
- 其他输入模态 mask 全 false；151D 去噪、3D joint/vertex、world translation、
  static/contact 人体监督保留。
- 继续使用 `v1_smpl_amass_bedlam`、`MM_V1_AMASS_LOCAL_BEDLAM_CAM` 和 151D。
- `pretrain_ckpt=null`、`ckpt_path=null`、`resume_mode=null`；只允许同实验完整恢复。

Dataset 每次均匀选一个 caption，恢复 `[150,1024]` float32 与 `[150]` bool mask。
60–119 帧尾部补到 120 但有效 mask 只覆盖真实帧；120–300 帧训练随机连续裁 120，
验证中心裁。padding mask 传到每个文本 MHA；CFG 无条件分支只清零有效 embedding，
保留相同 mask。全 padding 正式文本会在 Dataset/GEM 边界失败，防止 MHA NaN。

## 6. 阶段 D/E：smoke 与正式训练（需要单独授权）

默认是 64 × 8 × accumulate 1 = 全局 batch 512。若 20-step 探测 OOM，依次改为：

```bash
python scripts/train.py exp=gem_smpl_motionmillion_text_only \
  data.loader_opts.train.batch_size=32 pl_trainer.accumulate_grad_batches=2

python scripts/train.py exp=gem_smpl_motionmillion_text_only \
  data.loader_opts.train.batch_size=16 pl_trainer.accumulate_grad_batches=4
```

不要直接用正式命令“试一下”。启动前需重新核对 `nvidia-smi`、进程归属和用户授权，
并用独立测试输出目录跑 20-step OOM 探测及 100 optimizer step smoke；峰值低于
80 GB/卡，无 OOM、NaN/Inf、DDP hang。smoke 只证明运行健康，不证明动作质量。

正式配置：AdamW 2e-4，5,000 step 线性 warmup，余弦降至 2e-6，300,000 optimizer
step，gradient clip 0.5，每 10 step 日志、5,000 step 验证、10,000 step checkpoint。
300k 对应 `300000 × 512 = 1.536e8` 样本窗口。监控必须记录实际加权 loss、LR、
gradient norm、文本 dropout 实测比、data wait、step time、samples/s、显存、利用率、
最新完整 checkpoint 与稳定窗口 ETA。

代码会记录 `train/gradient_norm_2`、`Metric_diffusion/text_cfg_dropout_metric`、
`train_timer/data_waiting`、`train_timer/single_batch`、
`train_timer/global_samples_per_second`、`train_timer/eta_hours` 与 rank-0 CUDA
当前/峰值显存；W&B system metrics 用于 GPU 利用率。ETA 来自最近 5 个 batch 的稳定
窗口，只是运行时估计，不是预先承诺的工期。

## 7. 推理与评测

新 checkpoint 写入 `genmo_text_contract`，声明 150 token、1024D 和 text-only pipeline；
旧 checkpoint 缺少该字段时按历史 50 token + `gem_smpl` 解析。文本 demo 与 resident
runtime 都按 checkpoint 设置长度。v1 验收固定 120 帧、30 FPS、DDIM-50、CFG 2.5；
300 帧只能标为外推诊断。

自动评测先把生成 SMPL 转回官方 272D：

```bash
python tools/eval/motionmillion_smpl_to_272.py \
  --input <GENERATION_DIR>/motion.npz \
  --output <OFFICIAL_EVAL_INPUT>/motion.npy
```

固定的官方代码基线是
`8a2a7dfa66ecb6a1533d3d9cb49c743a697e1e1c`。该版本官方 evaluator loader 实际过滤
`<60` 或 `>200` 帧；因此训练仍接收 60--300 帧，但自动指标必须按 60--200 帧的原始
val eligibility，不能擅自把 201--300 帧加入 FID。先从只读 tar 流式物化最小 evaluator
视图，再冻结代码/权重/统计量/资格集合身份：

```bash
python tools/eval/prepare_motionmillion_official_eval.py \
  --raw-root /data0/user/liwei/datasets/MotionMillion/raw_hf \
  --motion-root /data0/user/liwei/datasets/MotionMillion/genmo_smpl_v1 \
  --output-root /data0/user/liwei/datasets/MotionMillion/official_evaluator

python tools/eval/fingerprint_motionmillion_evaluator.py \
  --root /data0/user/liwei/datasets/MotionMillion/official_evaluator \
  --code-root /data0/user/liwei/datasets/MotionMillion/official_evaluator/code \
  --checkpoint '/data0/user/liwei/datasets/MotionMillion/official_evaluator/checkpoints/evaluator/epoch=199.ckpt' \
  --eligibility /data0/user/liwei/datasets/MotionMillion/official_evaluator/eligibility.json \
  --output /data0/user/liwei/datasets/MotionMillion/official_evaluator/evaluator_identity.json
```

官方权重仍按官方仓库 `prepare/download_t2m_evaluators_on_motionmillion.sh` 获取；本仓库
不镜像 Google Drive 文件，也不预填未知 checksum。对完整 eligibility 生成 272D 后，
每个 seed 的 prediction JSONL 必须逐行提供 `motion_id/caption/path/length`。单轮命令：

```bash
python tools/eval/generate_motionmillion_val_predictions.py \
  --checkpoint <FIXED_STEP_CHECKPOINT> \
  --eligibility /data0/user/liwei/datasets/MotionMillion/official_evaluator/eligibility.json \
  --t5-model <PINNED_T5_3B_SNAPSHOT_PATH> \
  --t5-release /data0/user/liwei/datasets/MotionMillion/t5_3b_v1_fp16/embedding_release.json \
  --output-root <SEED_REPORT>/generation \
  --seed <FIXED_SEED> --num-frames 120 --ddim-steps 50 --cfg-scale 2.5

python tools/eval/run_motionmillion_official_metrics.py \
  --predictions <SEED_REPORT>/generation/predictions_272.jsonl \
  --eligibility /data0/user/liwei/datasets/MotionMillion/official_evaluator/eligibility.json \
  --evaluator-identity /data0/user/liwei/datasets/MotionMillion/official_evaluator/evaluator_identity.json \
  --checkpoint <FIXED_STEP_CHECKPOINT> \
  --experiment-config <RESOLVED_EXPERIMENT_YAML> \
  --dataset-release /data0/user/liwei/datasets/MotionMillion/genmo_smpl_v1/dataset_release.json \
  --seed <FIXED_SEED> --output <SEED_REPORT>/metrics.json
```

生成器常驻加载 T5/GEM/SMPL FK，按全局 seed 与 motion ID 稳定选择官方 caption 和逐动作
DDIM seed，并逐条原子记录恢复进度；它会实际占用 GPU，所以与 smoke/正式训练一样需要
单独授权，不能由代码同步自动触发。

运行器严格模拟官方 `shuffle=True/batch=32/drop_last=True`，调用官方
`EvaluatorModelWrapper272RPR` 提取 embedding，并计算要求的六类指标。必须运行 20 个
不重复固定 seed。每个 seed 的 JSON 身份字段见
`tools/eval/summarize_motionmillion_metrics.py`，汇总命令：

```bash
python tools/eval/summarize_motionmillion_metrics.py \
  --output <REPORT>/summary.json <REPORT>/seed_*.json
```

汇总器拒绝 checkpoint/config/data/evaluator/DDIM/CFG 漂移，输出 FID、Diversity、
R@1/@2/@3、Matching Score 的均值和 95% CI。候选先最低 FID；FID 近似时再按更高
R@1、最后更低 Matching Score 选择。上游 evaluator 环境仍执行官方 loader，本仓库
不复制或篡改 eligibility。

126 条无 GT prompt 的人工页面：

```bash
python tools/eval/build_motionmillion_review.py \
  --prompt-file /data0/user/liwei/datasets/MotionMillion/official_evaluator/assets/infer_batch_prompt.txt \
  --video-root <VIDEOS_NAMED_000_TO_125> \
  --checkpoint <FIXED_STEP_CHECKPOINT> \
  --output <REPORT>/index.html \
  --seed 20260909 --num-frames 120 --fps 30 --ddim-steps 50 --cfg-scale 2.5
```

页面保存 Text Alignment、Motion Smoothness、Physical Plausibility 三项 1–4 分并导出
绑定 checkpoint SHA256 的 JSON。16 条英文及中文直译只作跨语言诊断，中文不计入 v1
官方质量排序。

## 8. 当前边界

分支代码覆盖下载、metadata audit、流式转换、紧凑 T5、Dataset、sampler、空条件、
文本 mask、150-token checkpoint、配置、预检、272D 适配、官方 evaluator
物化/指纹/单轮指标和人工页面。截至 2026-09-09，用户已完成 gated 许可，服务器 1
已按不可变 revision `007582c4fc9637a3f36e548a67be1ef6eaf881a5` 将官方仓库
77/77 文件、`307655361973` 字节完整下载到
`/data0/user/liwei/datasets/MotionMillion/raw_hf`。60 个动作归档共
`306985351080` 字节；77 个文件均有 SHA256，65 个 LFS/Xet SHA 全部匹配，62 个 tar
共遍历 2932197 个成员，缺失和大小不一致均为 0。metadata 审计已完成，原始官方 split
为 train/val/test=`700903/44044/131819`，镜像跨 split 隔离后 eligibility 为
`624554/21211/70824`，其中 3 条缺文本。

以下后续运行项仍保持“待核验”：

- MotionGV 10,000 条 pilot、32 条视频与 1,000 条真实统计。
- 在线 T5 快照和真实 FP16 cosine parity。
- 单卡 forward/backward、吞吐与八卡 100-step smoke。
- 272D→SMPL 全量转换、T5-3B 全量特征和 300k 训练。
- 官方 val 20-seed 指标和 126 条真实视频评分。

这些运行项完成后，必须记录绝对路径、revision/SHA、真实数量、拒绝原因、命令、环境、
结果和清理状态，才可进入下一道门。
