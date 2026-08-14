# Music-only 四数据集人工动作筛选

本文说明如何把 AIST++、AIOZ-GDANCE、FineDance、CoMPAS3D 当前实际用于 GENMO
训练的 body motion 导出给外部人员筛选，并在结果回来后安全恢复动作与 EDGE baseline35
音乐特征的对应关系。

## 数据范围与筛选单位

完整导出覆盖 train、val、test，动作保持原始完整长度，不切成 120 帧训练窗口：

| 数据集 | 动作样本 | person-motion 小时 | EDGE35 文件 |
|---|---:|---:|---:|
| AIST++ | 1,020 | 3.984 | 1,020 |
| AIOZ-GDANCE | 6,011 | 60.766 | 1,624 |
| FineDance | 183 | 6.951 | 183 |
| CoMPAS3D | 72 | 3.016 | 36 |
| 合计 | 7,286 | 74.717 | 2,863 |

AIOZ 的每个 dancer、CoMPAS3D 的 leader/follower 都是独立筛选单位。它们可以共享音乐，
所以不能看到一个坏动作就直接删除音乐。音乐只有在 train/val/test 中已经没有任何保留动作
引用时才成为零引用文件。

## 审阅包格式

审阅包只有动作 NPZ、索引和 CSV，不含音乐、视频、SMPL-X 模型权重或原始数据。NPZ 是
30 FPS、完整长度、neutral SMPL-X body-only 参数：

```text
pose       float32 [T,66]，axis-angle
transl     float32 [T,3]，米
betas      float32 [T,10]
fps        30
num_frames T
review_id  dataset__sample_id
```

`pose[:, :3]` 是 global orientation，`pose[:, 3:66]` 是 21 个 body joint。手、脸、眼睛
和表情不在 GENMO 当前 151D motion contract 中，因此不会伪造或导出。

AIOZ、FineDance、CoMPAS3D 当前转换产物已经是 Y-up。AIST++ 源世界是 Z-up，导出工具只对
审阅副本做 Y-up 刚体变换，并通过 SMPL-X forward 检查顶点等价；服务器上的 AIST++ 源数据
不会被修改。

## 导出与交付

在服务器仓库执行：

```bash
cd /home/user/liwei/GENMO

.venv/bin/python tools/data/music_dance/curation/export_motion_review.py \
  --output-root /data0/user/liwei/datasets/music_dance_review/music_only_4set_v1 \
  --splits train val test \
  --review-coordinate y_up

.venv/bin/python tools/data/music_dance/curation/validate_review_package.py \
  --export-root /data0/user/liwei/datasets/music_dance_review/music_only_4set_v1 \
  --expect-full-four-set
```

交付给筛选人员的内容是 `motions/`、`review/decisions.csv`、`index/` 和 `README.md`。
服务器必须保留一份未经编辑的 `index/master.jsonl` 和 `source_fingerprints.json`。

筛选人员只能编辑 CSV 的以下四列：

```text
decision       keep / reject / unsure
issue_codes    多个问题用分号分隔
reviewer
notes
```

reject 必须填写至少一个受支持的问题代码。`unsure` 用于二次复核，不能直接应用到正式数据。
不能通过“对方没有传回某个 NPZ”推断 reject，因为这无法区分人工删除与传输丢失。

## 校验和应用结果

假设对方返回 `/path/to/decisions.csv`：

```bash
.venv/bin/python tools/data/music_dance/curation/validate_review_results.py \
  --export-root /data0/user/liwei/datasets/music_dance_review/music_only_4set_v1 \
  --decisions /path/to/decisions.csv \
  --strict
```

严格模式会拒绝未知 ID、重复 ID、漏标、被修改的 dataset/sample_id/duration、非法问题代码、
空白和 unsure。先做不写文件的预演：

```bash
.venv/bin/python tools/data/music_dance/curation/apply_review_results.py \
  --export-root /data0/user/liwei/datasets/music_dance_review/music_only_4set_v1 \
  --decisions /path/to/decisions.csv \
  --output-root /data0/user/liwei/datasets/music_dance_curated/music_only_4set_v1 \
  --dry-run
```

确认数量、时长和零引用音乐列表后再生成独立 curated 根目录：

```bash
.venv/bin/python tools/data/music_dance/curation/apply_review_results.py \
  --export-root /data0/user/liwei/datasets/music_dance_review/music_only_4set_v1 \
  --decisions /path/to/decisions.csv \
  --output-root /data0/user/liwei/datasets/music_dance_curated/music_only_4set_v1 \
  --apply \
  --quarantine-zero-ref-music

.venv/bin/python tools/data/music_dance/curation/validate_curated_datasets.py \
  --root /data0/user/liwei/datasets/music_dance_curated/music_only_4set_v1 \
  --strict \
  --loader-smoke
```

工具先从 split/manifest 移除 reject 动作，再计算全局剩余音乐引用。curated 数据只包含保留
动作所需的 musicfeat_v2；零引用 EDGE35 以硬链接或恢复副本放入 `quarantine/`。源转换目录、
原始 WAV 和正在运行的训练均不修改。跨文件系统时无法建立硬链接的文件会自动复制。

## 使用筛选后的训练集

筛选结果验证通过后使用独立实验，不覆盖原四数据集实验：

```bash
NCCL_CUMEM_HOST_ENABLE=0 \
NCCL_IB_DISABLE=1 \
NCCL_SOCKET_IFNAME=lo \
TORCH_NCCL_BLOCKING_WAIT=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/python -u scripts/train.py \
  exp=gem_smpl_music_only_4set_curated \
  pl_trainer.devices=8
```

该配置仍然满足：

```python
pipeline.args.in_attr == ["encoded_music"]
pipeline.args.train_modes == ["diffusion"]
```

