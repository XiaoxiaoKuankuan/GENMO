# Stage1 四库 PASS 动作 90/5/5 划分

用户于 2026-09-23 指定将筛选后的动作重新按 train/val/test = 90%/5%/5% 划分。
该版本合并旧三套清单中的全部 PASS 动作，在动作序列级别重新归属，然后才由已有
Dataset 生成 120 帧窗口。训练来源抽样仍为 AIST++/AIOZ/FineDance/Mine = 20/35/25/20，
与这里的数据留出比例独立。

## 分组和整数规则

- 先用并查集合并相同源动作/音乐特征/音频 SHA256、同库 sequence/music_group/audio_key、
  规范化歌曲标题；AIOZ 再按 11 位原视频 ID 合并不同片段和舞者。
- Mine 的 `-complete`、`-nutuan数字` 和末尾编号保守归为同曲家族。此规则不宣称已经
  完成声学去重，未知别名、不同录音版本仍可能存在，报告明确保留该证据边界。
- 跨库相连的组整体保留在 train，具体组 ID、来源和条数列在 `split_report.json`。
- 每库 val/test 条数目标分别为 `round_half_up(0.05*N)`，train 使用余数；固定 seed=42，
  按固定顺序子集和选择完整组。无法精确满足条数时选最近可行值，绝不拆组、重复或删样本。
- 同条动作在新清单中恰好出现一次。逐条增加 `resplit_provenance` 记录原 split、组 ID
  和版本，其余生产者字段保持原值。

实际服务器元数据预演得到以下可达计数，正式发布后以报告指纹为准：

| 来源 | train | val | test | 总条数 |
|---|---:|---:|---:|---:|
| AIST++ | 351 | 20 | 20 | 391 |
| AIOZ-GDANCE | 3745 | 208 | 208 | 4161 |
| FineDance | 102 | 6 | 6 | 114 |
| Mine | 89 | 5 | 5 | 99 |
| 合计 | 4287 | 239 | 239 | 4765 |

## 发布、校验和恢复边界

维护入口为 `tools/data/bumi/resplit_bumi_music_dataset.py`。`--plan-only` 只读清单；实际
发布先写同盘独立 staging，payload 硬链接优先，跨设备才校验复制。清单与 meta 是独立
文件，原 manifest/meta/stats 归档到 `provenance/original_release`。输出目录已存在时拒绝
覆盖。失败只清理本次精确 staging，所有 validator 和统计步骤成功后才原子重命名。

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python tools/data/bumi/resplit_bumi_music_dataset.py \
  --source-root /data0/user/liwei/datasets/bumi_music_umr70_mine_pass_v1 \
  --output-root /data0/user/liwei/datasets/bumi_music_umr70_mine_pass_90505_v1 \
  --seed 42
```

复用 `validate_bumi_music_dataset.py` 对四库三套 split 全量检查真实 payload、帧数、
音乐对齐、质量/资产/接触契约和来源哈希。复用 `compute_bumi_30d_stats.py` 只遍历新 train，
生成新目录下的 `stats/qpos30_train_stats.json`，其四库 train manifest SHA 必须匹配。
proprio48 继续使用 Stage1 明确物理尺度，不自动拟合统计量。

**旧 checkpoint 的边界不因换 split 而消失。** 报告逐库给出新 val/test 来自旧 train 的
条数；从已见过这些样本的旧模型 warm start 后，不能声称它们是整个模型的未见数据。
新 train stats 与旧 checkpoint stats 也不同，现有严格 warm-start 检查保持不变，不静默
绕过。需要严格新留出实验时，使用遵守新划分的训练起点；历史模型仍使用原 stats/config。

Mine 元数据为 `legacy_body_origin_min_zero`，公开库为 `umr_foot_sole_ground_zero_v1`。
本任务不修改 Root Z、地面或接触标签，现有 Stage1 训练器拒绝 Mine 的旧地面语义这一
边界仍存在。重新划分并通过数据契约不等于已通过四库 Stage1 训练准入；本任务不启动训练。
