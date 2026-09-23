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

服务器1已实际发布并全量验证，计数如下（序列数占比约89.9685%/5.0157%/5.0157%）：

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

2026-09-23 服务器1验收记录：

- 发布位置：`/data0/user/liwei/datasets/bumi_music_umr70_mine_pass_90505_v1`。
- `split_report.json` 保存划分、12套严格校验结果、源/新manifest指纹及旧train进入留出集
  的条数；`stage1_dataset_verification.json` 保存独立逐条核对和Stage1实际batch探针。
- 4765条原动作恰好出现一次；8277个payload文件全部硬链接原字节，独立核验同inode；
  原manifest/meta/stats指纹保持不变。固定seed重算归属一致，已知音频哈希跨split交集为0。
- 四库各train/val/test共12个真实Stage1 Dataset batch均通过现有collate/validator；
  新stats可由原BumiEndecoder严格加载并归一化出有限值。该验收没有运行Actor训练。
- 新train统计量SHA256：`f51df1c167ba953b87953d7efb7a611c92e0e2da7616a8b88b18eb8c79c3df79`。
  原统计量仍为 `a695a1bb09bb9a936869aefb87672e4895bc4f35ab23efb40047bf335fa4d8fc`。
- 重新划分验收时，`build_stage1_loader` 的 Mine 地面准入拒绝已实际复现并保存在当时
  报告中；该结果是历史边界。第 5 步准备已补齐下述完整源序列地面监督并允许 Mine 加载。
  默认验证的 2 个 batch 仍只是接通检查，不是全量评测。

**旧 checkpoint 的边界不因换 split 而消失。** 报告逐库给出新 val/test 来自旧 train 的
条数；从已见过这些样本的旧模型 warm start 后，不能声称它们是整个模型的未见数据。
新 train stats 与旧 checkpoint stats 也不同，现有严格 warm-start 检查保持不变，不静默
绕过。需要严格新留出实验时，使用遵守新划分的训练起点；历史模型仍使用原 stats/config。

Mine 元数据为 `legacy_body_origin_min_zero`，公开库为 `umr_foot_sole_ground_zero_v1`。
重新划分没有修改 Root Z、地面或接触标签。后续第 5 步训练准备通过既有
`meta.ground_supervision` 携带完整源序列的 loss-only 地面监督，解决了这一加载阻塞：
Mine 复用原接触标签器的 FK 足底 2% 分位估计，公开库保持世界地面 Z=0；估计绑定源
文件与运动学身份并缓存，混合损失统一转换到原 canonical Z。原动作、接触 payload、
表示与 Actor 条件字段均不改变，也不使用当前 crop 或网络预测估计地面。
本地 loader 集成与损失一致性测试已通过；真实 GPU、完整模型短训结果由第 5 步独立
验收报告记录，不能以重新划分或本地契约测试替代。本数据重新划分任务本身没有启动训练。
