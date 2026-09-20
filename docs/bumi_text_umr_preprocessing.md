# MotionMillion / HumanML3D UMR → BUMI 文本动作预处理

适用分支：`feature/bumi-text-only`。入口复用 `tools/data/bumi/prepare_bumi_text.py`。
质量计算复用音乐分支的 `filter_sonic_npz_motions.evaluate_motion`，原音乐代码和阈值配置不改动。

## HumanML3D 交付适配与完整训练数据

HumanML3D 与 MotionMillion 的 UMR 机器人输出都按 Z-up 处理。配置的
`source_contracts` 仅区分重定向前的人体来源，不对机器人 qpos 再旋转或贴地。
以下默认目录针对服务器2的23,242条HumanML3D训练派生交付：

| 用途 | 路径 |
|---|---|
| 原始机器人交付 | `/data2/user/liwei/hml3d_umr` |
| 补齐的原始人体包 | `/data0/user/liwei/datasets/humanml3d_umr_source_v1` |
| 筛选报告 | `/data0/user/liwei/dataset_reports/humanml3d_umr_bumi3_v1` |
| 最终完整动作训练分片 | `/data0/user/liwei/datasets/bumi_text_humanml3d_umr_pass_v1` |
| 配套T5特征 | `/data0/user/liwei/datasets/bumi_text_humanml3d_umr_pass_v1_t5` |

原人体包须与交付的 `source_metadata` 三个文件逐字节一致。适配器核对交付
`SHA256SUMS`、源manifest SHA、机器人/人体的全部数值及完整时间线；缺少原人体
文件会明确INVALID，不降级为只检查元数据。旧输出目录、旧XML路径通过参数显式
映射；使用前必须核验生成端和当前端XML、网格与重定向配置身份相同。

```bash
cd /home/user/liwei/GENMO-bumi-text
# 仅全量筛选，CPU执行；中断时追加 --resume。
BUMI_DATASET=humanml3d bash scripts/prepare_bumi_text_umr.sh

# 首次完整流水线：筛选 -> 原caption绑定 -> GPU T5编码 -> PASS分片 -> train统计 -> 全量加载核验。
BUMI_DATASET=humanml3d BUMI_BUILD_TRAINING=1 BUMI_WORKERS=32 \
  bash scripts/prepare_bumi_text_umr.sh
```

上述两种首次命令选择其一。若已完成筛选而未构建，可在完整流水线命令后追加
`--resume` 复用筛选结果；已经存在的conversion、T5目录或release不会覆盖。
后半程失败时根据已完成阶段单独使用 `humanml-conversion`、`encode_text_features.py`、
`build`、`stats`、`preflight` 继续，不把已存在release当作全部完成。
`BUMI_ENCODE_PYTHON` 默认使用服务器2的GENMO-cu128环境，`BUMI_TEXT_DEVICE`
默认cuda:0，`BUMI_T5_MODEL`默认本地`/data0/user/liwei/models/t5-3b_bed96aab`。

完整保留原训练身份和caption；`M`前缀镜像保留为独立训练样本，同时与原动作共享
防泄漏分组。`__seg_起点毫秒_终点毫秒`记录真实完整区间；原序列不足标注终点时，
仅在母序列帧数能够证明边界截取的情况下接受，并同时保存原标注区间。没有重新
裁剪动作或随机生成val/test。最终只有train记录，val/test清单为空。
训练分片内保存qpos、接触及文本引用；T5分片也必须随训练数据保留。统计量位于
release的`train_stats.json`，全量读取结果位于报告目录`training_preflight.json`。
该脚本准备数据，不启动模型训练。

2026-09-20服务器2全量验收：23,242条全部完成，PASS8,048、REVIEW4,976、REJECT10,218，
INVALID/ERROR均为0。PASS中419条不满足60–300帧，最终train7,629条、1,570,581帧、
14.5424小时、20,608条caption，保留32.8242%，val/test为0。训练时应关闭验证或另配
真实验证集，不能把该训练派生包的动作随机冒充官方验证数据。
最终15个动作分片及资产/统计合计209,464,943字节，60个T5分片及清单6,357,787,606字节，
两者合计6,567,252,549字节。全部7,629条通过现有loader逐caption核验，crop_count=0；
train统计量为v4、dataset=humanml3d，现有BumiEndecoder读取与实际样本编码有限值通过。
报告目录中的`delivery_summary.json`记录完整数量、大小、来源构成、运行身份和关键文件SHA；
`quality_summary.json`是质量汇总，`reports/out_umr.jsonl`是逐条判定，
`training_preflight.json`是最终全量训练加载核验。

下文默认路径和folder分片约定针对MotionMillion；通用质量规则与报告机制共用。

## 数据与判定契约

- 输入是 `folder*/bumi3/batch_summary.json` 及其登记的原生UMR NPZ，同时对照实际文件集合，记录缺失和未登记文件。
- 每条检查完整NPZ解压/CRC、必需字段、float32 qpos[T,28]、21个唯一关节名称、单位wxyz、30Hz、从0连续的frame_ids。
- 核对summary与原人体路径、帧数、FPS、数值及 `source_format=motionmillion_272/output_up=y`。UMR已将机器人输出转为Z-up，不再次旋转。
- 核对XML、运动学JSON与资产清单中的所有网格SHA；关节限位取XML和UMR生成配置的交集，包括双膝0.1rad下界。
- 复用速度、加速度、jerk、根位移/旋转速度检查。新增完整脚网格穿地、候选支撑脚滑、长时间双脚悬空、非相邻刚体凸包碰撞。
- 支撑候选依据脚底高度、垂向速度、持续时间，不使用水平速度提前排除滑动帧。
- 低姿态存在非足部刚体低高度支撑迹象时，双脚离地只保留诊断，避免把躺/跪/手撑动作误当全身悬空；该迹象不是精确接触测量。
- 凸包碰撞逐刚体对报告原始深度及XML默认姿态基线，以额外穿透判定；默认已有的踝部结构重叠不会直接淘汰全部动作。
- `posture_policy: diagnostic` 默认保留坐、跪、躺等姿态的诊断，不执行音乐任务的站立风格门禁。穿地、滑移、碰撞等独立质量规则仍生效。
- 状态分为 `PASS / REVIEW / REJECT / INVALID / ERROR`。INVALID是输入合同失败，ERROR是执行失败；存在ERROR时程序退出码为2，禁止构建正式数据。
- 质量状态和训练适用性分开。仅PASS且完整60–300帧进入训练候选；15帧、301帧等不通过长度条件，不因此伪称数值损坏。
- 不平移Root、不归一化坏四元数、不截取局部动作、不循环补长。异常区间只用于定位，不能沿用整段caption作为局部标注。
- `source_file` 用于恢复原MotionMillion身份；原272D文件未在本阶段再次读取。文本、官方split和T5绑定在后续build时核验，质量候选本身不是最终训练release。

配置：`configs/bumi/quality_filter_umr_text_30hz_v1.yaml`。新增脚部/碰撞阈值是首版离线门禁，需要结合真实报告和回放校准；PASS不代表动力学、控制器或实机验收。

## 服务器2启动

代码目录：`/home/user/liwei/GENMO-bumi-text`。

启动器默认使用 `/home/user/miniconda3/envs/ykj_umr/bin/python`，只用CPU，16个worker，每worker单线程。
输入为 `/data2/user/motion_dataset/millionmotion/umr_change/all`，人体源为同级 `pre_change/all`。
默认校验清单条数必须为559924，报告放到：

```text
/data0/user/liwei/dataset_reports/motionmillion_umr_bumi3_v1/
```

首次执行：

```bash
cd /home/user/liwei/GENMO-bumi-text
bash scripts/prepare_bumi_text_umr.sh
```

中断后使用完全相同路径和配置续跑：

```bash
bash scripts/prepare_bumi_text_umr.sh --resume
```

worker数可以改变，例如追加 `--workers 8`。任务身份绑定代码、配置、资产、库版本、输入清单及选取范围；改变这些条件需新的 `--output`。续跑重新计算每条动作和源人体SHA，仅复用未改变的结果，ERROR自动重试。

长任务可在tmux内执行并记录控制台日志：

```bash
mkdir -p /data0/user/liwei/logs/bumi_text_umr
tmux new -s bumi-text-umr-filter
cd /home/user/liwei/GENMO-bumi-text
set -o pipefail
bash scripts/prepare_bumi_text_umr.sh --resume 2>&1 | tee -a /data0/user/liwei/logs/bumi_text_umr/filter.log
```

按 `Ctrl+b`、再按 `d` 脱离；重新进入使用 `tmux attach -t bumi-text-umr-filter`。
启动器支持 `BUMI_PYTHON / BUMI_UMR_ROOT / BUMI_INPUT_ROOT / BUMI_SOURCE_ROOT / BUMI_REPORT_ROOT / BUMI_WORKERS / BUMI_EXPECTED_RECORDS` 环境变量。数据与报告目录必须分离。

## 输出和恢复机制

| 文件 | 内容 |
|---|---|
| `run.json` | 输入路径、配置/资产/代码/库指纹、有效关节限位、完成状态和partial标志 |
| `quality.sqlite` | 有事务保护的输入索引、逐条指标/原因/异常帧区间；每32条提交一次 |
| `reports/folderN.jsonl` | 按原目录划分的完整逐条报告 |
| `train_candidates.jsonl` | 完整60–300帧PASS候选、原MotionMillion ID、镜像归一化ID和双输入SHA |
| `quality_summary.json` | 状态和原因计数、各目录/来源统计、帧数/小时数、字节数及保留比例 |
| `.lock` | 防止两个写进程同时接管同一报告目录的进程锁 |

并行队列最多保留2×worker任务；一次只读取一个上游summary和有限任务，不把56万条指标装入内存。JSONL与汇总在任务结束后原子发布；运行中以控制台和SQLite为进度依据。若任务中断，以 `run.json` 的状态为准，不把遗留旧汇总当成本轮完成结果。

`--limit N` 或 `--folders folder0` 产生显式partial报告，不能用于正式build。验证必须指定独立临时输出并在同任务清理，禁止向正式目录写测试结果。使用folder子集时还需把 `--expected-records` 改为该子集的实际数量。

阈值修改目前需要新目录重新执行计算，不能把旧指标报告直接当作新阈值筛选结果。报告中的采样候选数量是质量/长度适用性统计，最终训练采样比例仍由文本配对、split、去重结果与训练sampler决定。

## 与已有文本构建器衔接

准备已有契约 `genmo.bumi_text_conversion.v1` 的conversion清单后，可直接引用原生UMR文件作为 `qpos_path`，无需另存56万个标准化NPZ：

```bash
python tools/data/bumi/prepare_bumi_text.py build \
  --source /path/to/verified_conversion.json \
  --quality-report /data0/user/liwei/dataset_reports/motionmillion_umr_bumi3_v1 \
  --output /data0/user/liwei/datasets/bumi_text_motionmillion_umr_pass_v1
```

此命令中的conversion文件需要实际提供，筛选器不会推测caption或随机分配split。每条清单的 `provenance.source_id` 和 `text_source_motion_id`（未提供时取 `motion_id`）必须与报告恢复的原MotionMillion ID一致；caption/embedding SHA、完整帧数和原split仍由现有构建器检查。

构建器读取质量报告后再次验证源动作/人体SHA，只接收PASS，按具名顺序重排qpos、派生接触标签，在来源分组之前绑定镜像归一化ID。完整动作分片、manifest成功后才由隔离staging原子发布。未指定 `--quality-report` 时仍接受原标准 `joint_names` NPZ路径，兼容既有HumanML3D与文本构建流程。

## 验证范围

`tests/bumi/test_umr_text_preprocess.py` 用真实XML和网格、合成源数据测试数值异常、关节顺序、时间线、源SHA变化、有效限位、脚滑/悬空、凸包默认重叠、并行与续跑、长度策略、PASS构建与失败清理。真实数据验证的条数和结果记录在根目录 `记录文本.md`；不得将小样本比例写成559924条的全量质量结果。

## 已有筛选结果分析与高低质量视频

复用原`tools/eval/render_bumi_motion.py`，质量复核模式按报告原判定选样并渲染，不重跑
全量筛选、不改写历史报告。分析模块`tools/eval/bumi_quality_review.py`核对汇总SHA、
完整JSONL统计、候选清单SHA，按动作归并原因，并区分质量状态和长度资格。

```bash
cd /home/user/liwei/GENMO-bumi-text
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONDONTWRITEBYTECODE=1 \
  /home/user/miniconda3/envs/ykj_umr/bin/python -B -u tools/eval/render_bumi_motion.py \
  --quality-report /data0/user/liwei/dataset_reports/motionmillion_umr_bumi3_v1 \
  --output-dir /data0/user/liwei/dataset_reports/motionmillion_umr_bumi3_v1/visual_review \
  --per-group 30
```

输出两段H.264/30FPS双视角视频、中文`分析报告.md`、含每条动作完整指标和视频时间
索引的`analysis.json`，以及原run/summary/按SHA从Git恢复的筛选配置。高组按活动类型
和目录均衡选PASS，低组按脚滑、自碰撞、穿地、突变选REJECT；都是目的性复核样本，
不能拿60条的分布估计全库比例。每条完整播放，原qpos/帧率不修改；右视角半透明地面
便于观察穿地，红框对应原报告当前异常帧。原始动作和人体SHA、真实XML/全部网格须
匹配报告；渲染结束核对解码帧数后原子发布。新结果须用独立目录，验收后替换旧合集。

### 2026-09-20 MotionMillion 实际复核结果

服务器2报告`motionmillion_umr_bumi3_v1`全量559,924条逐项汇总校验通过：PASS
291,348（52.03%）、REVIEW 175,478（31.34%）、REJECT 93,098（16.63%），
INVALID/ERROR为0。PASS中105,559条不足60帧、29,978条超过300帧，完整训练候选
155,811条（27.83%）/181.32小时。原机器人NPZ为9.604 GB，候选对应原NPZ为
2.441 GB；尚不能将候选大小称为包含文本特征的训练release大小。

按真正触发REJECT的原因、逐动作去重：脚滑70,459、自碰撞21,400、脚穿地2,662、
根高度越界1,160、速度/旋转/关节突变96。脚滑与自碰撞合并去重90,067条，占REJECT
96.74%；REVIEW中138,890条有脚滑信号，占79.15%。这使支撑脚滑移和碰撞成为优先
复核方向；长短不适用应继续与动作质量问题分开。目录差异不能直接解释为语义来源差异：
folder7的PASS率65.64%、候选率37.00%，folder9分别28.49%、6.06%。另读两目录
原报告确认folder9的15,952条PASS中12,267条不足60帧，短序列也是候选率低的主要因素。

该批全部原始来源标识为`Mirror_MotionGV`，结论仅覆盖当前转换集合。旧辅助
`source_sequence_key`字段被变量覆盖为`trans_orig`的问题已修正未来输出，原报告
不改写；本次60条视频另以原机器人/人体SHA、NPZ身份与source_file核对来源。

正式复核保存在服务器2：
`/data0/user/liwei/dataset_reports/motionmillion_umr_bumi3_v1/visual_review`。
本地验收结果：`/home/weili/GENMO-bumi-text/outputs/motionmillion_umr_quality_review`。

| 文件 | 完整动作数 | 帧数 | 时长 | 字节数 |
|---|---:|---:|---:|---:|
| high_quality_30.mp4 | 30 | 5,638 | 187.933秒 | 11,574,780 |
| low_quality_30.mp4 | 30 | 4,589 | 152.967秒 | 20,626,277 |

高组移动/转向/低姿态/较活跃肢体/一般肢体各6条，10个目录各3条；低组脚滑10、
碰撞10、穿地6、突变4条，覆盖10个目录。低姿态是当前文本筛选的诊断类别，不自动
淘汰，故高组13–18号含地面动作。两组用于复核当前规则，不是全库随机抽样或全局排名。
H.264/1280×720/30FPS，两端完整解码与原始帧数一致；7个交付文件跨机SHA全部相同。
人工查看60条中点联系表及4张详细帧，确认画面、双视角和标记可读；这是离线可视化
验收，不作为动力学或控制跟踪证明。原输入Z-up qpos不变、完整播放，没有二次旋转或裁剪。

`分析报告.md`包含汇总、目录差异和每条动作的视频时间索引，`analysis.json`包含全部
60条原报告指标、选样类别和验证身份；同时交付原run/summary及按SHA恢复的原筛选配置。
日志`/data0/user/liwei/logs/bumi_text_umr/motionmillion_quality_review.log`退出码0。
