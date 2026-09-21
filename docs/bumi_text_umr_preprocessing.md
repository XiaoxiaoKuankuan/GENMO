# MotionMillion / HumanML3D / BONES-SEED UMR → BUMI 文本动作预处理

适用分支：`feature/bumi-text-only`。入口复用 `tools/data/bumi/prepare_bumi_text.py`。
质量计算复用音乐分支的 `filter_sonic_npz_motions.evaluate_motion`，原音乐代码和阈值配置不改动。

## BONES-SEED-SMPL：完整轨迹筛选与文本核验（2026-09-21）

复用原筛选器和渲染器，`BUMI_DATASET=bones_seed`接受单批`bumi3/batch_summary.json`。
源人体是Z-up、50Hz的72维姿态，输出是30Hz原生qpos28；逐条验证完整NPZ、原始PKL
身份和`floor((源帧数-1)/50*30+1e-9)+1`输出帧数，不再次重采样或修正动作。
共享质量配置的支撑脚滑移REJECT从1.5改为3.0m/s，REVIEW仍为0.75m/s，连续阈值
仍为6帧；其他质量规则保持不变。历史MotionMillion/HumanML3D结果仍绑定旧配置SHA，
本次不重写历史报告。

```bash
cd /home/user/liwei/GENMO-bumi-text
BUMI_DATASET=bones_seed BUMI_WORKERS=64 bash scripts/prepare_bumi_text_umr.sh
/home/user/miniconda3/envs/ykj_umr/bin/python -B tools/data/bumi/prepare_bumi_text.py bones-pass \
  --quality-report /data0/user/liwei/dataset_reports/bones_seed_umr_bumi3_latest \
  --output /data0/user/liwei/datasets/bones_seed_umr_pass_latest
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONDONTWRITEBYTECODE=1 \
  /home/user/miniconda3/envs/ykj_umr/bin/python -B -u tools/eval/render_bumi_motion.py \
  --quality-report /data0/user/liwei/dataset_reports/bones_seed_umr_bumi3_latest \
  --output-dir /data0/user/liwei/dataset_reports/bones_seed_umr_bumi3_latest/visual_review \
  --per-group 30 --individual-clips
```

`bones-pass`发布全部PASS原生NPZ及`manifests/pass.jsonl`，不按训练候选的60..300帧
额外删除短/长动作；`all_records.jsonl`保存全体质量状态与文本匹配结果。文本来自
`/data0/user/liwei/datasets/BONES-SEED/metadata/seed_metadata_v004.csv`的四条原始
自然语言描述，严格按移除转换后缀的`filename`匹配；缺失文本保留为空并分别输出
`missing_text_ids.txt`、`pass_missing_text_ids.txt`。源标注的原始帧数不冒充50/30Hz
帧数。没有生成caption、T5特征或虚构train/val/test划分，split明确为unassigned。
分段描述另外存在于原始`seed_metadata_v002_temporal_labels.jsonl`，本次不把子段
标注当成完整动作描述。`dataset_info.json`绑定质量身份、文本和manifest SHA。

渲染仍从完整报告按状态/原因选择30条PASS与30条REJECT，双视角完整回放，输出两份
合集及`pass/`、`reject/`中的逐条视频和原生NPZ；每个视频解码核对30Hz和完整帧数。
候选池先按母来源去重再限制容量，镜像或子片段不挤占不同来源的名额；选样仍保留
原有活动/时长条件，不能把目的性视频样本的类型比例作为总体估计。

2026-09-21全量筛选结果：131,454条，PASS772（0.5873%）、REVIEW485（0.3690%）、
REJECT130,197（99.0438%），INVALID/ERROR均0。全部772条PASS已发布到
`/data0/user/liwei/datasets/bones_seed_umr_pass_latest`，与既有MotionMillion/HumanML3D
发布目录同级；194,450帧/1.800463小时，其中553条为60..300帧，另外219条长动作
仍完整保留。全库文本覆盖131,418条/525,672条caption；PASS有文本766条/3,064条
caption，缺失文本6条，其余30条文本缺失属于非PASS。

报告位于`/data0/user/liwei/dataset_reports/bones_seed_umr_bumi3_latest`。逐动作合并
左右脚/多个碰撞对后，REJECT触发原因包括：根倾角130,116、根高度35,534、脚穿地
25,743、连续性20,879、自碰撞13,137、支撑脚滑移26条；原因重叠，不能相加。
多数淘汰来自>30度连续至少15帧的根倾角规则，不能用放宽脚滑阈值解释或消除。
这些统计反映当前重定向产物和门禁规则，未证明上游人体坐标及机器人根朝向的语义
正确性，也不等同于动力学、控制器跟踪或实机验证。

`delivery_verification.json`记录对全部131,454条文本绑定的独立复核，以及全部772个
交付NPZ的源SHA、30Hz、有限qpos[T,28]、完整帧数及文件集合一致性；数据根目录
另有`README.md`和`SHA256SUMS`。本次筛选fingerprint：
`686fee77c3d0fbd72fea15b30e82e21ec67812eae9f42f94bd4417980e945cdb`。

视频交付在报告的`visual_review/`：`high_quality_30.mp4`为5,433帧/181.1秒，
`low_quality_30.mp4`为5,795帧/193.1667秒；另有`pass/`与`reject/`各30个完整
MP4及对应原生NPZ。全部视频H.264、1280×720、30Hz，60条来自60个不同母来源。
PASS选样类型为转向4/低姿态26；REJECT为根倾角10/脚滑10/碰撞6/穿地2/突变2，
只是目的性复核样本。`verification.json`记录完整帧数、视频/轨迹SHA和渲染commit。
本地查看目录：`/home/weili/GENMO-bumi-text/outputs/bones_seed_umr_quality_review`。

## HumanML3D 交付适配与完整训练数据

HumanML3D 与 MotionMillion 的 UMR 机器人输出都按 Z-up 处理。配置的
`source_contracts` 仅区分重定向前的人体来源，不对机器人 qpos 再旋转或贴地。
以下默认目录针对服务器2的23,242条HumanML3D训练派生交付：

| 用途 | 路径 |
|---|---|
| 原始机器人交付 | `/data2/user/liwei/hml3d_umr` |
| 补齐的原始人体包 | `/data0/user/liwei/datasets/humanml3d_umr_source_v1` |
| 筛选报告 | `/data0/user/liwei/dataset_reports/humanml3d_umr_bumi3_latest` |
| 最终完整动作训练分片 | `/data0/user/liwei/datasets/bumi_text_humanml3d_umr_pass_latest` |
| 配套T5特征 | `/data0/user/liwei/datasets/bumi_text_humanml3d_umr_pass_latest_t5` |

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

2026-09-20新规则全量验收：23,242条，PASS9,060、REVIEW7,644、REJECT6,538，INVALID/ERROR均为0。PASS中390条不足60帧、9条超过300帧，最终train8,661条、1,777,644帧、16.4597小时、23,387条caption，保留37.2644%，val/test为0。原动作4,322、镜像4,339，母来源4,206；完整源7,793、既有子片段868，两组是交叉维度。

训练release含17动作分片及资产/统计，共23文件237,211,713字节；T5为68分片及清单，共69文件7,215,192,474字节，合计7,452,404,187字节。全部8,661条通过现有loader逐caption核验，crop_count=0，train统计量为v4且非placeholder。报告目录`delivery_summary.json`绑定指纹、文件SHA及数量，`training_preflight.json`是全量加载结果；原包仍仅含训练身份，val/test清单为空。

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
- `posture_policy: diagnostic` 保留既有低姿态诊断，不执行音乐任务的站立风格门禁。新增`root_tilt`是独立门禁：根倾角>30°连续至少15帧会淘汰，包括满足该条件的弯腰或躺姿；不受diagnostic策略豁免。穿地、滑移、碰撞等独立规则仍生效。
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
/data0/user/liwei/dataset_reports/motionmillion_umr_bumi3_latest/
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
  --quality-report /data0/user/liwei/dataset_reports/motionmillion_umr_bumi3_latest \
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
  --quality-report /data0/user/liwei/dataset_reports/motionmillion_umr_bumi3_latest \
  --output-dir /data0/user/liwei/dataset_reports/motionmillion_umr_bumi3_latest/visual_review \
  --per-group 30
```

输出两段H.264/30FPS双视角视频、中文`分析报告.md`、含每条动作完整指标和视频时间
索引的`analysis.json`，以及原run/summary/按SHA从Git恢复的筛选配置。高组按活动类型
和目录均衡选PASS，低组按脚滑、自碰撞、穿地、突变选REJECT；都是目的性复核样本，
不能拿60条的分布估计全库比例。每条完整播放，原qpos/帧率不修改；右视角半透明地面
便于观察穿地，红框对应原报告当前异常帧。原始动作和人体SHA、真实XML/全部网格须
匹配报告；渲染结束核对解码帧数后原子发布。新结果须用独立目录，验收后替换旧合集。

### MotionMillion当前复核

服务器2正式视频位于`/data0/user/liwei/dataset_reports/motionmillion_umr_bumi3_latest/visual_review`；本地位于`/home/weili/GENMO-bumi-text/outputs/motionmillion_umr_quality_review`。高组30条5,551帧/185.0333秒，低组30条3,906帧/130.2秒。高组五种活动类别各6条；低组根倾角10、脚滑10、碰撞6、穿地2、突变2。两组共60个不同母来源，原caption与文本SHA附在analysis.json，数据统计和文本覆盖见本文末尾当前结果。

### HumanML3D复核使用同一入口

将上述`--quality-report`改为`/data0/user/liwei/dataset_reports/humanml3d_umr_bumi3_latest`，
`--output-dir`改为该目录下的`visual_review`即可。入口自动识别报告数据集，复用原
HumanML3D加载器校验源交付清单、机器人/人体SHA、迁移前路径、XML与镜像/子片段
身份，不对已经Z-up的机器人输出再次旋转。视频额外显示第一条原始caption，完整
caption集合写入analysis.json；按母动作canonical_source_id去重，避免同组被镜像
或同一母动作的子片段重复占据。不存在的REJECT类型由其他实际问题类别补齐。

分析中分别统计原动作/镜像和完整源动作/子片段，避免将两个交叉维度相加。若原报告
已有delivery_summary.json，还须核对其筛选身份、训练条数和关键文件SHA后记录
已有训练交付；不能仅根据筛选候选数宣称训练数据已经构建。不同数据集的视频属于
不同复核场景，本次HumanML3D不替换此前用户要求保留的MotionMillion交付。

### HumanML3D当前复核

服务器2正式视频位于`/data0/user/liwei/dataset_reports/humanml3d_umr_bumi3_latest/visual_review`；本地位于`/home/weili/GENMO-bumi-text/outputs/humanml3d_umr_quality_review`。高组30条6,763帧/225.4333秒，低组30条5,837帧/194.5667秒。高组五类各6条，低组根倾角14、脚滑10、碰撞6；没有用不存在的穿地或突变REJECT凑数。两组共60个不同母来源。

两数据集视频均为H.264/1280×720/30FPS，原qpos/完整时序保持，仅mj_forward，无动力学推进。全部四视频两端SHA和完整解码通过，120条中点及详细帧画面复核通过。高组是按当前规则通过且有活动的目的性样本，不是随机总体抽样或实机证明。服务器与本地旧完整报告/视频均由上述最新结果替代；HumanML3D旧训练release/T5也在新加载验证后清理。

## 2026-09-20用户确认的新筛选规则

脚滑REVIEW/REJECT分别改为>0.75/>1.5 m/s，仍需连续6个相邻帧区间。
自碰撞额外深度阈值保留>1/>5 cm，REVIEW/REJECT均需连续超过10帧，即至少11帧。
新增独立根倾角规则：根竖直轴相对世界Z轴倾角>30度，连续至少15帧即REJECT；
绕世界Z轴的纯转向不触发。它独立于旧的坐躺诊断策略，因此弯腰/侧倾/躺姿持续超限
也会被淘汰。其他数值、穿地、悬空及完整60–300帧要求不变。

启动器默认新报告目录改为`humanml3d_umr_bumi3_latest`和
`motionmillion_umr_bumi3_latest`；HumanML3D训练输出改为
`bumi_text_humanml3d_umr_pass_latest`及对应`_t5`。新一轮全量结果核验通过后删除
被替代的旧报告和旧质量视频，保留用户原始数据；旧结果的数量记录仅作变更历史。

### 原生UMR MotionMillion文本绑定

通过`prepare_bumi_text.py motionmillion-texts --quality-report REPORT --texts-archive texts.tar.gz --split-archive split.tar.gz --output CATALOG`构建文本SQLite目录，再用`bind-motionmillion-texts --quality-report REPORT --text-catalog CATALOG`核验全量动作来源并写入`text_binding.json`。索引保存原始文本字节、完整来源ID、双输入SHA及官方t2m_60_300划分；缺失文本或split明确报告，不推断。原文本包放在服务器2`/data0/user/liwei/datasets/motionmillion_text_source_v1/`，最新匹配目录为`/data0/user/liwei/datasets/motionmillion_umr_text_latest/`。文本绑定不等同于T5特征编码或可训练分片构建。最新报告带text_binding时，已有质量视频入口自动核验并显示MotionMillion原caption。

## 当前最新全量重筛结果

2026-09-20按已确认新规则执行，报告根目录均改为`*_latest`。两批complete且partial_scan=false，INVALID/ERROR均为0。逐条核对新旧relative_path、完整来源ID、双输入SHA及帧数完全一致；`refilter_comparison.json`只保留状态变化汇总，旧完整报告和旧合集在新结果验收后删除。

| 数据集 | 原始动作 | PASS | REVIEW | REJECT | 完整60–300帧PASS候选 | 保留比例 |
|---|---:|---:|---:|---:|---:|---:|
| MotionMillion | 559,924 | 310,023 | 66,942 | 182,959 | 156,385 | 27.9297% |
| HumanML3D | 23,242 | 9,060 | 7,644 | 6,538 | 8,661 | 37.2644% |

脚滑REJECT从70,459/9,845降至6,893/863；碰撞REJECT从21,400/512降至14,443/276；新增根倾角REJECT为169,107/5,874（依次MotionMillion/HumanML3D，各原因可能重叠）。MotionMillion旧PASS中79,385条、HumanML3D旧PASS中1,415条被新增根倾角约束淘汰。候选净增574/1,032；不是简单保留旧PASS并追加动作。

MotionMillion原文本匹配559,922/559,924条，共11,791,690条caption。缺失ID为`Mirror_MotionGV/folder8/494361`与`Mirror_MotionGV/folder8/500208`，分别54/30帧，均不是训练候选；不生成替代文本。156,385条质量及长度候选全部有文本，共3,294,293条caption，并全部恢复官方t2m_60_300划分：train125,085、val7,987、test23,313。该官方划分保持原样；后续混合训练还需进行HumanML3D跨来源去重与泄漏检查，不能直接把156,385条全作为train。本轮MotionMillion交付为全量质量报告和绑定文本目录，尚未编码其新的BUMI T5或构建训练分片。

### 待复核REVIEW第三组视频

REVIEW当前不进入训练。复用原入口，加`--quality-groups review`只渲染待复核组，每个数据集各30条：

```bash
python tools/eval/render_bumi_motion.py \
  --quality-report /data0/user/liwei/dataset_reports/humanml3d_umr_bumi3_latest \
  --output-dir /data0/user/liwei/dataset_reports/humanml3d_umr_bumi3_latest/visual_review/review \
  --quality-groups review --per-group 30
```

MotionMillion替换对应数据集目录名。仅选完整60–300帧、training_eligible=false的REVIEW，按实际生效复核原因分层，目录均衡、稳定哈希选样并去重；不只挑极端峰值，样本原因比例也不能代表总体。视频黄色标注REVIEW / NOT IN TRAINING，原动作、文本、配置和资产身份继续核验。默认PASS/REJECT入口行为不变；REVIEW子目录作为现有最新复核集合的第三组，保留前两组对照。

已交付：MotionMillion REVIEW30条，5,020帧/167.3333秒，选样类别脚滑8、碰撞8、悬空7、穿地7；HumanML3D REVIEW30条，6,307帧/210.2333秒，脚滑8、碰撞6、悬空6、穿地4、连续性6。两段均1280×720/30FPS，文件名`review_30.mp4`，附analysis.json、分析报告.md、原run/summary/config。本地位于既有`outputs/{motionmillion,humanml3d}_umr_quality_review/review/`；每库30个不同母来源，文本/动作SHA和完整解码通过。REVIEW仍不进入训练；所选HumanML3D动作与现有8,661条train清单交集为空。
