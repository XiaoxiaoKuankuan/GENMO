# MotionMillion UMR → BUMI 文本动作预处理

适用分支：`feature/bumi-text-only`。入口复用 `tools/data/bumi/prepare_bumi_text.py`。
质量计算复用音乐分支的 `filter_sonic_npz_motions.evaluate_motion`，原音乐代码和阈值配置不改动。

## 数据与判定契约

- 输入是 `folder*/bumi3/batch_summary.json` 及其登记的原生UMR NPZ，同时对照实际文件集合，记录缺失和未登记文件。
- 每条检查完整NPZ解压/CRC、必需字段、float32 qpos[T,28]、21个唯一关节名称、单位wxyz、30Hz、从0连续的frame_ids。
- 核对summary与原人体路径、帧数、FPS、数值及 `source_format=motionmillion_272/output_up=y`。UMR已将机器人输出转为Z-up，不再次旋转。
- 核对XML、运动学JSON与资产清单中的所有网格SHA；关节限位取XML和UMR生成配置的交集，包括双膝0.1rad下界。
- 复用速度、加速度、jerk、根位移/旋转速度检查。新增完整脚网格穿地、候选支撑脚滑、长时间双脚悬空、非相邻刚体凸包碰撞。
- 支撑候选依据脚底高度、垂向速度、持续时间，不使用水平速度提前排除滑动帧。
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

