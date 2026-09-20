# BUMI 文本生成动作：完整序列 A0

本分支 `feature/bumi-text-only` 从 SMPL A0 提交
`460c2c9260b10ebb28f36da3c989ace05c7f4c8f` 创建；机器人 qpos30、FK、接触和 v5
损失来自 BUMI 音乐分支 `9046e36`。共享 Transformer、T5、训练入口和 A0 padding
修复以 SMPL 基线为准。开发目录是 `/home/weili/GENMO-bumi-text`，原工作树保持独立。

这是第一阶段代码交付：合成数据、真实机器人运动学和小型网络 CPU 测试。没有真实
BUMI 文本训练 checkpoint，没有启动正式训练、GPU推理、TensorRT构建、服务器任务或
替换现有网页服务。真实转换数据接入和真实模型验收分别属于第二、第三阶段。

## 1. 数据与模型实际做什么

```text
转换后的完整机器人动作 + 原caption + 150-token T5引用
  → BumiTextDataset：保留F帧，尾部补至300，真实length=F
  → qpos30编码、训练集统计量归一化、有效帧接触推导
  → 带噪[B,300,30] + [B,150,1024]文本/mask + length[B]
  → 1024维、16层、8头Transformer；max_len=120局部运动注意力
  → 30D去噪动作 + 2D左右脚接触logits
  → 反归一化、解码、FK、可选根XY足锁
  → qpos[F,28] / MuJoCo动画 / 网页MP4
```

qpos依次为根XYZ（米）、根四元数wxyz、21个原生顺序关节角（弧度）。只接受30FPS、
Z-up、地面零点明确的fe934机器人。30D依次为朝向坐标系XY位移2、相对默认根高1、
根6D旋转6、21关节角。网络不预测SMPL体型、相机或独立连杆位置；连杆通过FK计算。

默认根高统一为 **0.48120910 m**；新 stats 使用 v4 并显式记录基准，旧 v3 stats
加载时只补偿高度均值，使已有模型的世界动作高度保持一致。预览待机按实际姿态 FK
贴地。资产来源值与兼容公式见 [BUMI 默认根高与统计量兼容](BUMI_ROOT_HEIGHT.md)。

F取60–300，训练和验证均保留完整动作，没有随机/中心裁剪、时间伸缩、循环或镜像。
300只是batch长度；文本仍150token。训练每次读取完整动作随机取一条caption，验证取
第一条。保留worker种子机制，不因caption多或动作长重复采样。联合采样沿用分片感知
rank分配、drop_last和epoch乱序，不使用Lightning二次分片，不增加长度分桶A1。

完整动作消除了训练裁剪人为造成的对应范围错误，不能证明原caption本身准确，不能
保证模型质量提升。局部mask仍作用于稠密300×300注意力，显存/速度必须重新测量。

## 2. 交给转换流程的数据规范

模型、运动学与渲染XML的来源SHA必须一致：
`fe93472dd764704fe8389b0f82052ae84ed8bc90f6d71b1467872f86e08a9ad3`。
使用仓库 `configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json` 的有序
`joint_order`，不能用另一份同名BUMI的21关节排列。转换器输出每条NPZ的必需字段：

| 字段 | 形状/意义 |
|---|---|
| `qpos` | `[F,28]`有限数，根四元数单位wxyz |
| `fps` | 标量30；不允许以修改这个值代替真实重采样 |
| `joint_names` | 按qpos顺序的21个字符串，与kinematics严格相等 |
| `foot_contact`（可选） | `[F,2]`，左右脚0–1；缺少时从真实FK推导 |
| `foot_contact_available`（可选） | `[F,2]`有效性；不能以padding制造接触 |

转换清单 `conversion.json` 示例；尖括号内容必须替换为真实值，SHA不能照抄占位符：

```json
{
  "schema": "genmo.bumi_text_conversion.v1",
  "kinematics": {"path": "assets/kinematics.json", "sha256": "<文件SHA256>"},
  "records": [{
    "dataset": "motionmillion", "motion_id": "source_motion_001", "split": "train",
    "qpos_path": "motions/source_motion_001.npz", "fps": 30,
    "captions": ["A person walks forward."], "caption_ids": ["source_motion_001:0"],
    "text_source_motion_id": "source_motion_001",
    "ground_semantics": "retargeted_text_floor_zero_v1",
    "ground_alignment": {"applied": true, "offset_z": -0.06, "reference": "<真实地面依据>"},
    "provenance": {
      "source_id": "source_motion_001", "canonical_source_id": "<跨库可比的来源ID>",
      "interval_seconds": [0.0, 4.0], "retargeter": "UMR", "retarget_version": "<实际commit>"
    },
    "embeddings": [{
      "format": "motionmillion_t5_v1", "source_split": "train",
      "motion_manifest": "<原genmo_smpl_v1/manifests/train.json>",
      "motion_manifest_sha256": "<SHA256>",
      "embedding_manifest": "<原t5_3b_v1_fp16/manifests/train.json>",
      "embedding_manifest_sha256": "<SHA256>",
      "shard_id": 0, "record_index": 0, "text_index": 0,
      "motion_id": "source_motion_001", "caption_sha256": "<原文UTF-8 SHA256>"
    }]
  }]
}
```

记录的`motion_id`是动作ID；如重定向时改名，`text_source_motion_id`绑定原文本记录ID。
不填写时要求文本ID等于动作ID。每条caption对应一个embedding引用和caption ID。
路径相对于清单所在目录，可用绝对路径。构建后的外部文本引用会解析为绝对路径，
搬迁数据时应基于原文件相同SHA重新构建关联release；**不修改原SMPL/T5 manifest**。

MotionMillion引用校验原schema、split、build fingerprint、分片顺序和motion SHA绑定，
读取源motion的caption并检查motion_id/text_index，原150-token FP16特征直接复用。
大分片只在首次或文件身份变化时哈希，并使用有限LRU；不是每次getitem重算全文件SHA。
本轮未核验服务器生产分片；真实制品仍需运行preflight。

HumanML3D必须先按原时间字段得到对应完整片段：`0/0`全段描述与时间段描述分别处理，
只把相同区间caption归到一条动作；已切分记录不得再次切分。`interval_seconds`必须是
该记录真实的来源时间范围，时长与F/30允许约一帧取整误差。没有时间字段的长caption
不能据此凭空生成时间标注。超出60–300帧的记录只报告排除，不静默截断。

两套数据共享原始来源时填一致的`canonical_source_id`，以便排除训练与任一val/test
的区间重叠；同源同区间精确重复保留一条并报告。未填写时只能在数据集内比较，报告
`unverified_cross_dataset_lineage`，不能宣称已经排除所有跨库泄漏。联合训练按保留的
动作记录自然比例采样。坐卧/大倾角不应用音乐站姿筛选；预检只报告根高<0.3m或
倾角>0.8rad的记录，阈值是人工排查入口，不是合法性判决。

地面修正需在转换阶段明确完成。使用整条轨迹固定Z平移并记录offset，保留跳跃和竖直
运动，不能逐帧把脚压地。地面依据不明的记录先隔离。未进行修正须applied=false、
offset_z=0且说明原数据地面依据。工具不自动重定向、不自动落地、不覆盖原文件。

## 3. 接入、特征和统计量

本机开发复用已有训练环境，不安装另一套环境：

```bash
cd /home/weili/GENMO-bumi-text
PY=/home/weili/GENMO/.venv/bin/python
```

其他机器的训练环境按根README的完整训练环境安装步骤准备，随后`PY=.venv/bin/python`。
纯部署环境用第8节的一键安装，不安装完整训练包。下面所有数据命令都等待真实转换
数据与执行授权，未对生产数据运行。

已有MotionMillion150-token特征不重算。HumanML3D只有50-token特征时，不能补零冒充
150token；在新的转换清单中将需要编码记录的`embeddings`设为空数组：

```bash
: "${CONVERSION_JSON:?请指定真实转换清单}"
: "${FEATURE_OUTPUT:?请指定新的文本特征目录}"
: "${T5_LOCAL:?请指定本地T5-3B完整快照}"
"$PY" tools/data/bumi/encode_text_features.py --source "$CONVERSION_JSON" \
  --output "$FEATURE_OUTPUT" --t5-model "$T5_LOCAL" --device cuda:0
# 新清单位于 $FEATURE_OUTPUT/conversion.json；已有非空特征引用保持复用。
```

特征原生分片格式`bumi_text_t5_v1`记录motion_id、caption原文集合、encoder=t5-3b、
max_text_len=150、`embeddings[N,150,1024]` FP16、`attention_mask[N,150]`，引用含
分片SHA、record_index、text_index和caption SHA。既有50token引用不会自动被删除。

```bash
: "${CONVERSION_JSON:?指定含完整embedding引用的清单}"
: "${BUMI_TEXT_DATA_ROOT:?指定尚不存在的新release目录}"
"$PY" tools/data/bumi/prepare_bumi_text.py build \
  --source "$CONVERSION_JSON" --output "$BUMI_TEXT_DATA_ROOT"
"$PY" tools/data/bumi/prepare_bumi_text.py preflight \
  --root "$BUMI_TEXT_DATA_ROOT" --split train --limit 128 \
  --output "$BUMI_TEXT_DATA_ROOT/preflight_train128.json"
# 全量preflight使用--limit 0，必须另行明确执行；val/test分别检查。
"$PY" tools/data/bumi/prepare_bumi_text.py stats \
  --root "$BUMI_TEXT_DATA_ROOT" --output "$BUMI_TEXT_DATA_ROOT/stats.json"
"$PY" tools/data/bumi/prepare_bumi_text.py stats --dataset motionmillion \
  --root "$BUMI_TEXT_DATA_ROOT" --output "$BUMI_TEXT_DATA_ROOT/stats_motionmillion.json"
"$PY" tools/data/bumi/prepare_bumi_text.py stats --dataset humanml3d \
  --root "$BUMI_TEXT_DATA_ROOT" --output "$BUMI_TEXT_DATA_ROOT/stats_humanml3d.json"
```

三个stats分别只使用对应训练集有效元素，XY最后一帧不计入，其他28维仍计入。不能复用
SMPL151D或音乐stats。训练启动核对stats与train release SHA、kinematics和单/联合子集。
build输出`manifests/{train,val,test}.json`、qpos分片、kinematics副本、build_report。
失败时留下的新目录用于诊断，**不能当作完成的release使用**；检查并精确清理该目录后
再用新路径重试。全部工具拒绝覆盖已有报告和release，原生产数据不被改写。

## 4. 损失、padding和恢复边界

基准权重完整列于 `configs/pipeline/text_only_bumi_qpos30.yaml`，数值来自9046e36
的v5基础表，不继承音乐续训覆盖。例如表示根/旋转/关节=1/2/1，FK位置=1，接触BCE=1，
关节速度/加速度/jerk=0.1/0.02/0.003；其他关节限位、top-k/max、脚滑、接触高度、
穿地、GT相对倾角及GT包络项也按原表保留。这些是运动学监督，不是动力学约束。

`valid_per_sample`先对每条记录广播后的真实有效元素取均值，再乘该样本扩散时间权重，
最后batch平均。top-k/max仅选有效元素，无接触样本返回可求导零，仍占batch中的一份。
一至三阶差分分别要求2/3/4个真实帧；XY最后真实帧没有下一帧，不监督其位移。
最后真实帧接触速度使用最后一个真实帧对，不使用padding的静止差分。旋转、FK前先
以最后有效预测替换padding计算姿态，避免无效旋转；有效输出不读取padding key。
归约与旧legacy训练数值不相同，不应把归约差异全部解释为完整动作收益。

checkpoint包含文本/序列契约、30D/2D、qpos28/wxyz/关节顺序、资产SHA、完整损失配置、
去噪器与扩散配置、数据身份。默认随机初始化。首版阻断weights-only初始化及跨SMPL、
音乐、93D或不同fullseq契约的完整resume，后续迁移学习需要独立实现和实验。
相同契约的`resume_mode`会恢复optimizer/scheduler/global_step；不是只加载网络权重。
复用原训练配置和预算，不保证中途恢复的数据worker随机状态逐位等同不中断运行。

## 5. CPU检查与GPU smoke

CPU命令只产生临时目录，自动清理，不需要真实转换数据或GPU。合成T5不是实际T5验证：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
"$PY" - <<'PY'
import subprocess, sys, tempfile
with tempfile.TemporaryDirectory(prefix='bumi-text-cpu-') as d:
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
        '--basetemp', d+'/pytest', 'tests/bumi/test_bumi_text_fullseq.py',
        'tests/bumi/test_bumi_text_runtime.py', 'tests/bumi/test_bumi_text_integration.py',
        'tests/bumi/test_bumi_losses_v1.py', 'tests/bumi/test_bumi_feature_codec.py',
        'tests/test_motionmillion_fullseq.py', 'tests/test_demo_smpl_text.py',
        'tests/test_resident_text_motion.py', 'tests/test_text_motion_web.py'])
sys.exit(result.returncode)
PY
"$PY" scripts/train.py exp=gem_bumi_text_fullseq --cfg job --resolve
```

三个实验入口为`gem_bumi_text_fullseq`、`gem_bumi_motionmillion_text_fullseq`、
`gem_bumi_humanml3d_text_fullseq`。后两者需对应单集stats。
候选64×8×4=2048有效global batch **尚未显存验证**。默认AdamW2e-4，215000步，
LR warmup5000，机器人辅助项warmup10000；不能沿用120帧实验墙钟估计。

真实数据准备后导出环境变量，以下GPU命令仅供授权后执行，本轮未执行：

```bash
export BUMI_TEXT_DATA_ROOT=/实际路径/bumi_text_release
export BUMI_TEXT_STATS_PATH="$BUMI_TEXT_DATA_ROOT/stats.json"
export BUMI_KINEMATICS_PATH="$PWD/configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json"
```

单卡短程；临时路径经验证后清理，日志需要排障保留时另行明确保留：

```bash
(
  set -e
  BUMI_SMOKE="$(mktemp -d /tmp/bumi-text-gpu1.XXXXXXXX)"
  [[ "$BUMI_SMOKE" == /tmp/bumi-text-gpu1.* && -d "$BUMI_SMOKE" && ! -L "$BUMI_SMOKE" ]]
  trap 'rm -rf -- "$BUMI_SMOKE"; test ! -e "$BUMI_SMOKE"' EXIT
  PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$BUMI_SMOKE/mpl" CUDA_VISIBLE_DEVICES=0 \
  "$PY" scripts/train.py exp=gem_bumi_text_fullseq "output_dir=$BUMI_SMOKE/run" \
    training_budget.max_steps=8 training_budget.warmup_steps=2 \
    training_budget.auxiliary_warmup_steps=2 pl_trainer.devices=1 \
    pl_trainer.accumulate_grad_batches=1 data.loader_opts.train.batch_size=1 \
    data.loader_opts.train.num_workers=0 data.loader_opts.val.num_workers=0 \
    +pl_trainer.limit_val_batches=0 pl_trainer.num_sanity_val_steps=0 \
    +pl_trainer.enable_checkpointing=false
)
```

多卡在同样临时目录模式下，把设备和启动器改为下面命令；仍batch1，只验证DDP与反向：

```bash
(
  set -e
  BUMI_SMOKE="$(mktemp -d /tmp/bumi-text-gpu8.XXXXXXXX)"
  [[ "$BUMI_SMOKE" == /tmp/bumi-text-gpu8.* && -d "$BUMI_SMOKE" && ! -L "$BUMI_SMOKE" ]]
  trap 'rm -rf -- "$BUMI_SMOKE"; test ! -e "$BUMI_SMOKE"' EXIT
  PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$BUMI_SMOKE/mpl" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train.py exp=gem_bumi_text_fullseq "output_dir=$BUMI_SMOKE/run" \
    training_budget.max_steps=8 training_budget.warmup_steps=2 \
    training_budget.auxiliary_warmup_steps=2 pl_trainer.devices=8 \
    pl_trainer.accumulate_grad_batches=1 data.loader_opts.train.batch_size=1 \
    data.loader_opts.train.num_workers=0 data.loader_opts.val.num_workers=0 \
    +pl_trainer.limit_val_batches=0 pl_trainer.num_sanity_val_steps=0 \
    +pl_trainer.enable_checkpointing=false
)
```

CPU sampler检查不等于已运行多GPU。先确认真实release、所有rank batch数、NCCL、无NaN，
再独立验证batch64及累积4；数据过少时drop_last可能使loader为空。

## 6. 正式训练与完整恢复（未执行）

```bash
: "${BUMI_RUN:?指定全新正式实验输出目录}"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PY" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 scripts/train.py \
  exp=gem_bumi_text_fullseq "output_dir=$BUMI_RUN" \
  pretrain_ckpt=null ckpt_path=null resume_mode=null

# 完整恢复必须是同一BUMI文本实验；复用该实验的原配置/预算/资产。
: "${BUMI_TEXT_CKPT:?指定实际存在的BUMI文本fullseq checkpoint}"
test -f "$BUMI_TEXT_CKPT"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PY" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 scripts/train.py \
  exp=gem_bumi_text_fullseq "output_dir=$BUMI_RUN" \
  "resume_mode=$BUMI_TEXT_CKPT" pretrain_ckpt=null ckpt_path=null
```

## 7. 单条生成、常驻预览、网页与固定验证（无真实模型，未执行）

```bash
: "${BUMI_TEXT_CKPT:?指定真实BUMI文本checkpoint}"
: "${T5_LOCAL:?指定本地T5-3B目录}"
"$PY" scripts/demo/demo_bumi_text.py --checkpoint "$BUMI_TEXT_CKPT" \
  --t5-model "$T5_LOCAL" --prompt 'Walk forward, then turn left.' \
  --num-frames 240 --ddim-steps 50 --preview --render

"$PY" scripts/demo/demo_bumi_text.py --checkpoint "$BUMI_TEXT_CKPT" \
  --t5-model "$T5_LOCAL" --console --preview
```

控制台可输入文本或`play 文本`；`frames 240`、`steps 20`改后续请求，`pause/resume/stand`
控制本地播放，`status/quit`查看/退出。无音频，无音乐滑窗，不连接控制器。首次加载后
复用GENMO/T5；DDIM不变不重建。关闭窗口不关控制台；退出清理自己启动的窗口。默认
30FPS、seed42、CFG2.5、DDIM50，长度60–300。足锁只改根XY，可`--no-postproc`禁用。
输出同时含qpos_raw、qpos、contact logits、关节顺序与模型/资产指纹。

纯动画逐帧设置qpos并执行mj_forward，不执行mj_step；允许低姿态和大倾角，不采用
GMT Bridge站姿门限。坐卧动画可显示不等于机器人能保持该姿态。桌面需DISPLAY/图形库，
离线视频需MuJoCo EGL、FFmpeg/PyAV；生成后完整解码核验1280×720/30FPS/真实F帧。

网页复用既有入口。真实模型验收之前不要替换当前8766服务；可在获授权后用空闲端口：

```bash
"$PY" scripts/demo/demo_smpl_text_web.py --port 8767 \
  --output_root outputs/bumi_text_web_review
```

在 `http://127.0.0.1:8767/` 添加真实checkpoint；模型列表标明SMPL/BUMI与帧数范围。
默认入口地址仍8766。新模型如果从服务器迁回，应同时迁移同SHA stats/kinematics；
CLI支持`--stats`/`--kinematics`搬迁覆盖；原路径不存在时，网页和CLI也会检查checkpoint
同目录的assets/原文件名或assets/kinematics.json、assets/stats.json，仍须SHA完全相等。
模型权重、文本和动作契约决定渲染后端，不能用文件名伪装。保留串行任务、手动播放、
失败保留旧视频、刷新恢复和最多60条历史；本轮没有改动在用站点或公开隧道。

固定128条，两集各64，默认每条第一caption、seed42、按GT真实F、DDIM50/CFG2.5：

```bash
: "${COHORT_JSON:?指定新的固定验证清单路径}"
: "${EVAL_OUTPUT:?指定新的评测目录}"
"$PY" tools/eval/evaluate_bumi_text.py cohort --root "$BUMI_TEXT_DATA_ROOT" --output "$COHORT_JSON"
"$PY" tools/eval/evaluate_bumi_text.py run --root "$BUMI_TEXT_DATA_ROOT" \
  --cohort "$COHORT_JSON" --checkpoint "$BUMI_TEXT_CKPT" --output "$EVAL_OUTPUT" --render
```

结果为每样本GT、原始/后处理NPZ、运动学JSON，可选视频，以及report.json和人工表
human_ratings.csv。指标分别查看关节限位、接触脚滑/穿地、速度/加速度/jerk，不混成
“模型总分”。人工建议至少3位评审盲评，每列1–5分：文本匹配（未执行→完整执行）、
连贯性（明显断裂→连贯）、姿态合理性（明显异常→未见异常），记录具体失败及评审ID。
汇总时分数据集、长度段报告均值/分布和评审差异，不能以脚滑较少替代文本匹配。
报告协议为full_sequence_matched_length；不是原人体官方4帧单位裁剪协议，不能
输入BUMI qpos计算MotionMillion官方R-Precision/FID，也不能与历史120帧SMPL报告混合。

## 8. 导出、TensorRT、独立部署包

所有导出/构建/对照留在训练仓库。下面路径变量必须指向新的输出，尚无真实产物。

```bash
: "${ONNX_OUTPUT:?如新的空目录下model.onnx的绝对路径}"
: "${ENGINE_OUTPUT:?如新的目录下model.engine的绝对路径}"
: "${PARITY_REPORT:?指定新的数值对照JSON}"
: "${DEPLOY_OUTPUT:?指定尚不存在的部署包目录}"
"$PY" tools/export/bumi_text.py export --checkpoint "$BUMI_TEXT_CKPT" --output "$ONNX_OUTPUT"
"$PY" tools/export/bumi_text.py build --onnx "$ONNX_OUTPUT" --output "$ENGINE_OUTPUT" --device cuda:0
"$PY" tools/export/bumi_text.py validate --checkpoint "$BUMI_TEXT_CKPT" \
  --onnx "$ONNX_OUTPUT" --engine "$ENGINE_OUTPUT" --device cuda:0 \
  --ddim-steps 50 --output "$PARITY_REPORT"
"$PY" tools/export/bumi_text.py package --onnx "$ONNX_OUTPUT" --engine "$ENGINE_OUTPUT" \
  --validation-report "$PARITY_REPORT" --output "$DEPLOY_OUTPUT"
```

先省略build/engine参数可做PyTorch↔CPU ONNX验证。export要求空输出目录以保护外部
权重；固定对外batch1、计算300帧、文本150×1024，六个输入为带噪动作、timestep、文本、
token mask、真实length、CFG scale。内部成对计算CFG，两个输出为300×30与300×2。
length是运行时输入，最终只保存F帧；T5、DDIM、解码/FK不进入ONNX图。

validate对7种长度比较单步、完整DDIM和解码误差，使用相同有效噪声。TRT阈值分单位
记录于报告（表示、接触、根位置米、关节/根旋转弧度）；它们是数值准入阈值，不保证
生成质量或可跟踪性。真实FP16 engine仍需实际通过该工具及真实文本验证。
导出的大模型外部权重逐个SHA绑定，不能只拷贝.onnx遗漏data文件。

package包含运行依赖闭包、相对路径ONNX/外部权重、可选engine与记录、训练stats和
kinematics、匹配XML/mesh、deployment.json，不含checkpoint、训练Dataset、Lightning、
Hydra或导出器。模型资产启动时严格SHA核验；software_inventory记录打包时源码身份，
不阻止用户编辑deployment.ini。无验证报告可以打开发包，但状态明确not_validated。

T5体积较大，包不自动下载。复制完整本地T5快照（含权重、config、tokenizer，HF缓存
软链接需解引用）到包内models/t5-3b，或在deployment.ini填写该机器的已有路径。

```bash
mkdir -p "$DEPLOY_OUTPUT/models/t5-3b"
cp -aL "$T5_LOCAL/." "$DEPLOY_OUTPUT/models/t5-3b/"
# 可压缩整个DEPLOY_OUTPUT；不要把原训练目录.venv拷到另一台机器。
cd "$DEPLOY_OUTPUT"
bash install.sh
bash run.sh
```

安装器自动准备uv/Python3.10/.venv，固定版本安装PyTorch、T5、ORT、MuJoCo/PyAV及
传递依赖，TensorRT后端另装10.13.3.9绑定与运行库；缺少FFmpeg/libGL时安装系统包，
这一步可能需要sudo。无需手动activate、不需要Redis/ROS/Docker/GMT或Node。
ONNX后端在device=cpu时用CPU；device=cuda:0时要求ORT CUDA可用。TensorRT engine
绑定构建GPU信息和库版本，同为4090也需版本兼容，不兼容时回训练仓库相应环境重建。
驱动不由脚本自动修改；真实新机器一键安装、GPU兼容和桌面显示仍待验收。

只需在deployment.ini编辑backend/device/t5_model/num_frames/ddim_steps及
`[preview] enabled=true/false`；这里没有需同步的控制器端口。原BUMI音乐部署包不变。

## 9. 第一阶段验收边界

已完成的具体测试、通过数、跳过原因和提交记录以根 `记录文本.md` 当次条目为准。
测试采用系统临时目录并清理；真实SMPL CPU回归可临时只读挂接现有本地人体资产，
不会将这些资产提交到新分支。合成小网络CPU ONNX通过，不代表真实1024D/16层
网络的GPU显存、TensorRT导出、视觉或质量验收通过。

待转换数据到来：核验真实NPZ/T5映射、来源划分/去重、地面依据、统计量、固定128条；
授权后执行单卡/多卡短程、全损失有限性及完整resume。待真实checkpoint到来：逐帧
生成、MuJoCo图形、网页模型切换、PyTorch/ONNX/TRT数值对照、新电脑部署与人工评分。
不会自动启动训练、全量转换、全量评测或停止已有训练。
