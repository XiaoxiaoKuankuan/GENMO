# BUMI四库120帧文本训练

2026-09-22，`feature/bumi-text-only` 默认配置恢复4秒/120帧训练窗口。主入口仍为
`configs/exp/gem_bumi_text_fullseq.yaml`；此文件是本分支`configs/exp/`下唯一入口，
直接选择组件，不再继承SMPL实验配置。`configs/train.yaml`也默认选择它。实际实验名为
`gem_bumi_text_crop120`，序列契约为v2。原full300 checkpoint继续按自己的v1契约读取，
不能用新配置完整resume旧模型。相同契约可以恢复模型、优化器及调度器；当前普通
DataLoader不保存epoch中途的逐抽样游标，不承诺中途断点后逐batch完全复现原顺序。

## 完整存储与训练窗口

- 磁盘保留完整源qpos，`source_storage: full`转换清单通过全部PASS质量门禁，取消
  60–300帧训练候选限制；源动作仍须至少4帧。旧转换清单的历史长度策略兼容保留。
- Dataset在训练时随机选择caption，然后在该caption有效区间内均匀随机选连续120帧。
  区间不足120帧时重复最后一帧补齐，同时保留真实`length`及布尔`mask.valid`。
  padding不参与有效帧loss、时序差分或接触监督。完整qpos和完整文本索引不被改写。
- 验证固定第一条caption及其区间的中心窗口；不依赖worker或训练随机种子。
- 30Hz、Z-up、fe934资产、21关节顺序、qpos30表示及150-token T5-3B保持一致。
  `endecoder.sequence_mode: full`表示编解码按真实length处理边界，与Dataset的crop取窗
  职责不同，不能一起改成会破坏有效长度处理的模式。
- stats只使用train源中的不重复文本区间，确定性中心取窗，每窗口最后XY差分无效。
  它是可复现的归一化参考，不声称精确等于训练随机加权分布；必须使用新生成的
  `data_kind: bumi_text_crop120`统计量。

## 数据集与采样

| 数据集 | 默认概率 | 当前文本时间范围 |
|---|---:|---|
| BONES-SEED | 60% | 独立事件级秒标注，按事件description和区间训练 |
| HumanML3D | 25% | 本批部分记录已经按原文本时间裁剪，其余为整段文本 |
| KIT-ML | 10% | 本批whole_motion标注，有全段起止，不能视为事件级标注 |
| MotionMillion | 5% | 本批无可靠事件级时间，使用完整记录文本 |

比例是初始训练策略，可在`data.text_sampling.dataset_probabilities`调整。每次抽样先选
数据集，再均匀选canonical母来源，再选其镜像/子片段记录、caption和窗口。动作长、
caption多、镜像多不会自动增加母来源权重。每epoch默认262144次有放回抽样，DDP各rank
分担互不重叠的全局抽样序号；同一动作可以重复抽中。概率是长期期望，并非每个batch配额。
抽样由seed、epoch、全局抽样序号确定，worker数量不改变窗口随机序列。

MotionMillion有官方split的母来源保持官方分组；无官方split的来源，以及HumanML3D
train派生数据、KIT-ML、BONES按母来源的固定hash分为90%/5%/5%内部train/val/test。
该比例是哈希分布期望，不保证有限数据恰好占比。镜像与同源子片段同组。
HumanML3D内部留出不冒充官方val/test。当前四库间尚无完整统一母来源映射，报告明确
`cross_dataset_lineage_verified: false`；内部验证不用于宣称官方无泄漏基准成绩。

## BONES时间标注无需物理预裁

原始文件为
`/data0/user/liwei/datasets/BONES-SEED/metadata/seed_metadata_v002_temporal_labels.jsonl`。
按完整文件名精确关联，使用事件原文`description`。秒数映射到机器人30Hz帧时间线：
`start = ceil(start_time * 30)`，`end = floor(end_time * 30)`，区间为半开`[start,end)`。
禁止使用原人体50Hz直接计算机器人帧号。最多允许尾部0.1秒重采样差异并显式记录截边；
更大错位直接报错。没有可用事件或换算后不足4帧的事件不进入文本训练。

例如事件位于10–16秒，对应`[300,480)`：训练每次从该事件内随机取120帧，并使用该事件
文本；验证取第一事件的中心窗口。若事件只有2秒，则60帧有效、60帧padding；不借用
相邻事件凑满4秒。文件仍保存整段源动作，release记录每条caption的`caption_intervals`。
QualityGate在构建时再次核对时间文件SHA、事件原文与范围。BONES Dataset启用
`require_temporal_annotations: true`，旧的只有整段caption的release会明确失败。

## 四库数据准备与启动

以下使用服务器2实际训练环境。准备命令仅用于尚不存在的新输出目录；已完成的正式
release不重复构建。在线T5按采样文本编码，完整分片和train统计量须先完成。

```bash
cd /home/user/liwei/GENMO-bumi-text
export BUMI_TRAIN_PYTHON=/data0/user/liwei/envs/GENMO-cu128/bin/python
export BUMI_TEXT_PREP_ROOT=/data0/user/liwei/datasets/bumi_text_crop120_prepare_v2
export BUMI_TEXT_DATA_ROOT=/data0/user/liwei/datasets/bumi_text_crop120_v2
export BUMI_TEXT_ONLINE_T5=true
export BUMI_T5_MODEL_PATH=/data0/user/liwei/models/t5-3b_bed96aab
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

"$BUMI_TRAIN_PYTHON" tools/data/bumi/prepare_bumi_text.py four-conversion \
  --releases \
    /data0/user/liwei/datasets/motionmillion_umr_pass_latest \
    /data0/user/liwei/datasets/humanml3d_umr_pass_latest \
    /data0/user/liwei/datasets/kitml_umr_pass_latest \
    /data0/user/liwei/datasets/bones_seed_umr_pass_latest \
  --bones-temporal /data0/user/liwei/datasets/BONES-SEED/metadata/seed_metadata_v002_temporal_labels.jsonl \
  --output "$BUMI_TEXT_PREP_ROOT/conversion.json"

"$BUMI_TRAIN_PYTHON" tools/data/bumi/prepare_bumi_text.py build \
  --source "$BUMI_TEXT_PREP_ROOT/conversion.json" --output "$BUMI_TEXT_DATA_ROOT" \
  --text-feature-mode online_t5 --workers 32 --records-per-shard 64
"$BUMI_TRAIN_PYTHON" tools/data/bumi/prepare_bumi_text.py stats --sequence-mode crop --workers 32 \
  --root "$BUMI_TEXT_DATA_ROOT" --output "$BUMI_TEXT_DATA_ROOT/stats.json"
export BUMI_TEXT_STATS_PATH="$BUMI_TEXT_DATA_ROOT/stats.json"

# 预检报告是临时诊断产物，用后按agent.md精确路径清理。
"$BUMI_TRAIN_PYTHON" tools/data/bumi/prepare_bumi_text.py preflight --sequence-mode crop \
  --root "$BUMI_TEXT_DATA_ROOT" --split train --limit 0 \
  --output /tmp/bumi_text_crop120_preflight.json

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_CUMEM_HOST_ENABLE=0 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=lo TORCH_NCCL_BLOCKING_WAIT=1
"$BUMI_TRAIN_PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train.py exp=gem_bumi_text_fullseq
```

原生UMR构建依赖MuJoCo及资产核验，训练依赖GENMO完整环境。不要只因为Python可执行
就假定两个环境依赖相同。旧单库、SMPL及音乐实验入口已从本分支删除；相关独立功能
应在对应功能分支运行。`network`、`pipeline`、`endecoder`、四库Dataset、优化器等组件
配置仍被唯一入口实际引用，属于当前配置组成部分。测试源代码保留；验证本分支时使用
BUMI文本测试，不把依赖已删除实验入口的其他功能测试作为本分支训练入口。
固定评测复用`tools/eval/evaluate_bumi_text.py cohort --sequence-mode crop`，四库分别固定
验证窗口，GT渲染与模型生成按相同有效帧数比较。运行时、ONNX和TensorRT接口形状根据
checkpoint读取120或300，网页长度范围也来自模型契约；TensorRT须在目标GPU另行验收。

CPU合成数据、缩小网络测试只能证明这些代码路径及契约行为；生产数据、GPU训练、
生成质量及机器人闭环跟踪分别验收，实际运行证据追加于记录文本.md。

## 服务器2大显存训练接入补充（2026-09-22）

四库完整PASS转换后有551242条带可用文本的完整动作。MotionMillion约899万条caption，
不要求先物化全部150-token特征；`build --text-feature-mode online_t5 --workers 32`
显式发布在线文本release，训练设置`BUMI_TEXT_ONLINE_T5=true`和本地
`BUMI_T5_MODEL_PATH`。同一冻结T5-3B按抽到的原文批量编码，CPU LRU仅缓存有效token；
输出仍为150×1024，缓存保存FP16，网络接收FP32特征并进行bf16混合精度计算。
权重、config及SentencePiece指纹进入checkpoint资产契约；T5不进入优化器或checkpoint权重。
未配置在线编码器时读取在线release会报错，禁止把缺失特征当作全零文本训练。

原预计算release保持默认兼容。完整源qpos、BONES事件配对、随机120帧、短动作mask和
验证中心窗口均保持；`stats --workers 32`并行计算全部train样本的确定性区间窗口，
不读取验证/测试动作计算均值方差。并行构建仍在实际读取前后验证机器人/人体SHA。

大batch容量测量复用`tools/train/preflight_distributed.py --bumi-batch-size N
--bumi-data-root RELEASE --bumi-stats STATS --t5-path MODEL`，通过torchrun执行。
测量包括完整网络、实际T5、完整辅助损失、反向及AdamW状态；输出显存峰值、loss、
梯度范数和后续步骤耗时，不产生正式checkpoint。

完整网络含496545824个可训练参数。单卡实测batch256峰值分配64858.7 MiB、
预留66524 MiB；batch320分配78940.4 MiB、预留80992 MiB。单卡后续步骤分别
1.977/2.383秒，均包括原文编码与真实AdamW更新；这只是容量用小型真实数据集上的
测量，正式8卡速度另行记录。选择256为DDP及运行波动保留余量，8×256×1=2048，
全局batch与原64×8×4相同，学习率2e-4及215000步预算保持。训练8个worker/rank；
每5000步在线验证每库固定前64条中心窗口（8批×8），不把该小规模检查当作全量成绩。
