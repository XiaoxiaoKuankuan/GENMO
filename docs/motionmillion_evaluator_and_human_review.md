# MotionMillion 自动 evaluator 与人工评测说明

本文说明当前 GENMO 文本生成动作模型的评测状态、已下载资产、实际依赖和可复现的加载
检查，并区分 MotionMillion 自动指标与 MotionMillion-Eval 人工评审。更新时间：
2026-09-17；所属分支：`feature/smpl-text-only`。

## 1. 当前已经做了什么

本次检查未发现当前 MotionMillion 文本模型的正式 R-Precision@1/@2/@3、Matching
Score、FID、Diversity 报告；服务器1原配置的
`/data0/user/liwei/datasets/MotionMillion/official_evaluator` 仍为空目录。
训练 loss、固定小样本监控中的 FK diversity/足滑代理指标，都不能替代这些正式指标。

本次已把自动 evaluator 的代码、权重、文本骨干、官方归一化统计量和126条人工评测
prompt 放到本机，并完成 CPU 离线双编码器加载检查。没有执行完整验证集生成或质量评分。

本地目录：

```text
/home/weili/GENMO/inputs/MotionMillion/official_evaluator/
├── code/                              # 官方仓库，固定 commit，未修改
│   ├── models/evaluator_wrapper_motionmillion_rpr272.py
│   ├── mld/models/architectures/temos/  # 文本/动作编码器
│   ├── dataset/dataset_TM_eval_motionmillion.py
│   ├── utils/eval_trans.py
│   └── assets/infer_batch_prompt.txt   # 官方126条英文
├── checkpoints/evaluator/epoch=199.ckpt
├── distilbert-base-uncased/            # 骨干权重、配置、词表和 tokenizer
├── dataset/MotionMillion/mean_std/vector_272/
│   ├── mean.npy
│   └── std.npy
└── asset_manifest.json                # 来源、文件SHA256、环境与加载验证记录
```

目录实际占盘约1003 MiB；不是完整 MotionMillion 数据集。`inputs/` 按仓库规则被 Git
忽略，Git 推送不会上传这些大文件，也不会自动把它们装到服务器或其他电脑。

| 资产 | 固定身份与本次来源 |
|---|---|
| 官方代码 | `VankouF/MotionMillion-Codes`，commit `8a2a7dfa66ecb6a1533d3d9cb49c743a697e1e1c` |
| evaluator checkpoint | 官方下载脚本指向的 Google Drive 文件，764972866 bytes；SHA256 `0e46ad8e7eeec72f4bab0f522ed55be655b223e46f626a67cc2754c1877f3f8c` |
| DistilBERT | `distilbert/distilbert-base-uncased`，revision `12040accade4e8a0f71eabdb258fecc2e7e948be` |
| mean/std | 服务器1已有官方 release `007582c4fc9637a3f36e548a67be1ef6eaf881a5`；原 `mean_std/Mean.npy`、`Std.npy`，复制后按 wrapper 要求改为小写文件名；字节内容及 SHA256 与源文件一致 |
| 人工 prompt | 官方代码的 `assets/infer_batch_prompt.txt`，126条；SHA256 `5d0daad4842441307e157c613ae22d3d8f41bd6058c2a51766e7f8c6b2740550` |

checkpoint 内部保存 epoch=199、global_step=87800，包含414个 state_dict 项；还携带训练
元数据、优化器、decoder 等内容。评测 wrapper 实际只加载 `textencoder.*` 和
`motionencoder.*`，两者均使用 strict=True。不能把这个 evaluator 的 step 当作 GENMO
生成模型的训练步数。

## 2. evaluator 如何工作

```text
英文条件 ── DistilBERT + 4层 Transformer ── 文本特征 [B,512]
                                                        │
生成动作/真实动作 ── 原始尺度272D ── 官方mean/std归一化    │
                    └─ 4层动作 Transformer ── 动作特征 [B,512]
                                                        │
                         距离、检索排名、分布统计 ←─────┘
```

两个分支的 VAE 形式输出中，wrapper 取分布均值 `.loc` 作为特征，没有在评分阶段随机
抽取 latent。动作分支按长度 mask，官方实现张量补齐到300帧。

GENMO 自己使用 T5-3B、150 token 作为生成条件；这里的 DistilBERT 是独立评判模型，
没有把 GENMO 改成 DistilBERT，也没有把 GENMO 的文本上限改回50。官方 tokenizer 此处
没有显式截断，文本还受 DistilBERT 自身位置长度限制，不能把生成器上限套到 evaluator。

| 指标 | 本仓库计算含义 | 如何解读 |
|---|---|---|
| R-Precision@1/@2/@3 | batch=32 时，对每个文本在32个动作 embedding 中按距离排序，统计配对动作进入前1/2/3名的比例 | 越高越好；候选数、样本和随机种子必须一致；不是逐个动作的百分制人工评分 |
| Matching Score / MM-Dist | 每条条件与其生成动作的 embedding 欧氏距离，再求均值 | 越低越好；这里 MM 指跨模态距离，不是下面的多样性 |
| FID | 比较生成动作与真实动作 embedding 的均值、协方差 | 越低越接近真实数据分布；不是图像 Inception 网络，也不直接证明动作符合文本 |
| Diversity | 在生成动作集合中采样不同动作，计算 embedding 距离均值 | 应结合真实数据参考范围；越大不一定越好，异常动作也可能增大距离 |

例如 R@1=0.40 表示约40%的配对项在该检索协议下排名第一，不表示40%的动作完全正确。
同一文本多种 seed 的变化程度通常另称 Multimodality，不能用集合 Diversity 替代。

官方实现依据：[wrapper](https://github.com/VankouF/MotionMillion-Codes/blob/8a2a7dfa66ecb6a1533d3d9cb49c743a697e1e1c/models/evaluator_wrapper_motionmillion_rpr272.py)、
[文本编码器](https://github.com/VankouF/MotionMillion-Codes/blob/8a2a7dfa66ecb6a1533d3d9cb49c743a697e1e1c/mld/models/architectures/temos/textencoder/distillbert_actor.py)、
[动作编码器](https://github.com/VankouF/MotionMillion-Codes/blob/8a2a7dfa66ecb6a1533d3d9cb49c743a697e1e1c/mld/models/architectures/temos/motionencoder/actor.py)。

## 3. 当前环境和加载验证

这条 encoder 路径在现有 `.venv` 上已经通过，没有安装官方整份旧训练 requirements，
没有替换 GENMO 的 PyTorch。下载器通过 `uv tool run --from gdown==5.2.0 gdown` 隔离运行。
直接使用本仓库指标脚本不需要另外下载 GloVe、FLAN-T5-XL 或官方3B/7B生成模型；完整
官方生成/训练入口可能需要它们，不能把两种依赖范围混淆。

实际通过的环境快照：Python3.10，torch `2.6.0+cu124`，pytorch-lightning `2.6.5`，
transformers `5.13.1`，numpy `1.23.5`，scipy `1.15.3`，omegaconf `2.3.1`，
huggingface-hub `1.23.0`。这是本机加载验证记录，不是论文同环境数值复现声明。

官方 wrapper 的 `torch.load` 未指定 `weights_only`，而该 checkpoint 含有 OmegaConf、
NumPy、官方数据类等训练元数据。PyTorch2.6 默认 weights_only=True 会拒绝加载。
本次保持官方源码和 checkpoint 原样，仅对下面独立验证进程设置
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`，恢复官方旧版加载语义。此方式仅用于这里已经核对
下载来源和 SHA256 的官方 checkpoint，不要设成全局环境或用于陌生文件。

可复现本次加载检查，输出只打印到终端，不生成动作或质量分数：

```bash
cd /home/weili/GENMO
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=4 .venv/bin/python - <<'PY'
from pathlib import Path
from argparse import Namespace
import os
import sys
import torch

root = Path("inputs/MotionMillion/official_evaluator").resolve()
sys.path.insert(0, str(root / "code"))
os.chdir(root)
from models.evaluator_wrapper_motionmillion_rpr272 import EvaluatorModelWrapper272RPR

torch.manual_seed(42)
evaluator = EvaluatorModelWrapper272RPR(Namespace(dataname="motionmillion"), torch.device("cpu"))
motion = torch.from_numpy(evaluator.mean).float()[None, None].expand(2, 300, -1).clone()
text, movement = evaluator.get_co_embeddings(
    ["A person walks forward.", "A person raises both arms."],
    motion, torch.tensor([120, 200], dtype=torch.long),
)
assert text.shape == movement.shape == (2, 512)
assert torch.isfinite(text).all() and torch.isfinite(movement).all()
print("PASS：CPU 离线加载与编码正常；合成输入不代表模型质量。")
PY
```

本次另外验证了全部 checkpoint state_dict tensor 有限、mean/std 都为272维有限数组且
std 全正。临时目录 `/tmp/motionmillion-evaluator-smoke-5wjjd0ty` 已删除；官方代码干净。

## 4. 正式自动评分还需要什么

本次 `asset_manifest.json` 是下载与加载验收单，**不是**
`fingerprint_motionmillion_evaluator.py` 生成的 `evaluator_identity.json`。后者必须绑定
真实资格集合，不能在没有 GT 数据时伪造。

后续复用已有实现，依次完成：

1. `prepare_motionmillion_official_eval.py`：从官方 val 来源归档准备真实272D动作、文本、
   split与 `eligibility.json`。本机目前只有mean/std，没有完整验证集视图。
2. `fingerprint_motionmillion_evaluator.py`：冻结代码、权重、数据资格和统计量身份。
3. `generate_motionmillion_val_predictions.py`：固定 GENMO checkpoint 和生成参数，
   为完整资格集合生成并转成272D，记录 prediction manifest。
4. `run_motionmillion_official_metrics.py`：调用 evaluator 计算每个 seed 的指标。
5. `summarize_motionmillion_metrics.py`：按现有协议汇总20个不同seed及95%置信区间。

完整接口见 [现有文本训练与评测说明](MOTIONMILLION_TEXT_ONLY.md#7-推理与评测)。
开始正式评测前还有两个已核实的准备点：

- 当前 `prepare_motionmillion_official_eval.py` 要求 raw-root 下
  `mean_std/vector_272/mean.npy`、`std.npy`，而真实 HF release 在 `mean_std/Mean.npy`、
  `Std.npy`。本次本地 evaluator 已按要求摆放统计量；尚未修改该准备脚本或服务器数据布局。
  后续运行准备脚本前须适配原始路径，不要直接复制旧命令宣称已经可跑完整评测。
- 固定版本官方 loader 过滤60–200帧，并按4帧单位随机裁剪；当前 GENMO v1 生成固定120帧，
  本仓库指标运行器读取 eligibility 原始长度。统一 evaluator 不等于整个论文协议相同。
  正式横向对比必须说明时长、裁剪、caption选择、测试集/验证集、候选batch和随机种子。

## 5. MotionMillion-Eval 人工评测是什么

它提供126条文本任务，由人观看生成动画评分，不是一个自动输出“平滑分/物理分”的模型。
论文表6的七组为：日常74、交流20、体育16、工作6、战斗6、艺术/舞蹈2、非人行为2。
官方三个维度各用1–4分，可用以下简写理解锚点：

| 分数 | 文本匹配 TA | 动作平滑 MS | 物理合理 PP |
|---|---|---|---|
| 4 | 全部细节符合 | 自然连续 | 符合物理 |
| 3 | 大体符合、细节偏差 | 局部小瑕疵 | 有不合理处，主体仍连贯 |
| 2 | 明显遗漏或偏离 | 明显断续 | 明显违背物理 |
| 1 | 核心动作不符 | 严重卡顿突变 | 严重失真、难理解 |

论文还报告三位专业标注者的模型两两比较，以胜/平/负统计。它与绝对打分是两个环节；
原文没有完整说明绝对评分的随机化、盲法、每条评审人数和汇总细节，不能自行填成官方事实。
PP原文还涉及物体交互、光照、阴影及碰撞，不能宣称单人SMPL渲染完整测到了这些因素。
依据：[论文5.3、表6及补充材料第9节](https://arxiv.org/html/2507.07095v1)。

## 6. 针对当前 GENMO 的建议设计（本仓库建议，非官方原样协议）

建议分别保留官方126条的结果，以及面向当前单人身体动作能力的简单任务结果；不要删除
困难prompt后仍称完整官方126条。之前的训练集200条中英文本不属于独立泛化测试集。

建议的评审流程：

1. 比较不同checkpoint时，使用相同prompt、seed集合、时长规则、FPS、DDIM、CFG、相机
   和后处理。简单动作可用120帧；连续多动作需预先规定足够时长，单列长序列结果，避免
   把要求十几秒的文本强压成4秒后只归因于语义失败。每条3个seed全保留，不挑最好一次。
2. 模型用A/B/C匿名标签，随机排列视频；至少3名评审独立评分，先用少量示例统一尺度。
   可附中文释义帮助理解，但英文条件和评分细节要核对一致。
3. 每段分别给TA、MS、PP三项分数，并记录错误标签和时间点；不要只填一个“好看分”。
   TA观察动作类型、方向、左右侧、次数和顺序；MS观察突变、抖动、卡顿、过渡；PP重点
   观察脚滑、穿地、无支撑漂浮、扭曲和不合理关节姿态。当前画面没有物体、手指/表情
   能力时，注明不可观察部分；这种面向身体动作的PP属于明确适配版。
4. 分别报告三项均值、每类别均值、低分比例与失败例子。可用按prompt聚类的bootstrap
   区间，并报告评审一致性；同一prompt的多个seed和多个评审不能全部当成独立样本。
   全局按条目均值会受74条日常动作主导，可同时给七类别等权均值，清楚标明算法。
5. 若比较两个checkpoint，可另做同prompt的胜/平/负盲评。不要把这项偏好统计混进三项
   绝对分；也不要把视觉上合理解释为真实接触力、动力学稳定或机器人安全验证。

示例（仅用于解释评分，不是真实模型结果）：prompt要求“向前走两步，然后蹲下”，
视频却一直自然站立。TA应很低，MS/PP仍可能很高；因此只看平滑或FID不足以确认听懂指令。

当前已有 [build_motionmillion_review.py](../tools/eval/build_motionmillion_review.py)，
为编号000–125的视频生成离线HTML，每条三项1–4分，浏览器保存并导出JSON。它目前没有
完整的多评审账号、匿名随机排序、自动汇总或一致性分析；这些是上面建议，未在本次实现。
localStorage键只包含checkpoint SHA，同checkpoint不同seed/协议不能在同一浏览器中
混着评分而不导出隔离。每名评审应保存独立的结果文件并注明评审标识和协议。

页面命令模板（须先生成全部126段视频；本次未生成它们）：

```bash
cd /home/weili/GENMO
.venv/bin/python tools/eval/build_motionmillion_review.py \
  --prompt-file inputs/MotionMillion/official_evaluator/code/assets/infer_batch_prompt.txt \
  --video-root <包含000.mp4到125.mp4的目录> \
  --checkpoint <参与评审的GENMO_checkpoint> \
  --output <正式评审目录>/index.html \
  --seed 42 --num-frames 120 --fps 30 --ddim-steps 50 --cfg-scale 2.5
```

该命令的120帧是当前工具的固定时长示例；不能代表已经解决长文本任务的时长公平性。
