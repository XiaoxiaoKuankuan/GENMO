# GEM 训练代码框架说明

本文档帮助快速理解 `python scripts/train.py exp=gem_smpl_regression` 和
`python scripts/train.py exp=gem_smpl` 这两条训练命令背后的代码结构。

## 训练总流程

1. `scripts/train.py` 作为训练入口，读取 `configs/train.yaml`。
2. Hydra 根据 `exp=gem_smpl_regression` 或 `exp=gem_smpl` 组合实验配置。
3. 训练入口实例化 `data`、`model`、callbacks、logger。
4. `gem.datamodule.mocap_trainX_testY.DataModule` 创建训练和验证 DataLoader。
5. `gem.gem.GEM` 在 `training_step` 中准备 batch，并按训练模式调用 pipeline。
6. `gem.pipeline.gem_pipeline.Pipeline` 调用网络前向、解码预测结果、计算训练损失。
7. PyTorch Lightning `Trainer.fit` 负责循环、反传、优化器 step、验证和 callback 调用。

## 两个主要训练实验

| 实验 | 配置文件 | 主要用途 |
| --- | --- | --- |
| `gem_smpl_regression` | `configs/exp/gem_smpl_regression.yaml` | 只训练视频/2D/相机条件到 SMPL motion feature 的回归路径。 |
| `gem_smpl` | `configs/exp/gem_smpl.yaml` | 同时训练 regression 与 diffusion，支持文本、音乐、音频等多模态条件生成。 |

## 配置文件职责

| 文件 | 功能 |
| --- | --- |
| `configs/train.yaml` | 训练总配置入口，定义 defaults、输出目录、resume/checkpoint、Lightning Trainer 和 logger 默认值。 |
| `configs/hydra/default.yaml` | 控制 Hydra 输出目录和日志文件位置。 |
| `configs/data/mocap/trainX_testY.yaml` | 指定训练 DataModule、训练集、验证集和 DataLoader 参数。 |
| `configs/data/collate_cfg/default.yaml` | 定义 batch 拼接时缺失字段的默认 shape/value。 |
| `configs/model/gem.yaml` | 指向 `gem.gem.GEM`，并传入 pipeline、optimizer、scheduler 和模型级训练选项。 |
| `configs/pipeline/regression_only.yaml` | 回归实验使用，只训练 regression 路径。 |
| `configs/pipeline/dual_mode.yaml` | 完整实验使用，同时训练 regression 和 diffusion，并定义 loss 权重。 |
| `configs/network/diffusion.yaml` | 标准尺寸 GEMDiffusion 网络，回归实验默认使用。 |
| `configs/network/diffusion_lg.yaml` | 大尺寸 GEMDiffusion 网络，完整实验默认使用。 |
| `configs/endecoder/v1_smpl_amass_bedlam.yaml` | 定义 151 维 motion feature 的编码/解码方式和统计量。 |
| `configs/diffusion/ddim.yaml` | 定义扩散噪声日程、采样步数、DDIM 和 guidance 参数。 |
| `configs/optimizer/adamw_2e-4.yaml` | AdamW 优化器配置。 |
| `configs/scheduler/epoch_half_200_350.yaml` | 回归实验默认学习率调度。 |
| `configs/scheduler/epoch_half_500_750.yaml` | 完整实验默认学习率调度。 |

## 训练入口和核心模块

| 文件 | 功能 |
| --- | --- |
| `scripts/train.py` | 训练入口。实例化模型/数据/callback/logger，处理 checkpoint 和 resume，最后调用 `Trainer.fit` 或 `Trainer.test`。 |
| `gem/datamodule/mocap_trainX_testY.py` | LightningDataModule。训练集使用 `ConcatDataset`，验证/测试集使用 `CombinedLoader` 顺序遍历。 |
| `gem/gem.py` | LightningModule。负责 batch 预处理、条件特征构造、训练/验证 step、日志、优化器和 checkpoint 兼容。 |
| `gem/pipeline/gem_pipeline.py` | 模型前向流水线。连接网络、EnDecoder 和 loss 计算。 |
| `gem/network/gem_diffusion.py` | 扩散/回归网络外壳。创建 train/test diffusion 对象并调用 denoiser。 |
| `gem/network/gem_denoiser.py` | RoPE Transformer denoiser。融合时序 motion feature 与多模态条件。 |
| `gem/network/endecoder.py` | 在 SMPL 参数和 151 维归一化 motion feature 之间编码/解码。 |
| `gem/network/gem_cfg_sampler.py` | 推理采样时的 classifier-free guidance 包装器。 |
| `gem/network/stats_compose.py` | 保存训练特征归一化所需的均值和标准差。 |

## 扩散工具文件

| 文件 | 功能 |
| --- | --- |
| `gem/diffusion_utils/model_util.py` | 根据配置创建 `SpacedDiffusion`。 |
| `gem/diffusion_utils/gaussian_diffusion.py` | 扩散训练 loss、加噪、去噪、DDIM/采样核心逻辑。 |
| `gem/diffusion_utils/respace.py` | 把完整 diffusion 时间步压缩成推理用的稀疏时间步。 |
| `gem/diffusion_utils/resample.py` | 训练时采样 diffusion timestep。 |
| `gem/diffusion_utils/losses.py` | KL、Gaussian log-likelihood 等扩散概率损失工具。 |
| `gem/diffusion_utils/nn.py` | diffusion loss 中用到的张量求和/平均工具。 |

## 训练数据集

| 配置文件 | Python 类 | 作用 |
| --- | --- | --- |
| `configs/train_datasets/amass_v11.yaml` | `gem.datasets.pure_motion.amass.AmassDataset` | 纯 SMPL 动作数据，提供人体运动先验。 |
| `configs/train_datasets/bedlam_v2.yaml` | `gem.datasets.bedlam.bedlam.BedlamDataset` | 合成视频数据，提供图像特征、相机和人体监督。 |
| `configs/train_datasets/h36m_v1.yaml` | `gem.datasets.h36m.h36m.H36MDataset` | Human3.6M mocap 视频监督。 |
| `configs/train_datasets/3dpw_v1.yaml` | `gem.datasets.threedpw.threedpw_motion_train.ThreedpwSmplDataset` | 真实户外视频训练数据。 |
| `configs/train_datasets/3dpw_occ_v1.yaml` | `gem.datasets.threedpw.threedpw_occ_motion_train.ThreedpwOccSmplDataset` | 遮挡场景训练数据。 |
| `configs/train_datasets/aistpp_train.yaml` | `gem.datasets.aistpp.aistplusplus.AISTPlusPlusSmplDataset` | 音乐舞蹈动作数据。 |
| `configs/train_datasets/beat2_static_train.yaml` | `gem.datasets.beat2.beat2.BEAT2SmplDataset` | 语音/音频动作数据。 |
| `configs/train_datasets/humanml3d_static_train.yaml` | `gem.datasets.pure_motion.humanml3d.Humanml3dDataset` | 文本描述到动作的数据。 |

## 验证数据集和指标

训练过程中 Lightning 会按 `val_check_interval` 周期跑验证集。验证集配置来自
`configs/test_datasets/`，指标由 callback 计算。

| 数据/指标 | 相关文件 |
| --- | --- |
| EMDB split 1/2 | `gem/datasets/emdb/emdb_motion_test.py`，`gem/callbacks/metric/metric_emdb.py` |
| 3DPW | `gem/datasets/threedpw/threedpw_motion_test.py`，`gem/callbacks/metric/metric_3dpw.py` |
| 3DPW-OCC | `gem/datasets/threedpw/threedpw_occ_motion_test.py`，`gem/callbacks/metric/metric_3dpw_occ.py` |
| RICH | `gem/datasets/rich/rich_motion_test.py`，`gem/callbacks/metric/metric_rich.py` |
| 文本动作可视化 | `gem/callbacks/vis/vis_text.py` |

## 训练 callback

| 配置文件 | Python 类 | 作用 |
| --- | --- | --- |
| `configs/callbacks/ckpt_saver/every10000s_top100.yaml` | `gem.callbacks.simple_ckpt_saver.SimpleCkptSaver` | 周期性保存 checkpoint。 |
| `configs/callbacks/prog_bar/prog_reporter_ed1.yaml` | `gem.callbacks.prog_bar.ProgressReporter` | 显示训练进度、损失和耗时。 |
| `configs/callbacks/train_speed_timer/base.yaml` | `gem.callbacks.train_speed_timer.TrainSpeedTimer` | 记录数据等待时间和单 batch 训练耗时。 |
| `configs/callbacks/lr_monitor/pl.yaml` | `pytorch_lightning.callbacks.lr_monitor.LearningRateMonitor` | 记录学习率。 |
| `gem/callbacks/autoresume_callback.py` | `AutoResumeCallback` | 在特定运行环境下支持自动恢复训练。 |

## 阅读建议

建议按下面顺序读代码：

1. 先看 `configs/exp/gem_smpl_regression.yaml` 或 `configs/exp/gem_smpl.yaml`，确认本次实验用了哪些组件。
2. 再看 `scripts/train.py`，理解配置如何变成模型、数据和 Trainer。
3. 看 `gem/datamodule/mocap_trainX_testY.py`，理解 batch 字段从哪里来。
4. 看 `gem/gem.py` 的 `training_step` 和 `prepare_batch`，理解 batch 如何变成网络输入。
5. 看 `gem/pipeline/gem_pipeline.py`，理解前向输出和 loss。
6. 最后看 `gem/network/gem_diffusion.py`、`gem/network/gem_denoiser.py` 和 `gem/network/endecoder.py`，理解核心网络与 motion feature 表达。
