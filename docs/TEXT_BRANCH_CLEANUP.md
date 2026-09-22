# BUMI 文本分支整理与兼容边界

本次整理从 `feature/bumi-text-only` 的 `23fd802` 开始。当前入口见根目录 README；
本页说明删除范围、公共依赖位置与验证依据，不作为旧功能的运行教程。

## 跨分支退役范围

- 删除旧 `482138` 运动学、脚底代理点，以及旧 GMR/SONIC 质量规则默认文件。
- 删除 `bumi_93d*`、`diffusion_lg_bumi93*`、旧 93D pipeline 配置和
  `compute_bumi_93d_stats.py`。不把旧 93D checkpoint 改名解释为 qpos30。
- main 与两个 SMPL 分支同步删除其旧 93D 编解码、损失、模型及专属数据消费者，
  修正包导出；SMPL 模型和不依赖学习表示的 BUMI 运动学继续保留。
- 音乐分支删除 6 个实际合成为 93D 的旧实验；保留的 5 个 qpos30 实验展开旧父配置，
  Hydra 合成内容与整理前逐项一致。保留的数据筛选工具改为必须显式传入质量配置，
  避免自动选择已退役阈值；外部已交付数据中的质量规则快照不被改写。
- 部署分支原本没有这批文件；没有为制造差异而更改部署代码，也未修改原有 deployment.ini。

## 文本分支公共模块

| 职责 | 当前位置 | 原来的依赖 |
|---|---|---|
| 区间、速度、限位及姿态统计 | `gem/robots/bumi/motion_quality.py` | SONIC筛选CLI、旧质量筛选模块 |
| SLERP与线性重采样 | `gem/utils/motion_resampling.py` | AIOZ音乐数据工具 |
| NumPy pickle路径兼容 | `gem/utils/pickle_compat.py` | 旧GMR动作读取器 |
| T5常驻编码 | `gem/runtime/text_encoding.py` | SMPL常驻文本服务 |
| 文本特征与padding mask校验 | `gem/runtime/text_condition.py` | GEM人体模型模块 |
| TensorRT版本、GPU与执行上下文 | `gem/runtime/tensorrt_core.py` | 音乐滑窗运行时 |
| Lightning优化、日志与checkpoint | `gem/text_training.py` | GEM人体/视觉基类 |
| 机器人监督与验证 | `gem/bumi_gem.py` | 原继承链中的音乐适配逻辑 |
| qpos30/FK扩散管线 | `gem/pipeline/bumi_pipeline.py` | `bumi_music_pipeline.py` |
| 分片文本采样 | `gem/datamodule/sequence_sampler.py` | 按MotionMillion命名的采样模块 |
| 源文本身份与旧T5特征只读契约 | `gem/datasets/text_source_contract.py` | SMPL构建器及272D转换公共模块 |

当前继承链为 `BumiTextGEM → BumiGEM → TextTrainingBase`。pipeline、endecoder、
denoiser属性名及模型参数键保持不变，训练不导入原 `gem.gem`、SMPL资产、音乐服务或SONIC CLI。

旧MotionMillion预计算T5的来源校验仍保留，这是现有文本数据兼容接口，不能把它与
已删除的人体动作构建器混为一谈。full300文本checkpoint仍按自身序列契约读取，
当前crop120配置不能跨契约完整恢复它。数据筛选的代码指纹随模块位置变化，旧报告
保持原内容；跨代码版本继续同一筛选任务会按既有身份检查拒绝，需启动新任务。

## 当前入口与保留内容

文本分支只保留默认训练实际合成的24个YAML、当前UMR质量规则、fe934运动学和代理点
JSON及独立部署INI。保留训练前DDP检查、资产导出/FK验证、文本评测、渲染和部署打包工具。
音乐、SMPL、HMR/视频、多模态、旧人体评测与相应构建器已从本分支移除。

网页入口改为 `scripts/demo/demo_bumi_text_web.py` 与 `share_bumi_text_web.py`；
模型发现和worker只接受带真实 `bumi_text_contract` 的BUMI模型。默认候选路径是
`inputs/checkpoints/bumi_text/model.ckpt`，不自动下载或伪造checkpoint。

`docs/archive/` 保存旧说明并标注不适用于当前入口。`tests/` 中历史源码没有删除；
默认测试集合由 `pytest.ini` 明确列出，现行测试需要的人工源数据夹具独立保存。
部署安装配置、容器启动器、当前 fe934 XML/22个mesh、四库审计报告、既有评测证据
和训练相关产物均按各自用途保留。

## 验证依据

- 当前文本测试包含数据拒绝路径、crop120和历史full300、真实CPU微型Transformer
  前向/反向、Lightning训练与完整恢复、在线T5替身、ONNX数值对照、可搬运部署包、
  网页和依赖隔离；实际数量与最终结果见根目录记录文本。
- 固定随机种子，对整理前后微型网络进行对照：84个checkpoint参数张量、80个梯度
  张量与loss逐元素完全一致；新代码严格加载整理前保存的checkpoint。
- 音乐分支5个qpos30实验合成配置逐项一致，表示与根高测试22项通过。
- main、两个SMPL分支和音乐分支的显式SMPL实验可以合成，SMPL数据包与BUMI运动学包可导入。
  这些分支原有 `exp=mixed` 缺省值不存在的问题未在本次范围内修改，运行时仍需选择实际实验。
- 各分支保留源码语法检查与已删除模块的静态引用检查通过；文本修改文件的Ruff错误检查通过。
- 当前显示资产和3份历史评测报告与整理前字节一致；四库报告自带SHA256SUMS的11项全部通过。

验证不包含完整数据重建、正式训练、真实T5下载、GPU TensorRT引擎执行、容器构建或机器人
动力学验收。测试文件仅写入独立系统临时目录，运行后删除；未同步或重启训练服务器作业。
