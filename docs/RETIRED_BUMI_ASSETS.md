# 已退役的 BUMI 资产、表示与生产入口

本文记录 `feature/bumi-music-closedloop` 分支在 2026-09-22 完成的仓库自洽性清理。
当前唯一正式音乐生成主链是 `genmo.bumi_motion_features.qpos30.v3`、独立 2D contact
head、fe934 运动学资产和 `physical_qpos30_contact_v5`；旧 93D/482138、GMR
manual-q1、legacy pickle 与 SONIC 50 Hz 生产入口不再属于当前工作树。

## 当前可复现入口

- 正式 v5 scratch s350000 配置：
  `configs/exp/gem_bumi_music_only_5set_robot_retargeter_pass_v2_qpos30_contact_v5_scratch_s350000.yaml`
- 当前 30 Hz 筛选规则：
  `configs/bumi/quality_filter_robot_retargeter_30hz_v1.yaml`
- 当前自建 CSV 规则：
  `configs/bumi/quality_filter_csv_mine_robot_retargeter_fe934_v2.yaml`
- 当前 fe934 运动学与足底代理：
  `configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json`、
  `configs/bumi/sole_proxies_robot_retargeter_fe934_v1.json`

训练、demo、checkpoint 选择、ONNX 导出以及 ONNX/TensorRT parity CLI 的默认实验均指向
上述 frozen v5 s350000 配置。`configs/train.yaml` 不再提供无效的隐式实验，调用者必须
显式选择 `exp=...`。

## 已从版本库移除

以下内容只能从对应 Git 历史提交复现，不应在当前分支重新引入：

- 旧 93D/482138 配置、资产描述、统计入口与失效测试；
- GMR manual-q1 v3 质量配置、选择 producer 和两份 qpos30 v2 loss 实验；
- legacy pickle reader、质量筛选、dataset builder 和 renderer；
- SONIC 七字段 50 Hz NPZ filter、dataset builder、锁定 50 Hz 的 SMPL-X 离线导出 CLI、
  旧 HQ 原动作对比 CLI；
- 对应的 legacy/SONIC 测试和操作文档；
- 零引用的 qpos30 pipeline v2/v3/v4、旧 dataset/eval、scheduler、callback 等配置；
- 被 Git 误跟踪的 `logs/resident_music_s050000.log`。

四份 93D/482138 历史说明已迁入
[`docs/archive/legacy_93d/`](archive/legacy_93d/README.md)。归档文档保留历史正文供审计，
但其中的 checkpoint、stats、配置、资产路径和命令不得在当前分支直接执行。

## 当前共享 helper 边界

为避免现行 producer 反向依赖已退役入口，公共能力已迁入中性模块：

- `gem/robots/bumi/motion_utils.py`：SHA256、可信本地 NumPy pickle 兼容读取、root-tilt；
- `gem/robots/bumi/quality_common.py`：质量枚举和左闭右开区间；
- `tools/data/bumi/dataset_publish_utils.py`：manifest、配对、路径校验和原子发布基础；
- `tools/data/bumi/qpos_resample_utils.py`：连续四元数、SLERP 和 body-origin 落地；
- `tools/data/bumi/npz_quality_utils.py`：当前 NPZ 质量评估、汇总和报告写出。

UMR、robot_retargeter、CSV 和 transfer filelist 当前生产链直接依赖这些模块。回归测试同时
检查 import closure，禁止它们重新指向 legacy/SONIC producer。

## 本地运行产物

本机 `/home/weili/GENMO` 下 10 个旧 93D/482138 模型、ONNX、TensorRT、视频和分析目录
已按精确绝对路径删除，共 1,200 个文件、`10,541,691,826` bytes（约 9.82 GiB）。删除前
身份、占用、Git 状态、符号链接、硬链接和现行资产隔离证据见
[`LOCAL_ARTIFACT_RETIREMENT_20260922.md`](archive/legacy_93d/LOCAL_ARTIFACT_RETIREMENT_20260922.md)。

本次明确保留：

- 正式 qpos30/v5 s350000 checkpoint bundle 及匹配的 stats、fe934 kinematics；
- 当前 qpos30/v5 ONNX、TensorRT 和部署 metadata；
- `data/motions`、`data/motions_npz_bumi3_smooth_q1` 等历史源数据。

源数据是否继续归档是独立的数据治理决策，不能与旧生产代码或模型运行产物退役混为一谈。

## 验证边界

本次清理验证代码导入、Hydra 合成、现行 producer 回归、CLI 默认值、文档链接和全仓
pytest collection。它不重新证明 checkpoint 训练质量、ONNX/TensorRT 数值一致性、GMT
动力学稳定性、部署状态或实机安全；这些必须使用保留的 qpos30/v5 成套资产单独验收。
