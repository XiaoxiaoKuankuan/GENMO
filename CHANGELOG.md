# Changelog

## Unreleased

- 将 BUMI 音乐主链统一为 `genmo.bumi_motion_features.qpos30.v3`：模型输出 30 维 qpos
  与 2 维足接触 logits，link 几何统一由 qpos FK 计算，现行物理损失使用 v5 pipeline。
  当前 checkpoint、stats、ONNX/TensorRT 和 fe934 运动学资产必须成套匹配，不能与历史
  93D/482138 资产混用。
- 将旧 93D/482138 方法、质量筛选和部署说明移入 `docs/archive/legacy_93d/`，保留历史正文
  供审计，但明确禁止把其中的旧配置、资产路径和命令直接用于当前 qpos30/v5 链路。
- 冻结正式 v5 scratch s350000 的 Hydra 源配置，要求 `configs/train.yaml` 显式选择实验，
  并将 demo、checkpoint 选择、ONNX 导出及 ONNX/TensorRT parity CLI 默认值统一到该配置。
- 提取中性 motion、quality、dataset publish 与 qpos 数值 helper，使 UMR、
  robot_retargeter、CSV 和 transfer 当前 producer 不再反向依赖旧 GMR/legacy/SONIC 入口。
- 成组删除 GMR manual-q1、legacy pickle、SONIC 50 Hz、qpos30 v2 loss 旧生产链及对应测试；
  同时删除零引用配置和误跟踪运行日志，新增当前 producer 与 import closure 回归测试。

## 历史：BUMI 93D/482138 阶段

- 将 BUMI 93D 升级为 `heading-local ΔXY + (z-z_default)` 混合根运动表示，辅助 63D
  改为逐帧 root-relative 几何；长音乐先融合物理增量、绝对根高、根旋转和关节，再对
  完整时间轴只积分一次水平轨迹。stats/checkpoint/ONNX/TensorRT 合约同步升版并明确拒绝
  旧 s430000，防止同维度特征被静默误解。
- Add a source-MJCF-bound BUMI3 legacy motion quality pipeline with strict NumPy pickle/
  quaternion/joint-order contracts, OMG-style temporal anomaly statistics, sustained floor-style
  rejection, auditable reports, safe PASS-only materialization, full-length review rendering, and
  synthetic regression tests.
- Clarify that ``legacy`` denotes the GMR pickle format rather than an old robot revision, bind the
  plan to the user-confirmed production ``bumi3`` asset, and document generation-chain provenance,
  MuJoCo FK parity, review, and formal GENMO conversion phases.

## 2026-08-18
- 新增 `scripts/export_smplx_to_bumi3_offline_npz.py`：将 SMPL-X 动作先以
  SO(3) SLERP 重采样到 50 Hz，再逐帧调用无 Redis 的 GMR-CPP 同步 batch
  server，按 Isaac-Lab 关节顺序导出 SONIC 部署所需的七字段 BUMI3 NPZ；同时
  校验精确 float32 `fps=50.0`、字段形状、有限值与 wxyz 单位四元数，并把配置
  哈希、关节/body 名称和求解耗时放入独立 metadata 文件。
- 调整 `server_music_wav_4set_10_20260818_videos/index.html` 的结果展示与筛选顺序
  为 FineDance、CoMPAS3D、AIOZ-GDance、AIST++，便于按指定数据集顺序查看。
- 新增 `outputs/server_music_wav_4set_10_20260818_videos/index.html`：为 40 个音乐
  生成结果及其 GMR BUMI3 重定向视频提供离线索引页，支持数据集筛选、名称搜索、
  成对同步播放、统一暂停、视口懒加载和直接下载；全部资源使用可移动的相对路径。
- 新增 `scripts/retarget_smplx_to_bumi3_capture.py`：通过指定的 GMR-CPP
  SMPL-X→BUMI3 配置执行真实 SMP1/C++ IK 重定向，并在播放期间持续读取 Redis
  stream，避免长于 512 帧的动作因 stream 淘汰策略而丢失前半段；同时增加完整
  时间轴覆盖、帧数、有限值和四元数校验，输出可复现的 qpos、原始捕获与配置哈希。

## v1.0.0 — 2025-10-15
- Initial public release (ICCV 2025 Highlight)

## v1.1.0 — 2026-03-16
- Improve documentations
- Add multi-modal conditioning
