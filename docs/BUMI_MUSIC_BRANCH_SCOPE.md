# BUMI 音乐分支维护范围

适用分支：`feature/bumi-music-only`。2026-09-23 按用户授权收敛入口、实验和测试，
保留音乐 → qpos30/contact2 → BUMI/GMT 主链，不向其他分支传播本次删除。
清理前提交：`270cbb3b54a188f5fb64424e0e282c3b659e3d55`；历史源码仍可从 Git 恢复。

## 1. 默认实验和保留配置

默认实验统一为 `gem_bumi_music_only_umr70_mine_scratch_350k`：

- `configs/train.yaml`：替换不存在的默认 `mixed`。
- `scripts/demo/demo_music_bumi.py`。
- `tools/export/export_bumi_music_onnx.py`。
- `tools/eval/select_bumi_checkpoints.py`。
- `tools/eval/validate_bumi_music_onnx.py`。
- `tools/eval/validate_bumi_music_tensorrt.py`。

后五项的 `--exp` 不再指向已退役的 manual_q1 50k 实验。
另保留 `gem_bumi_music_only_5set_robot_retargeter_pass_v2_qpos30_contact_latest`，
用于显式指定 checkpoint 的五库 v5 兼容训练。两个实验的 YAML 本体未修改；分别显式
compose 后，完整 Hydra 组合与清理前逐项相同。改变的是缺省选择，不是训练参数。

`tests/bumi/test_bumi_config_contract.py` 改为覆盖当前四来源、120 帧 EDGE35、
qpos30/contact2、v5、scratch 350k、8×256 和统计量指纹要求；保留五库权重续训验证，
新增五个工具默认 `--exp` 回归。配置测试只使用临时路径，不伪造正式 stats。

## 2. 删除的 12 个实验

以下路径均相对于 `configs/exp/`：

```text
gem_bumi_music_only_5set_manual_q1_v3_qpos30_contact_50k.yaml
gem_bumi_music_only_5set_manual_q1_v3_qpos30_contact_scratch_350k.yaml
gem_bumi_music_only_5set_manual_q1_zup_v2_qpos30_contact_scratch_350k.yaml
gem_smpl.yaml
gem_smpl_motionxpp.yaml
gem_smpl_music_only.yaml
gem_smpl_music_only_4set.yaml
gem_smpl_music_only_4set_curated.yaml
gem_smpl_music_only_4set_physics_v1.yaml
gem_smpl_music_only_4set_physics_v2.yaml
gem_smpl_regression.yaml
gem_smpl_server.yaml
```

同时删除旧 `configs/demo.yaml` 和两份 manual_q1/GMR 专用质量规则；不替换当前
robot_retargeter/UMR 的显式质量配置，不修改生产数据中冻结的规则及其 SHA。

## 3. 删除与保留的入口

| 区域 | 已删除 | 保留范围 |
|---|---|---|
| `scripts/demo/` | SMPL 视频/文本/Webcam、多模态服务与客户端、SMPL→GMR/GMT、旧 ONNX runner、硬编码 s200000 Shell 包装 | README 列出的 9 个 BUMI 音乐入口，包括两个 buffered Python 入口 |
| `gem/runtime/` | 人体/文本/视频/多模态常驻模块、旧 source mux、SMPL robot stream、GMR viewer/streamer | BUMI resident/stream/protocol/GMT、共享 DDIM/TensorRT、独立定时器 |
| `gem/` 旧桥 | `gmr_udp_bridge.py`、`smplx_gmr_reference.py` | BUMI qpos/GMT 直连链 |
| `tools/export/` | 人体 denoiser、HMR2、ViTPose、SMPL music ONNX/TRT | BUMI 导出、engine 构建、打包和部署分支准备 |
| `tools/eval/` | SMPL music 评测、旧 legacy BUMI 渲染及旧模型固定批评测包装 | 当前 BUMI 评测、渲染、checkpoint 选择、ONNX/TRT/部署验证 |
| 专用数据 CLI | BEAT2、HumanML3D、Motion-X++ 构建/预检/T5 提取，旧 GMR manual selected-root/legacy 筛选入口 | 当前音乐源数据转换、审核、打包，UMR/robot_retargeter/Mine CSV 生产和验证 |
| 其他 | 旧 SMPL→BUMI 离线捕获/导出、SMPL idle pose、Webcam ONNX 上传、SMPL preflight/benchmark、旧四库人体比较脚本 | BUMI 完整音乐验收、原动作对比、训练和通用安装入口 |

`tools/data/bumi/build_bumi_music_dataset.py` 原来混有旧 pickle/482138 整库转换器。
现已删除 `convert_datasets`、旧 selection-info 逻辑、`main` 和固定资产 SHA，
保留 16 个当前生产者仍使用的配对/索引/校验/文件物化函数及原导入路径；函数 AST
与清理前一致。它现在是共享模块，不再是旧 GMR CLI。

## 4. 混合文件中的共享能力如何保留

| 旧位置或混合能力 | 当前维护位置 | 覆盖内容 |
|---|---|---|
| `motion_streamer` 中的定时器 | `gem/runtime/playback_timing.py` + `tests/test_playback_timing.py` | 单调时钟截止时间与落后丢帧；BUMI 离线播放器改用该模块 |
| `demo_utils` 中的数据审核渲染 | `tools/data/music_dance/render_utils.py` | 原 `render_global_frames`；AIOZ/FineDance/CoMPAS3D 三个验证器更新导入 |
| 旧 TRT streaming 混合测试 | `tests/bumi/test_bumi_ddim_sliding.py` | 30D 滑窗覆盖、尾部 padding、已知坐标覆盖、固定噪声和 CPU RNG；不保留 SMPL world-rollout 测试 |
| 旧 music specialist 混合测试 | `tests/bumi/test_bumi_music_conditions.py` | EDGE35、时间对齐、CFG 无文本路径、当前 BUMI condition dropout、qpos30/contact2 双头 CPU 前反向 |
| 旧 GMT trajectory 混合测试 | `tests/bumi/test_bumi_gmt_trajectory.py` | 30→50、速度、关节重排、21×52 窗口、CRC/ACK；增量计划直接测试当前 BUMI builder |
| 旧 quality filter 混合测试 | `tests/bumi/test_quality_intervals.py` | 半开区间和坏帧 halo 的通用规则 |
| SONIC 数据构建器中的共用测试 | `tests/bumi/test_qpos_data_utils.py` | 具名关节重排、SLERP、帧数/末帧保持、地面规范和 AIST 变体 |
| 名为 v1 的物理损失测试 | `tests/bumi/test_bumi_losses.py` | 保留全部既有 v2/v3/v4/v5 数学回归，文件名不再误导为仅旧版 |

原 `test_music_dance_curation.py` 只删除依赖已退役 SMPL 实验的配置用例，源数据审核
测试继续保留。对应源数据验证器删除 `--loader-smoke` 与 151D SMPL 编码烟测，
报告明确标为 `validation_scope=source_motion_music_only`，不再暗中 compose 已删除实验。
`test_bumi_buffered_deployment.py` 只删除两个旧 Shell 包装的固定
资产测试，缓存协议、状态推进与可选 C++/CUDA 集成测试保留。

## 5. 为什么还会出现 SMPL、GMR、SONIC 字样

- 音乐数据的权威来源、人工评分和 SMPL-X 打包仍服务于当前 BUMI 数据生产，相关测试保留。
- `build_bumi_music_dataset_from_sonic_npz.py`、`filter_sonic_npz_motions.py`、
  `legacy_motion.py`、`quality_filter.py` 仍含 UMR/CSV/robot_retargeter 或对比评估
  实际复用的函数；不能按文件名整块删除。旧数据显式兼容路径不升级为当前默认路径。
- 当前模型仍继承 GEM，并复用网络、数学、采样和部分 checkpoint/loss 兼容实现；
  本轮不删除底层父类、不改变 Transformer 或 checkpoint 契约，也不重构全部底层 YAML。
- `tools/data/motionxpp/common.py` 及其包标记作为仍存在的数据类导入依赖保留，
  Motion-X++ 的专用可执行工具和测试已移除。
- 保留有日期的历史技术文档并加退役提示，根 README 已改为当前音乐入口导航。
  历史记录里的旧路径不是现行默认值，不自动代表当前模型、训练状态或性能。

## 6. 验证与未执行事项

本次对整个保留的 `tests/` 运行 CPU/本地回归，实际结果、跳过理由、静态检查和临时
目录清理记录见根 `记录文本.md` 的本次条目。另核验保留实验组合与 13 个当前 CLI 帮助入口。

未执行正式训练、生产数据全量转换、正式统计量计算、真实 checkpoint 生成、GPU 引擎
性能对照、GMT/Isaac Lab/ROS 启动或实机动作；没有登录或同步训练服务器。
没有把静态测试结果当作生成质量或动力学验证，没有修改其他工作树或其未提交内容。
