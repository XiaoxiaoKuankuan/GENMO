# Changelog

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
