# Changelog

## 2026-08-18
- 新增 `scripts/retarget_smplx_to_bumi3_capture.py`：通过指定的 GMR-CPP
  SMPL-X→BUMI3 配置执行真实 SMP1/C++ IK 重定向，并在播放期间持续读取 Redis
  stream，避免长于 512 帧的动作因 stream 淘汰策略而丢失前半段；同时增加完整
  时间轴覆盖、帧数、有限值和四元数校验，输出可复现的 qpos、原始捕获与配置哈希。

## v1.0.0 — 2025-10-15
- Initial public release (ICCV 2025 Highlight)

## v1.1.0 — 2026-03-16
- Improve documentations
- Add multi-modal conditioning
