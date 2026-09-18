# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""常驻推理接口的延迟导入入口。

保留原有from gem.runtime import接口；仅加载调用者实际使用的后端，避免纯BUMI文本
部署被SMPL、视频或其他训练依赖拖入。直接导入子模块同样不会启动无关模型。
"""

from importlib import import_module

_EXPORTS = {
    "MotionSourceMux": "motion_source_mux",
    "MuxState": "motion_source_mux",
    "MuxTick": "motion_source_mux",
    "MonotonicDeadline": "motion_streamer",
    "MotionPlayer": "motion_streamer",
    "MotionQueue": "motion_streamer",
    "MotionWatcher": "motion_streamer",
    "PlayerState": "motion_streamer",
    "SMPLFrame": "motion_streamer",
    "SMPLMotion": "motion_streamer",
    "align_motion_root_yaw": "motion_streamer",
    "align_motion_to_frame": "motion_streamer",
    "interpolate_axis_angle": "motion_streamer",
    "interpolate_frames": "motion_streamer",
    "load_smpl_motion": "motion_streamer",
    "sample_motion_at": "motion_streamer",
    "synthetic_idle_motion": "motion_streamer",
    "MultimodalMotionRequest": "resident_multimodal_motion",
    "ResidentMultimodalMotionEngine": "resident_multimodal_motion",
    "UnsupportedModeError": "resident_multimodal_motion",
    "build_text_music_data": "resident_multimodal_motion",
    "ResidentTextMotionEngine": "resident_text_motion",
    "TextMotionRequest": "resident_text_motion",
    "encode_prompt_with_loaded_t5": "resident_text_motion",
    "get_cuda_memory_snapshot": "resident_text_motion",
    "ResidentVideoModelStack": "resident_video_session",
    "ResidentVideoSession": "resident_video_session",
    "VideoSourceSession": "resident_video_session",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module("." + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
