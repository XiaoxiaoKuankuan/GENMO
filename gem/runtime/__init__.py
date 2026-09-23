# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""现行 GENMO 运行时公共接口。"""

from .resident_multimodal_motion import (
    MultimodalMotionRequest,
    ResidentMultimodalMotionEngine,
    UnsupportedModeError,
    build_text_music_data,
)
from .resident_text_motion import (
    ResidentTextMotionEngine,
    TextMotionRequest,
    encode_prompt_with_loaded_t5,
    get_cuda_memory_snapshot,
)
from .resident_video_session import (
    ResidentVideoModelStack,
    ResidentVideoSession,
    VideoSourceSession,
)
from .smpl_frame import SMPLFrame

__all__ = [
    "MultimodalMotionRequest",
    "ResidentMultimodalMotionEngine",
    "ResidentTextMotionEngine",
    "ResidentVideoModelStack",
    "ResidentVideoSession",
    "SMPLFrame",
    "TextMotionRequest",
    "UnsupportedModeError",
    "VideoSourceSession",
    "build_text_music_data",
    "encode_prompt_with_loaded_t5",
    "get_cuda_memory_snapshot",
]
