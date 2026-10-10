"""当前 Stage1 s595000 部署使用的固定输入、输出和采样契约。

本模块是部署分支唯一的模型接口来源：batch=1、音乐/动作窗口120帧、动作30Hz、
本体历史50步/48维。浮点输入使用float32，时间步使用int64，所有有效性掩码使用bool。
TensorRT编译专用图将四个掩码编码为0/1 INT32以接入CUDA插件；调用端仍传bool。
模型图包含音乐/历史/前缀编码和音乐CFG，图外执行DDIM及qpos解码。
只支持当前Stage1十一输入网络，旧纯音乐五输入模型和旧TensorRT engine明确拒绝。
GMT接收的qpos28与50Hz完整缓存格式保持原协议；采样身份与网络身份分别记录。
"""

BUMI_ONNX_CONTRACT_VERSION = "genmo.bumi_closedloop.stage1_onnx.v1"
BUMI_ENGINE_CONTRACT = "genmo.bumi_closedloop.stage1_trt.v2"
BUMI_STAGE1_SAMPLING_CONTRACT_VERSION = "genmo.bumi_stage1.autoregressive_prefix12.qpos30.v1"
BUMI_STAGE1_PRECISION_POLICY = "fp32_tf32_disabled_v1"
BUMI_STAGE1_TRT_BUILD_POLICY = "cuda_mask_plugins_int32_v3"
WINDOW_FRAMES = 120
PREFIX_FRAMES = 12
STRIDE_FRAMES = WINDOW_FRAMES - PREFIX_FRAMES
HISTORY_STEPS = 50
MUSIC_DIM = 35
MOTION_DIM = 30
PROPRIO_DIM = 48
SOURCE_FPS = 30

BUMI_ONNX_INPUTS = {
    "noisy_motion": [1, 120, 30],
    "diffusion_timestep": [1],
    "music_features": [1, 120, 35],
    "music_valid": [1, 120],
    "proprio_history": [1, 50, 48],
    "proprio_history_valid": [1, 50],
    "history_relative_times": [1, 50],
    "known_qpos30": [1, 120, 30],
    "known_qpos30_mask": [1, 120, 30],
    "future_valid": [1, 120],
    "guidance_scale": [1],
}
BUMI_ONNX_INPUT_DTYPES = {
    name: ("int64" if name == "diffusion_timestep" else
           "bool" if name in {"music_valid", "proprio_history_valid",
                             "known_qpos30_mask", "future_valid"} else "float32")
    for name in BUMI_ONNX_INPUTS
}
BUMI_TRT_INPUT_DTYPES = {
    name: "int32" if dtype == "bool" else dtype
    for name, dtype in BUMI_ONNX_INPUT_DTYPES.items()
}
BUMI_ONNX_OUTPUTS = {
    "pred_motion": [1, 120, 30],
    "pred_foot_contact_logits": [1, 120, 2],
}
BUMI_ONNX_OUTPUT_DTYPES = {name: "float32" for name in BUMI_ONNX_OUTPUTS}


def validate_stage1_engine_build_options(value):
    """绑定CUDA插件、图改写源码和资源身份，拒绝之前失败的融合规避策略缓存。

    优化等级/workspace、插件源码/共享库SHA以及图改写源码SHA均进入缓存键。
    此函数只检查JSON字段，不导入TensorRT，不执行构建、加载共享库或推理。
    """
    fingerprints = {"plugin_source_sha256", "plugin_library_sha256", "lowering_source_sha256"}
    required = {"policy", "optimization_level", "workspace_bytes"} | fingerprints
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Stage1 engine缺少完整构建策略，请重新构建")
    if value["policy"] != BUMI_STAGE1_TRT_BUILD_POLICY:
        raise ValueError("Stage1 engine CUDA掩码插件构建策略不匹配，请重新构建")
    level = value["optimization_level"]
    workspace = value["workspace_bytes"]
    if type(level) is not int or not 0 <= level <= 5:
        raise ValueError("Stage1 engine优化等级必须为0..5整数")
    if type(workspace) is not int or workspace <= 0:
        raise ValueError("Stage1 engine workspace必须为正整数字节")
    for name in fingerprints:
        digest = value[name]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError(f"Stage1 engine构建指纹无效: {name}")
    return value
