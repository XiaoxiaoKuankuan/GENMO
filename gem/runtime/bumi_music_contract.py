"""BUMI 音乐导出模型与部署运行时共用的固定接口。

这里只声明已经发布的 qpos30/contact 单步去噪协议，不导入网络构造器、训练框架或
checkpoint 加载工具。训练仓库负责生成模型，独立部署目录只需共享这些接口标识。
修改协议版本必须同时验证导出图、配套资产和在线消费者，不能靠文件名推断兼容性。
"""

BUMI_ONNX_CONTRACT_VERSION = "genmo.bumi_music_guided_denoiser_step.qpos30_contact.v3"
BUMI_ONNX_INPUTS = {
    "noisy_motion": [1, 120, 30],
    "diffusion_timestep": [1],
    "music": [1, 120, 35],
    "length": [1],
    "guidance_scale": [1],
}
BUMI_ONNX_OUTPUTS = {
    "pred_motion": [1, 120, 30],
    "pred_foot_contact_logits": [1, 120, 2],
}
