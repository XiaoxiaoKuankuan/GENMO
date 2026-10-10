# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""当前Stage1网络的ONNX/TensorRT单步运行器与因果前缀舞蹈生成器。

两个后端接收同一十一输入字典，严格保持float32/int64/bool及固定形状。
TensorRT持久缓冲把四个bool掩码转为0/1 INT32，并在反序列化前加载绑定的CUDA插件。
图内已经包含历史/音乐/前缀编码和音乐CFG；本模块只执行训练Actor相同的
SpacedDiffusion确定性DDIM、已知坐标约束、physical qpos30回填与qpos28解码。
生成从12帧标准站姿开始，以120帧窗口/12帧前缀/108帧续接滚动；历史只读取
截至决策帧的自身轨迹，种子使用seed+window_index。根位置和航向使用既有codec锚点。
输出是可以直接提交原安全桥的30Hz世界qpos28后缀，不做overlap-add、足锁或裁剪。
在线控制台逐块提交，buffered控制台收齐整段后提交，通信与播放时钟不由本模块改变。
本模块不加载训练checkpoint、训练Actor或数据集，不支持旧五输入模型。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from gem.diffusion_utils import gaussian_diffusion as gd
from gem.diffusion_utils.respace import SpacedDiffusion, space_timesteps
from gem.robots.bumi.endecoder import BumiEndecoder
from gem.robots.bumi.feature_codec import (
    BUMI_REPRESENTATION_CONTRACT_VERSION, make_quaternion_continuous,
)
from gem.runtime.bumi_music_contract import (
    BUMI_ENGINE_CONTRACT, BUMI_ONNX_CONTRACT_VERSION, BUMI_ONNX_INPUTS,
    BUMI_ONNX_INPUT_DTYPES, BUMI_ONNX_OUTPUTS, BUMI_ONNX_OUTPUT_DTYPES, BUMI_TRT_INPUT_DTYPES,
    BUMI_STAGE1_PRECISION_POLICY, BUMI_STAGE1_SAMPLING_CONTRACT_VERSION,
    HISTORY_STEPS, MUSIC_DIM, PREFIX_FRAMES, SOURCE_FPS, STRIDE_FRAMES, WINDOW_FRAMES,
    validate_stage1_engine_build_options,
)
from gem.runtime.bumi_stage1_history import CausalDemoProprio48Builder
from gem.runtime.music_only_trt import TensorRTStepRunner, gpu_fingerprint, sha256_file

BUMI_MOTION_DIM = 30
TORCH_DTYPES = {"float32": torch.float32, "int64": torch.int64, "bool": torch.bool}


def validate_stage1_metadata(metadata):
    """只接受当前固定形状Stage1导出图及其扩散/表示语义。"""
    if metadata.get("contract_version") != BUMI_ONNX_CONTRACT_VERSION:
        raise ValueError("模型不是当前Stage1十一输入ONNX")
    if metadata.get("input_contract") != BUMI_ONNX_INPUTS:
        raise ValueError("Stage1 ONNX输入形状不匹配")
    if metadata.get("output_contract") != BUMI_ONNX_OUTPUTS:
        raise ValueError("Stage1 ONNX输出形状不匹配")
    interface = metadata.get("interface", {})
    expected = {"history_steps": HISTORY_STEPS, "motion_frames": WINDOW_FRAMES,
                "motion_fps": SOURCE_FPS, "diffusion_steps": 1000, "noise_schedule": "cosine",
                "known_policy": "clean_x0_every_step_coordinate_mask",
                "cfg_policy": "music_only_dropout_shared_history_and_prefix",
                "contact_condition": False}
    for key, value in expected.items():
        if interface.get(key) != value:
            raise ValueError(f"Stage1语义不匹配: {key}={interface.get(key)!r}")
    identity = metadata.get("asset_identity", {})
    if identity.get("representation_contract_version") != BUMI_REPRESENTATION_CONTRACT_VERSION:
        raise ValueError("Stage1运动表示不匹配")
    return interface


def stage1_warmup_inputs(device, guidance_scale=2.5):
    """构造合法的空历史/无前缀输入，供运行时预热使用。"""
    device = torch.device(device)
    values = {name: torch.zeros(shape, dtype=TORCH_DTYPES[BUMI_ONNX_INPUT_DTYPES[name]],
                                device=device)
              for name, shape in BUMI_ONNX_INPUTS.items()}
    values["diffusion_timestep"].fill_(999)
    values["music_valid"].fill_(True)
    values["future_valid"].fill_(True)
    values["guidance_scale"].fill_(guidance_scale)
    return values


def _check_step_inputs(values):
    if set(values) != set(BUMI_ONNX_INPUTS):
        raise ValueError("Stage1必须提供全部十一输入")
    for name, shape in BUMI_ONNX_INPUTS.items():
        value = values[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
            raise ValueError(f"Stage1输入{name}必须为{shape}")
        if value.dtype != TORCH_DTYPES[BUMI_ONNX_INPUT_DTYPES[name]]:
            raise ValueError(f"Stage1输入{name}类型必须为{BUMI_ONNX_INPUT_DTYPES[name]}")


def bumi_engine_cache_key(*, onnx_sha256, checkpoint_sha256, metadata_sha256,
                          tensorrt_version, precision, gpu, build_options):
    build_options = validate_stage1_engine_build_options(build_options)
    value = {"contract": BUMI_ENGINE_CONTRACT, "onnx_sha256": onnx_sha256,
             "checkpoint_sha256": checkpoint_sha256, "metadata_sha256": metadata_sha256,
             "tensorrt_version": str(tensorrt_version), "precision": precision, "gpu": gpu,
             "precision_policy": BUMI_STAGE1_PRECISION_POLICY,
             "build_options": build_options,
             "inputs": BUMI_ONNX_INPUTS, "input_dtypes": BUMI_TRT_INPUT_DTYPES,
             "source_input_dtypes": BUMI_ONNX_INPUT_DTYPES,
             "outputs": BUMI_ONNX_OUTPUTS}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class BumiOrtStepRunner:
    """Stage1固定形状ONNX后端；CUDA请求不静默退回CPU。"""

    REQUIRED_INPUTS = {name: tuple(shape) for name, shape in BUMI_ONNX_INPUTS.items()}
    REQUIRED_OUTPUTS = {name: tuple(shape) for name, shape in BUMI_ONNX_OUTPUTS.items()}

    def __init__(self, onnx_path, *, device="cpu", provider="cpu"):
        import onnxruntime as ort

        self.path = Path(onnx_path).expanduser().resolve(strict=True)
        self.device = torch.device(device)
        provider_name = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}.get(provider)
        if provider_name is None or provider_name not in ort.get_available_providers():
            raise RuntimeError(f"请求的ONNX后端不可用: {provider}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        providers = ([(provider_name, {"device_id": self.device.index or 0, "use_tf32": 0}),
                      "CPUExecutionProvider"] if provider == "cuda" else [provider_name])
        self.session = ort.InferenceSession(str(self.path), sess_options=options, providers=providers)
        if self.session.get_providers()[0] != provider_name:
            raise RuntimeError("请求的ONNX provider未成功加载")
        self._validate_io(self.session.get_inputs(), BUMI_ONNX_INPUTS, BUMI_ONNX_INPUT_DTYPES)
        self._validate_io(self.session.get_outputs(), BUMI_ONNX_OUTPUTS, BUMI_ONNX_OUTPUT_DTYPES)

    @staticmethod
    def _validate_io(actual, shapes, dtypes):
        if [value.name for value in actual] != list(shapes):
            raise RuntimeError("Stage1 ONNX张量名称/顺序不匹配")
        types = {"float32": "tensor(float)", "int64": "tensor(int64)", "bool": "tensor(bool)"}
        for value in actual:
            if list(value.shape) != shapes[value.name] or value.type != types[dtypes[value.name]]:
                raise RuntimeError(f"Stage1 ONNX形状/类型不匹配: {value.name}")

    def __call__(self, values):
        _check_step_inputs(values)
        feed = {name: value.detach().cpu().numpy() for name, value in values.items()}
        outputs = self.session.run(list(BUMI_ONNX_OUTPUTS), feed)
        return tuple(torch.from_numpy(value).to(self.device) for value in outputs)


class BumiTensorRTStepRunner(TensorRTStepRunner):
    """Stage1专用TensorRT后端，保留ABI/GPU/完整文件指纹检查。"""

    REQUIRED_INPUTS = {name: tuple(shape) for name, shape in BUMI_ONNX_INPUTS.items()}
    REQUIRED_OUTPUTS = {name: tuple(shape) for name, shape in BUMI_ONNX_OUTPUTS.items()}
    REQUIRED_DTYPES = {**BUMI_TRT_INPUT_DTYPES, **BUMI_ONNX_OUTPUT_DTYPES}

    def _validate_manifest(self, required):
        path = self.engine_path.parent / "engine.json"
        if not path.is_file():
            raise RuntimeError(f"Stage1 engine元数据缺失: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("contract_version") != BUMI_ENGINE_CONTRACT:
            raise RuntimeError("只支持Stage1 TensorRT引擎")
        if payload.get("precision") != "fp32" or payload.get("precision_policy") != BUMI_STAGE1_PRECISION_POLICY:
            raise RuntimeError("Stage1部署要求FP32并关闭TF32")
        build_options = validate_stage1_engine_build_options(payload.get("build_options"))
        if payload.get("engine_sha256") != sha256_file(self.engine_path):
            raise RuntimeError("Stage1 engine SHA256不匹配")
        if payload.get("representation_contract_version") != BUMI_REPRESENTATION_CONTRACT_VERSION:
            raise RuntimeError("Stage1 engine运动表示不匹配")
        for key, expected in (("input_shapes", BUMI_ONNX_INPUTS),
                              ("input_dtypes", BUMI_TRT_INPUT_DTYPES),
                              ("source_input_dtypes", BUMI_ONNX_INPUT_DTYPES),
                              ("output_shapes", BUMI_ONNX_OUTPUTS),
                              ("output_dtypes", BUMI_ONNX_OUTPUT_DTYPES)):
            if payload.get(key) != expected:
                raise RuntimeError(f"Stage1 engine契约不匹配: {key}")
        for key, actual in (("tensorrt_version", str(self.trt.__version__)),
                            ("libnvinfer_version", self.linked_tensorrt_version)):
            if str(payload.get(key, "")).split(".")[:2] != actual.split(".")[:2]:
                raise RuntimeError(f"Stage1 engine ABI不匹配: {key}")
        gpu = gpu_fingerprint(self.device)
        if payload.get("gpu") != gpu:
            raise RuntimeError("Stage1 engine GPU/CUDA指纹不匹配")
        expected = bumi_engine_cache_key(
            onnx_sha256=payload["onnx_sha256"], checkpoint_sha256=payload["checkpoint_sha256"],
            metadata_sha256=payload["onnx_metadata_sha256"],
            tensorrt_version=payload["tensorrt_version"], precision="fp32", gpu=gpu,
            build_options=build_options)
        if payload.get("cache_key") != expected:
            raise RuntimeError("Stage1 engine缓存指纹不匹配")
        from gem.runtime.bumi_stage1_plugin import engine_plugin_path, load_stage1_plugin

        plugin_path, plugin_sha = engine_plugin_path(self.engine_path, payload)
        self.stage1_plugin_handle = load_stage1_plugin(plugin_path, plugin_sha, self.trt)
        return payload

    def __call__(self, values):
        _check_step_inputs(values)
        with self._lock:
            for name, value in values.items():
                # copy_按目标缓冲类型将bool转换为0/1 INT32；调用者契约仍为原Stage1 bool。
                self._buffers[name].copy_(value.to(device=self.device))
            if self.cuda_graph is None:
                self._execute()
            else:
                self.cuda_graph.replay()
            return tuple(self._buffers[name] for name in BUMI_ONNX_OUTPUTS)


@dataclass(frozen=True, slots=True)
class Stage1Window:
    index: int
    start: int
    valid_length: int
    prefix_length: int


def plan_stage1_windows(num_frames):
    """仅规划Stage1前缀续接，不复用旧30帧overlap-add规划器。"""
    if num_frames <= 0:
        raise ValueError("动作帧数必须为正数")
    result = []
    start = 0
    while True:
        count = min(WINDOW_FRAMES, num_frames - start)
        result.append(Stage1Window(len(result), start, count, min(PREFIX_FRAMES, count)))
        if start + count >= num_frames:
            return result
        start += STRIDE_FRAMES


@dataclass(frozen=True, slots=True)
class BumiOnlineGeneratedChunk:
    """已最终确定的世界qpos后缀，与原控制台数据块接口一致。"""

    window_index: int
    absolute_start_frame: int
    total_frames: int
    qpos: torch.Tensor
    is_last: bool


class Stage1DdimSampler:
    """从训练Actor剥离出的最小DDIM路径，使用同一扩散工具和编解码器。"""

    def __init__(self, runner, endecoder, *, device, steps, guidance_scale):
        if not 2 <= int(steps) <= 1000:
            raise ValueError("DDIM步数必须为2..1000")
        if not math.isfinite(guidance_scale) or guidance_scale <= 0:
            raise ValueError("CFG必须为有限正数")
        self.runner, self.endecoder = runner, endecoder
        self.device = torch.device(device)
        self.guidance_scale = float(guidance_scale)
        self.diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(1000, str(int(steps))),
            betas=gd.get_named_beta_schedule("cosine", 1000, 1.0),
            model_mean_type=gd.ModelMeanType.START_X, model_var_type=gd.ModelVarType.FIXED_SMALL,
            loss_type=gd.LossType.MSE, rescale_timesteps=False)

    @torch.inference_mode()
    def sample(self, conditions, noise):
        feed = {name: value.to(self.device) for name, value in conditions.items()}
        mask = feed["known_qpos30_mask"]
        valid = feed["future_valid"]
        known_physical = feed["known_qpos30"]
        known_x = self.endecoder.normalize(torch.where(mask, known_physical, 0.0))
        known_x = torch.where(mask, known_x, 0.0)

        def constrain(value):
            return torch.where(valid[..., None], torch.where(mask, known_x, value), 0.0)

        xt = constrain(noise.to(device=self.device, dtype=torch.float32))
        feed["guidance_scale"] = xt.new_tensor([self.guidance_scale])

        def denoise(value, timestep, **kwargs):
            feed["noisy_motion"], feed["diffusion_timestep"] = value, timestep
            motion, contact = self.runner(feed)
            if not bool(torch.isfinite(motion).all()) or not bool(torch.isfinite(contact).all()):
                raise FloatingPointError("Stage1去噪输出含NaN/Inf")
            return {"pred_x_start": motion, "static_conf_logits": contact}

        for index in range(self.diffusion.num_timesteps - 1, -1, -1):
            timestep = torch.full((1,), index, dtype=torch.int64, device=self.device)
            output = self.diffusion.ddim_sample(
                denoise, xt, timestep, clip_denoised=False, model_kwargs={"y": {}}, eta=0.0)
            xt = constrain(output["sample"])
        physical = self.endecoder.denormalize(xt)
        physical = torch.where(mask, known_physical, physical)
        physical = torch.where(valid[..., None], physical, 0.0)
        canonical = self.endecoder.codec.decode_to_canonical_qpos(physical)
        return {"qpos30": physical,
                "qpos": torch.where(valid[..., None], canonical, 0.0),
                "contact_logits": output["static_conf_logits"].clone()}


class BumiStage1QposGenerator:
    """两种控制台共用的Stage1自回归生成器，输出连续最终qpos块。"""

    def __init__(self, runner, endecoder, *, device="cuda:0", steps=20, guidance_scale=2.5):
        if not isinstance(endecoder, BumiEndecoder):
            raise TypeError("Stage1生成器要求BumiEndecoder")
        self.endecoder = endecoder
        self.device = torch.device(device)
        self.sampler = Stage1DdimSampler(runner, endecoder, device=device,
                                        steps=steps, guidance_scale=guidance_scale)
        self.history_builder = CausalDemoProprio48Builder(endecoder.kinematics)
        self.windows_generated = self.emitted_frames = self.pending_frames = 0

    @torch.inference_mode()
    def generate(self, music, *, seed=42):
        if music.ndim != 2 or music.shape[1] != MUSIC_DIM or len(music) <= 0:
            raise ValueError("音乐特征必须为有限的[T,35]")
        music = music.detach().cpu().float()
        if not bool(torch.isfinite(music).all()):
            raise ValueError("音乐特征含NaN/Inf")
        total = len(music)
        codec = self.endecoder.codec
        generated = codec.kinematics.make_standing_qpos().cpu().repeat(min(PREFIX_FRAMES, total), 1)
        self.windows_generated = self.emitted_frames = self.pending_frames = 0
        if total <= PREFIX_FRAMES:
            self.windows_generated = 1
            self.emitted_frames = total
            yield BumiOnlineGeneratedChunk(0, 0, total, generated.clone(), True)
            return
        for window in plan_stage1_windows(total):
            decision, count = window.start, window.valid_length
            prefix = generated[decision:decision + PREFIX_FRAMES]
            encoded = codec.encode(prefix)
            known = torch.zeros(WINDOW_FRAMES, 30)
            mask = torch.zeros_like(known, dtype=torch.bool)
            mask[:len(prefix), 2:] = True
            mask[:max(len(prefix) - 1, 0), :2] = True
            known[:len(prefix)] = encoded.physical_features
            known[~mask] = 0.0
            history, history_valid, history_times = self.history_builder.build_history(
                generated[:decision + 1], decision_frame=decision, history_steps=HISTORY_STEPS)
            music_window = torch.zeros(WINDOW_FRAMES, MUSIC_DIM)
            music_window[:count] = music[decision:decision + count]
            valid = torch.arange(WINDOW_FRAMES) < count
            conditions = {
                "music_features": music_window[None], "music_valid": valid[None],
                "proprio_history": history[None], "proprio_history_valid": history_valid[None],
                "history_relative_times": (history_times - decision / SOURCE_FPS).float()[None],
                "known_qpos30": known[None], "known_qpos30_mask": mask[None],
                "future_valid": valid[None],
            }
            noise = torch.randn((1, WINDOW_FRAMES, 30),
                                generator=torch.Generator().manual_seed(int(seed) + window.index))
            result = self.sampler.sample(conditions, noise)
            if not torch.equal(result["qpos30"][0].cpu()[mask], known[mask]):
                raise RuntimeError("Stage1已提交前缀发生变化")
            canonical = result["qpos"][0, :count].cpu()
            segment = codec.apply_world_anchor(canonical, {
                "root_xy": encoded.anchor.position_w[..., :2], "yaw": encoded.anchor.yaw,
                "anchor_z": encoded.anchor.default_root_height})
            old_count = len(generated)
            generated = torch.cat((generated, segment[len(prefix):]), 0)
            generated[:, 3:7] = make_quaternion_continuous(generated[:, 3:7])
            if not bool(torch.isfinite(generated).all()):
                raise FloatingPointError("Stage1生成轨迹含NaN/Inf")
            self.windows_generated += 1
            start = 0 if window.index == 0 else old_count
            qpos = generated[start:].clone()
            self.emitted_frames = len(generated)
            yield BumiOnlineGeneratedChunk(window.index, start, total, qpos, len(generated) == total)
