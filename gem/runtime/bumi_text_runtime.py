"""BUMI 文本常驻推理与单步导出共用运行时。

训练端只需提供带自描述契约的checkpoint；部署端仅加载deployment.json及导出资产，
不导入Lightning/Hydra，不加载训练权重。T5本地常驻，GENMO单步后端可为PyTorch、
ONNX或TensorRT，DDIM、qpos解码和足锁共用。固定300张量以真实length屏蔽padding，
最终只解码和保存F帧；初始噪声按F生成后补零，便于不同计算长度做数值对照。
"""

from __future__ import annotations

import copy
import gc
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import torch
from torch import nn

from gem.runtime.bumi_text_contract import (
    inspect_payload,
    resolve_assets,
    sha256_file,
    validate_contract,
)

INPUTS = {
    "noisy_motion": (1, 300, 30),
    "diffusion_timestep": (1,),
    "text_embed": (1, 150, 1024),
    "text_attention_mask": (1, 150),
    "length": (1,),
    "guidance_scale": (1,),
}
OUTPUTS = {"pred_motion": (1, 300, 30), "pred_foot_contact_logits": (1, 300, 2)}
EXPORT_SCHEMA = "genmo.bumi_text_onnx.v1"
BUNDLE_SCHEMA = "genmo.bumi_text_deployment.v1"
T5_DEFAULT = "/home/weili/.cache/huggingface/hub/models--t5-3b/snapshots/bed96aab9ee46012a5046386105ee5fd0ac572f0"


class BumiTextGuidedDenoiser(nn.Module):
    """同次Transformer计算有/无文本CFG；无条件分支保留相同token mask。"""

    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser

    def forward(
        self,
        noisy_motion,
        diffusion_timestep,
        text_embed,
        text_attention_mask,
        length,
        guidance_scale,
    ):
        paired = torch.cat((noisy_motion, noisy_motion), dim=0)
        y = dict(
            f_cond=noisy_motion.new_zeros(2, noisy_motion.shape[1], self.denoiser.latent_dim),
            length=torch.cat((length, length)),
            encoded_text=torch.cat((text_embed, torch.zeros_like(text_embed))),
            text_attention_mask=torch.cat((text_attention_mask, text_attention_mask)),
        )
        output = self.denoiser(
            paired, torch.cat((diffusion_timestep, diffusion_timestep)), y=y, inputs={}
        )
        scale = guidance_scale.reshape(1, 1, 1)

        def guided(value):
            return value[1:2] + scale * (value[:1] - value[1:2])

        return guided(output["pred_x_start"]), guided(output["static_conf_logits"])


def load_checkpoint_step(path, device="cpu"):
    from gem.network.gem_denoiser import NetworkEncoderRoPE

    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    contract = inspect_payload(payload)
    cfg = copy.deepcopy(payload["bumi_text_denoiser_config"])
    cfg.pop("_target_", None)
    if (
        cfg["output_dim"],
        cfg["xt_dim"],
        cfg["static_conf_dim"],
        cfg["pred_cam_dim"],
        cfg["attention_mode"],
        cfg["max_len"],
        cfg["encode_text"],
    ) != (30, 30, 2, 0, "valid_length", contract["sequence"]["attention_max_len"], True):
        raise ValueError("保存的denoiser配置与BUMI文本契约不一致")
    denoiser = NetworkEncoderRoPE(**cfg)
    prefix = "pipeline.denoiser3d.denoiser."
    weights = {
        k[len(prefix) :]: v for k, v in payload["state_dict"].items() if k.startswith(prefix)
    }
    denoiser.load_state_dict(weights, strict=True)
    return (
        BumiTextGuidedDenoiser(denoiser).to(device).eval(),
        contract,
        payload["bumi_text_diffusion_config"],
    )


class BumiTextSampler:
    """eta=0的原DDIM公式；只保留当前状态，后端共享同一时间步映射。"""

    def __init__(self, step, diffusion_config, ddim_steps=50):
        self.step, self.config = step, dict(diffusion_config)
        self.ddim_steps = None
        self.set_ddim_steps(ddim_steps)

    def set_ddim_steps(self, steps):
        if type(steps) is not int or not 2 <= steps <= 1000:
            raise ValueError("ddim_steps必须为2–1000整数")
        if steps == self.ddim_steps:
            return False
        from gem.diffusion_utils.model_util import create_gaussian_diffusion

        self.config["test_timestep_respacing"] = str(steps)
        self.diffusion = create_gaussian_diffusion(SimpleNamespace(**self.config), training=False)
        self.ddim_steps = steps
        return True

    @torch.no_grad()
    def generate(self, text, mask, frames, *, guidance=2.5, seed=42, noise=None, tensor_frames=300):
        if type(frames) is not int or not 60 <= frames <= 300 or not frames <= tensor_frames <= 300:
            raise ValueError("num_frames必须为60–300，计算长度不得小于真实长度")
        if (
            text.shape != (1, 150, 1024)
            or mask.shape != (1, 150)
            or not mask.any()
            or not torch.isfinite(text).all()
        ):
            raise ValueError("T5特征/attention mask非法")
        device = text.device
        generator = torch.Generator(device=device).manual_seed(seed)
        valid_noise = (
            torch.randn((1, frames, 30), generator=generator, device=device)
            if noise is None
            else noise.to(device)
        )
        if valid_noise.shape != (1, frames, 30) or not torch.isfinite(valid_noise).all():
            raise ValueError("noise必须是有限[1,F,30]")
        x = torch.cat((valid_noise, valid_noise.new_zeros(1, tensor_frames - frames, 30)), dim=1)
        length = torch.tensor([frames], dtype=torch.long, device=device)
        scale = torch.tensor([guidance], dtype=torch.float32, device=device)
        alpha = torch.as_tensor(self.diffusion.alphas_cumprod, dtype=torch.float32, device=device)
        previous = torch.as_tensor(
            self.diffusion.alphas_cumprod_prev, dtype=torch.float32, device=device
        )
        for index in reversed(range(self.diffusion.num_timesteps)):
            timestep = torch.tensor(
                [self.diffusion.timestep_map[index]], dtype=torch.long, device=device
            )
            pred, contact = self.step(x, timestep, text, mask, length, scale)
            epsilon = (x - alpha[index].sqrt() * pred) / (1 - alpha[index]).sqrt()
            x = previous[index].sqrt() * pred + (1 - previous[index]).sqrt() * epsilon
        if not torch.isfinite(x[:, :frames]).all() or not torch.isfinite(contact[:, :frames]).all():
            raise FloatingPointError("去噪生成了非有限动作或接触")
        return x[:, :frames], contact[:, :frames]


def read_export_metadata(onnx_path):
    path = Path(onnx_path).resolve(strict=True)
    meta = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    if meta.get("schema") != EXPORT_SCHEMA or meta.get("onnx_sha256") != sha256_file(path):
        raise ValueError("ONNX来源/指纹不符")
    validate_contract(meta["model_contract"])
    for name, record in meta.get("external_data", {}).items():
        asset = (path.parent / name).resolve(strict=True)
        if not asset.is_relative_to(path.parent) or sha256_file(asset) != record["sha256"]:
            raise ValueError("ONNX外部权重指纹不符")
    if {k: tuple(v) for k, v in meta["inputs"].items()} != INPUTS or {
        k: tuple(v) for k, v in meta["outputs"].items()
    } != OUTPUTS:
        raise ValueError("ONNX文本输入输出契约错误")
    return meta


class OnnxTextStep:
    def __init__(self, path, providers=None):
        import onnxruntime as ort

        self.metadata = read_export_metadata(path)
        self.session = ort.InferenceSession(
            str(path), providers=providers or ["CPUExecutionProvider"]
        )
        if {v.name: tuple(v.shape) for v in self.session.get_inputs()} != INPUTS or {
            v.name: tuple(v.shape) for v in self.session.get_outputs()
        } != OUTPUTS:
            raise ValueError("ONNX真实图形状不符")

    def __call__(self, *args):
        feed = {name: value.detach().cpu().numpy() for name, value in zip(INPUTS, args)}
        result = self.session.run(list(OUTPUTS), feed)
        return tuple(torch.from_numpy(value).to(args[0].device) for value in result)


def read_bundle(path):
    path = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(path.read_text())
    if payload.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("部署清单版本错误")
    validate_contract(payload["model_contract"])
    files = {}
    for key, value in payload["files"].items():
        asset = (path.parent / value["path"]).resolve(strict=True)
        if not asset.is_relative_to(path.parent) or sha256_file(asset) != value["sha256"]:
            raise ValueError(f"部署资产路径或指纹错误: {key}")
        files[key] = asset
    for required in ("onnx", "onnx_metadata", "kinematics", "stats", "robot_manifest"):
        if required not in files:
            raise ValueError(f"部署清单缺少{required}")
    meta = read_export_metadata(files["onnx"])
    if (
        meta["model_contract"] != payload["model_contract"]
        or meta["source_checkpoint_sha256"] != payload["source_checkpoint_sha256"]
    ):
        raise ValueError("部署清单与ONNX模型来源不一致")
    for key in ("kinematics", "stats"):
        if payload["files"][key]["sha256"] != payload["model_contract"]["assets"][key]["sha256"]:
            raise ValueError("部署资产与训练资产不匹配")
    return payload, files, meta


class ResidentBumiTextEngine:
    """与网页工作进程兼容的常驻引擎；锁覆盖生成和DDIM更新。"""

    max_text_len = 150

    def __init__(
        self,
        ckpt_path=None,
        *,
        deployment_manifest=None,
        backend="torch",
        t5_model=T5_DEFAULT,
        device="cuda:0",
        ddim_steps=50,
        guidance_scale=2.5,
        output_root="outputs/bumi_text",
        kinematics=None,
        stats=None,
        local_files_only=True,
        **compat,
    ):
        if not local_files_only:
            raise ValueError("BUMI部署仅使用本地T5")
        if ckpt_path is not None and deployment_manifest is not None:
            raise ValueError("checkpoint与deployment清单不能混用")
        if deployment_manifest is not None and (kinematics is not None or stats is not None):
            raise ValueError("清单模式统一指定资产，拒绝路径覆盖")
        if backend not in {"torch", "onnx", "tensorrt"}:
            raise ValueError("未知推理后端")
        self.ckpt_path, self.manifest, self.backend = ckpt_path, deployment_manifest, backend
        self.device, self.t5_model = torch.device(device), str(t5_model)
        self.ddim_steps, self.guidance_scale = ddim_steps, guidance_scale
        self.output_root, self.asset_overrides = (
            Path(output_root),
            dict(kinematics=kinematics, stats=stats),
        )
        self.lock = threading.RLock()
        self.initialized = False
        self.tokenizer = self.text_encoder = self.step = self.sampler = self.endecoder = None

    def initialize(self):
        with self.lock:
            if self.initialized:
                return
            from gem.robots.bumi.endecoder import BumiEndecoder

            if self.manifest is not None:
                payload, files, meta = read_bundle(self.manifest)
                self.contract = payload["model_contract"]
                assets = files
                self.source_checkpoint_sha256 = payload["source_checkpoint_sha256"]
                diffusion = meta["diffusion_config"]
                if self.backend == "onnx":
                    providers = ["CPUExecutionProvider"]
                    if self.device.type == "cuda":
                        import onnxruntime as ort

                        if "CUDAExecutionProvider" not in ort.get_available_providers():
                            raise RuntimeError("请求CUDA ONNX后端，但当前ORT不支持CUDA")
                        providers = [
                            ("CUDAExecutionProvider", {"device_id": self.device.index or 0}),
                            "CPUExecutionProvider",
                        ]
                    self.step = OnnxTextStep(files["onnx"], providers=providers)
                    if (
                        self.device.type == "cuda"
                        and "CUDAExecutionProvider" not in self.step.session.get_providers()
                    ):
                        raise RuntimeError("ORT CUDA初始化失败，请检查CUDA/cuDNN运行库")
                elif self.backend == "tensorrt":
                    from gem.runtime.bumi_text_tensorrt import TextTensorRTStep

                    self.step = TextTensorRTStep(
                        files["engine"], onnx_metadata=meta, device=self.device
                    )
                else:
                    raise ValueError("无checkpoint部署请选择onnx或tensorrt")
                self.robot_manifest = files["robot_manifest"]
            else:
                if self.backend != "torch" or self.ckpt_path is None:
                    raise ValueError("torch需要checkpoint，导出后端需要deployment清单")
                self.step, self.contract, diffusion = load_checkpoint_step(
                    self.ckpt_path, self.device
                )
                assets = resolve_assets(
                    self.contract, checkpoint=self.ckpt_path, **self.asset_overrides
                )
                self.source_checkpoint_sha256 = sha256_file(self.ckpt_path)
                self.robot_manifest = (
                    Path(__file__).resolve().parents[2] / "assets/bumi_viewer/manifest.json"
                )
            self.asset_paths = assets
            self.sequence_contract = self.contract["sequence"]
            self.endecoder = BumiEndecoder(
                assets["kinematics"], assets["stats"], sequence_mode="full"
            ).to(self.device)
            if list(self.endecoder.kinematics.joint_order) != self.contract["joint_names"]:
                raise ValueError("运行时关节顺序与checkpoint不一致")
            self.sampler = BumiTextSampler(self.step, diffusion, self.ddim_steps)
            self.initialized = True

    def set_ddim_steps(self, steps):
        with self.lock:
            self.initialize()
            changed = self.sampler.set_ddim_steps(steps)
            self.ddim_steps = steps
            return changed

    def encode_prompt(self, prompt):
        from transformers import T5Tokenizer, T5EncoderModel
        from gem.runtime.resident_text_motion import encode_prompt_with_loaded_t5

        if self.text_encoder is None:
            source = Path(self.t5_model).expanduser().resolve(strict=True)
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.tokenizer = T5Tokenizer.from_pretrained(str(source), local_files_only=True)
            self.text_encoder = (
                T5EncoderModel.from_pretrained(
                    str(source), local_files_only=True, torch_dtype=dtype
                )
                .to(self.device)
                .eval()
            )
            self.text_encoder.requires_grad_(False)
        embedding = encode_prompt_with_loaded_t5(
            prompt, self.tokenizer, self.text_encoder, self.device, 150
        )
        mask = self.tokenizer(
            [prompt.strip()],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=150,
        )["attention_mask"].bool()
        return embedding.unsqueeze(0).to(self.device), mask.to(self.device)

    @torch.no_grad()
    def generate_arrays(self, text, mask, frames, *, seed=42, postproc=True, noise=None):
        from gem.robots.bumi.postprocess import lock_bumi_foot_contacts

        self.initialize()
        features, contact = self.sampler.generate(
            text, mask, frames, seed=seed, guidance=self.guidance_scale, noise=noise
        )
        decoded = self.endecoder.decode(features)
        raw = self.endecoder.compose_qpos(
            decoded, world_anchor={"root_xy": [0.0, 0.0], "yaw": 0.0}
        )[0]
        logits = contact[0]
        qpos = (
            lock_bumi_foot_contacts(
                raw, logits, self.endecoder.kinematics, contact_is_logits=True
            ).qpos
            if postproc
            else raw.clone()
        )
        if not torch.isfinite(raw).all() or not torch.isfinite(qpos).all():
            raise FloatingPointError("qpos解码或后处理出现非有限值")
        return dict(
            qpos=qpos.cpu().numpy(),
            qpos_raw=raw.cpu().numpy(),
            foot_contact_logits=logits.cpu().numpy(),
            fps=np.array(30),
            joint_names=np.array(self.contract["joint_names"]),
            quaternion_convention=np.array("wxyz"),
        )

    def generate(self, request):
        with self.lock:
            started = time.monotonic()
            try:
                prompt = request["prompt"]
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("prompt不能为空")
                frames = request.get("num_frames", 120)
                if (
                    type(frames) is not int
                    or not 60 <= frames <= 300
                    or request.get("fps", 30) != 30
                ):
                    raise ValueError("BUMI文本需要60–300帧、30FPS")
                self.initialize()
                text, mask = self.encode_prompt(prompt)
                arrays = self.generate_arrays(
                    text,
                    mask,
                    frames,
                    seed=request.get("seed", 42),
                    postproc=request.get("postproc", True),
                )
                output = Path(request.get("output_root", self.output_root)) / uuid4().hex
                output.mkdir(parents=True, exist_ok=False)
                np.savez_compressed(output / "motion.npz", **arrays)
                metadata = dict(
                    motion_backend="bumi",
                    prompt=prompt,
                    num_frames=frames,
                    fps=30,
                    ddim_steps=self.ddim_steps,
                    seed=request.get("seed", 42),
                    guidance_scale=self.guidance_scale,
                    postproc=request.get("postproc", True),
                    checkpoint_sha256=self.source_checkpoint_sha256,
                    contract=self.contract,
                    kinematics_path=str(self.asset_paths["kinematics"]),
                    robot_manifest=str(self.robot_manifest),
                )
                (output / "metadata.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2)
                )
                return dict(
                    ok=True,
                    output_dir=str(output),
                    num_frames=frames,
                    timing={"total_seconds": time.monotonic() - started},
                )
            except Exception as exc:
                return dict(ok=False, error=f"{type(exc).__name__}: {exc}")

    def close(self):
        with self.lock:
            self.step = self.sampler = self.endecoder = self.text_encoder = self.tokenizer = None
            self.initialized = False
            gc.collect()
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
