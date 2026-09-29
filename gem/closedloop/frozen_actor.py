"""Stage8 常驻冻结 GENMO worker：严格加载 Stage1 权重并进行纯条件采样。

本模块直接复用 Stage1Actor、训练时的同一架构构造器和 load_stage1_checkpoint；仅从
checkpoint 恢复模型，不恢复 optimizer、scheduler、AMP 或训练 global_step。stats 和
kinematics 允许跨机器改变路径，但内容身份仍由原 checkpoint 校验器严格验证。不会导入
包含 MuJoCo 渲染的评估脚本，也不会读取训练动作数据。所有参数关闭梯度，采样运行在
eval/inference_mode 下；启动与关闭时对参数、全部 buffer 和资产文件执行冻结性检查。

RPC generate 只接受既有十个条件字段，监督和未知字段均报错。每次请求按 episode seed
与 decision_id 生成稳定的独立初始噪声；保留物理 qpos30 已知坐标，将完整窗口用请求中的
同一个 world_anchor 放回旧参考坐标系，返回 qpos_world[120,28]、qpos30[120,30] 和
contact[120,2]。600 Hz 时钟、计划提交和物理仿真由协调器与 GMT worker 管理。本模块的
延迟仅度量 Actor 处理过程，端到端通信/参考准备延迟由协调器另外度量。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from gem.closedloop.checkpoint import load_stage1_checkpoint
from gem.closedloop.contracts import STAGE1_CONDITION_KEYS, validate_stage1_condition_batch
from gem.closedloop.training import build_stage1_actor
from gem.robots.bumi.kinematics import sha256_file


def stable_noise_seed(episode_seed: int, decision_id: int | str) -> int:
    """跨进程稳定的请求噪声种子，不使用带进程盐值的 Python hash。"""
    if isinstance(episode_seed, bool) or not isinstance(episode_seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if isinstance(decision_id, bool) or not isinstance(decision_id, (str, int, np.integer)):
        raise TypeError("decision_id must be an integer or string")
    decision_id = int(decision_id) if isinstance(decision_id, np.integer) else decision_id
    key = json.dumps([int(episode_seed), decision_id], ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little") & ((1 << 63) - 1)


def _fingerprint(actor: torch.nn.Module) -> str:
    """包含非持久化 mean/std/物理尺度/FK buffer，逐张量取字节避免复制完整模型。"""
    digest = hashlib.sha256()
    for kind, items in (("parameter", actor.named_parameters()), ("buffer", actor.named_buffers())):
        for name, tensor in items:
            value = tensor.detach().cpu().contiguous()
            header = json.dumps([kind, name, str(value.dtype), list(value.shape)], separators=(",", ":"))
            digest.update(header.encode("utf-8"))
            # uint8 视图同样支持 bfloat16；不把低精度权重转换到不同的数值表示。
            digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


class FrozenStage1Actor:
    """只持有冻结 Actor 与资产身份，不提供训练或 optimizer 接口。"""

    def __init__(self, config: Mapping[str, Any]) -> None:
        config = OmegaConf.to_container(OmegaConf.create(config), resolve=True)
        paths, runtime, model = config["paths"], config["runtime"], config["model"]
        threads = int(runtime.get("torch_threads", 4))
        if threads <= 0:
            raise ValueError("runtime.torch_threads must be positive")
        torch.set_num_threads(threads)
        self.device = torch.device(runtime["genmo_device"])
        self.steps = int(model.get("ddim_steps", 20))
        self.guidance_scale = float(model.get("guidance_scale", 2.5))
        if not 2 <= self.steps <= 1000 or not math.isfinite(self.guidance_scale):
            raise ValueError("invalid DDIM steps/guidance_scale")
        self.paths = {key: Path(paths[key]).expanduser().resolve() for key in ("checkpoint", "stats", "kinematics")}
        for path in self.paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        self.asset_hashes = {key: sha256_file(path) for key, path in self.paths.items()}
        # 本项目自己的可信 checkpoint 内含 OmegaConf 和训练元数据；只消费架构与 state_dict。
        payload = torch.load(self.paths["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(payload, Mapping):
            raise TypeError("Stage1 checkpoint must be a mapping")
        train_config = OmegaConf.create(payload["config"])
        train_config.endecoder.stats_path = str(self.paths["stats"])
        train_config.endecoder.kinematics_path = str(self.paths["kinematics"])
        train_config.endecoder.allow_placeholder_stats = False
        history_steps = int(payload["actor_interface_config"]["history_steps"])
        if int(model["history_steps"]) != history_steps:
            raise ValueError("configured H differs from the frozen checkpoint interface")
        # 架构构造器只需这些字段；不加载、改写或实例化任何四库 Dataset。
        data = OmegaConf.create({
            "qpos30_stats": {"path": str(self.paths["stats"])},
            "dataset_defaults": {"kinematics_path": str(self.paths["kinematics"])},
            "sample_contract": {"history_steps": history_steps},
            "datasets": {},
        })
        actor = build_stage1_actor(train_config, data)
        loading = load_stage1_checkpoint(actor, payload)
        for name, tensor in actor.state_dict().items():
            if not bool(torch.isfinite(tensor).all()):
                raise FloatingPointError(f"checkpoint contains nonfinite weights: {name}")
        actor.eval().requires_grad_(False)
        self.initial_fingerprint = _fingerprint(actor)
        self.report = {
            "worker": "frozen_stage1_actor",
            "python": sys.executable,
            "source": str(Path(__file__).resolve()),
            "paths": {key: str(path) for key, path in self.paths.items()},
            "sha256": dict(self.asset_hashes),
            "checkpoint_sha256": self.asset_hashes["checkpoint"],
            "stats_sha256": self.asset_hashes["stats"],
            "kinematics_sha256": self.asset_hashes["kinematics"],
            "checkpoint_global_step_provenance": payload.get("global_step"),
            "checkpoint_version": payload["checkpoint_version"],
            "interface": dict(actor.interface_config),
            "loading": loading,
            "optimizer_restored": False,
            "global_step_restored": False,
            "parameter_fingerprint": self.initial_fingerprint,
            "fingerprint_includes_all_buffers": True,
            "device": str(self.device),
            "ddim_steps": self.steps,
            "guidance_scale": self.guidance_scale,
        }
        del payload
        gc.collect()
        self.actor = actor.to(self.device)
        self.closed = False
        self.final_report: dict[str, Any] | None = None

    def hello(self) -> dict[str, Any]:
        """返回严格加载身份；避免每次 hello 将整个 GPU 模型回传 CPU。"""
        return {**self.report, "frozen": self._is_frozen(), "closed": self.closed}

    def _is_frozen(self) -> bool:
        return not any(module.training for module in self.actor.modules()) and not any(
            parameter.requires_grad for parameter in self.actor.parameters()
        )

    @torch.inference_mode()
    def generate(self, conditions: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
        """仅条件生成；不允许混入 target 或未知键，不改变 caller 的张量和元数据。"""
        if self.closed:
            raise RuntimeError("frozen Actor worker is closed")
        if not self._is_frozen():
            raise RuntimeError("Actor left eval/frozen state")
        if set(conditions) != set(STAGE1_CONDITION_KEYS):
            missing = sorted(set(STAGE1_CONDITION_KEYS) - set(conditions))
            extra = sorted(set(conditions) - set(STAGE1_CONDITION_KEYS))
            raise ValueError(f"only Stage1 condition keys are accepted; missing={missing}, extra={extra}")
        required_meta = {"seed", "decision_id", "decision_tick", "prefix_frames", "world_anchor"}
        if required_meta - set(meta):
            raise ValueError(f"missing generation metadata: {sorted(required_meta - set(meta))}")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        batch = {}
        for key, value in conditions.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.detach().to(self.device)
            else:
                batch[key] = torch.as_tensor(np.array(value, copy=True), device=self.device)
        validate_stage1_condition_batch(batch, history_steps=self.actor.history_steps)
        if batch["decision_time"].shape != (1,):
            raise ValueError("Stage8 frozen worker currently requires B=1")
        if not math.isclose(float(batch["decision_time"][0]), int(meta["decision_tick"]) / 600.0, abs_tol=1e-8):
            raise ValueError("condition time differs from request decision_tick")
        prefix = int(meta["prefix_frames"])
        if not 0 <= prefix < 120 or int(batch["known_qpos30_mask"].any(-1).sum()) != prefix:
            raise ValueError("prefix metadata differs from coordinate mask")
        anchor = torch.as_tensor(meta["world_anchor"], dtype=torch.float32, device=self.device)
        if anchor.shape != (4,) or not bool(torch.isfinite(anchor).all()):
            raise ValueError("world_anchor must be finite [x,y,z,yaw]")
        if not torch.isclose(anchor[2], self.actor.endecoder.codec.default_root_height.to(anchor), atol=1e-6, rtol=0):
            raise ValueError("world_anchor Z must equal the existing codec default root reference")
        seed = stable_noise_seed(meta["seed"], meta["decision_id"])
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn((1, 120, 30), generator=generator, dtype=torch.float32).to(self.device)
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        # DDIM eta=0 仍调用 randn_like；隔离其 RNG 副作用，不污染其他 episode 的随机状态。
        with torch.random.fork_rng(devices=devices):
            torch.random.set_rng_state(generator.get_state())
            if devices:
                device_generator = torch.Generator(device=self.device).manual_seed(seed)
                torch.cuda.set_rng_state(device_generator.get_state(), self.device)
            sample = self.actor.sample(batch, steps=self.steps, guidance_scale=self.guidance_scale, noise=noise)
        world = self.actor.endecoder.codec.apply_world_anchor(sample["qpos"], anchor)
        result = {**dict(meta)}
        for name, value, width in (("qpos_world", world, 28), ("qpos30", sample["qpos30"], 30), ("contact", sample["contact"], 2)):
            if value.shape != (1, 120, width) or not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"Actor produced invalid {name}")
            result[name] = value[0].detach().cpu().to(torch.float32).numpy().copy()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        result.update({"noise_seed": seed, "latency_s": time.perf_counter() - started})
        return result

    def verify_frozen(self) -> dict[str, Any]:
        """逐字节比较参数/全部 buffer，并重新核验磁盘资产；不写模型文件。"""
        current = _fingerprint(self.actor)
        hashes = {key: sha256_file(path) for key, path in self.paths.items()}
        checks = {
            "eval_and_no_grad": self._is_frozen(),
            "parameters_and_buffers_unchanged": current == self.initial_fingerprint,
            "asset_files_unchanged": hashes == self.asset_hashes,
        }
        report = {"frozen_checks": checks, "parameter_fingerprint": current, "sha256": hashes}
        if not all(checks.values()):
            raise RuntimeError(f"frozen Actor integrity failed: {json.dumps(checks)}")
        return report

    def close(self) -> dict[str, Any]:
        """RPC server 在返回本报告后结束服务；关闭重复调用不重新执行耗时校验。"""
        if self.final_report is None:
            self.final_report = {"closed": True, **self.verify_frozen()}
            self.closed = True
        return self.final_report


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结 Stage1 GENMO 纯条件采样 worker")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    from gem.runtime.closedloop_protocol import RpcServer

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    worker = FrozenStage1Actor(config)
    RpcServer(args.socket, worker).serve()


if __name__ == "__main__":
    main()
