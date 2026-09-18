"""本地 SMPL/BUMI 文本 checkpoint 发现与校验。

按真实路径去重，使用文件身份、大小和纳秒时间戳缓存检查结果。扫描线程通过
CPU mmap 读取权重元数据，分别核对 SMPL 151D 或 BUMI 30D/2D 与机器人资产契约。
音乐和仅回归模型被排除；文件变化后重新读取，生成前再次核对文件身份。
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping
from pathlib import Path

from .storage import DEFAULT_CHECKPOINT, ROOT, atomic_json, fingerprint


def inspect_checkpoint(path: Path) -> dict:
    import torch

    from scripts.demo.demo_smpl_text import validate_text_generation_payload

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError as exc:
        if "mmap" not in str(exc):
            raise
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint 必须包含权重字典")
    if checkpoint.get("bumi_text_contract") is not None:
        from gem.runtime.bumi_text_contract import inspect_payload, resolve_assets
        robot = inspect_payload(checkpoint)
        resolve_assets(robot, checkpoint=path)
        return {"contract": {"max_text_len": 150, "encoded_text_dim": 1024,
                             "sequence_contract": robot["sequence"]},
                "motion_backend": "bumi", "min_frames": 60, "max_frames": 300,
                "global_step": checkpoint.get("global_step")}
    contract = validate_text_generation_payload(checkpoint, path)
    state = checkpoint.get("state_dict", checkpoint)
    prefix = "pipeline.denoiser3d.denoiser."
    output = state.get(prefix + "final_layer.fc2.weight")
    conditioning = state.get(prefix + "add_cond_linear.weight")
    if output is None or output.ndim != 2 or output.shape[0] != 151:
        raise ValueError("仅支持输出为 151 维的 SMPL 文本模型，不能加载 BUMI 模型")
    if conditioning is None or tuple(conditioning.shape) != (
        output.shape[1],
        output.shape[1] + 151,
    ):
        raise ValueError("缺少 SMPL 扩散条件权重，不能加载仅回归模型")

    def rejects(value):
        if isinstance(value, Mapping):
            return any(
                (key == "regression_only" and bool(item))
                or (key == "motion_backend" and item != "smpl")
                or rejects(item)
                for key, item in value.items()
            )
        return False

    if rejects(checkpoint.get("hyper_parameters", {})):
        raise ValueError("checkpoint 声明为回归或非 SMPL 模型")
    seq = contract.get("sequence_contract")
    return {"contract": contract, "motion_backend": "smpl", "global_step": checkpoint.get("global_step"),
            "min_frames": seq["min_frames"] if seq else 1, "max_frames": seq["max_frames"] if seq else 900}


class ModelRegistry:
    def __init__(self, output_root: Path, roots=None, inspector=inspect_checkpoint):
        self.path = output_root / "models.json"
        self.roots = (
            list(roots)
            if roots is not None
            else [ROOT / "inputs/pretrained", ROOT / "inputs/checkpoints"]
        )
        self.inspector = inspector
        self.lock = threading.RLock()
        self.scan_lock = threading.Lock()
        self.models = {}
        self.cache = {}
        self.scanning = False
        self.error = None
        self.closed = threading.Event()
        self.thread = None
        self.registered = []
        if self.path.exists():
            self.registered = json.loads(self.path.read_text()).get("paths", [])

    def add(self, value: str, *, persist=True) -> dict:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("请输入 checkpoint 本地路径")
        path = Path(value.strip()).expanduser().resolve(strict=True)
        if not path.is_file() or path.suffix.lower() != ".ckpt":
            raise ValueError("请选择本地 .ckpt 文件")
        key = str(path)
        identity = fingerprint(path)
        with self.scan_lock:
            cached = self.cache.get(key)
            if cached and cached[0] == identity:
                if isinstance(cached[1], str):
                    raise ValueError(cached[1])
                info = cached[1]
            else:
                try:
                    info = self.inspector(path)
                    if identity != fingerprint(path):
                        raise ValueError("checkpoint 校验期间发生变化，请等待文件写入完成后重试")
                except Exception as exc:
                    self.cache[key] = (identity, str(exc))
                    with self.lock:
                        self.models.pop(hashlib.sha256(key.encode()).hexdigest()[:20], None)
                    raise ValueError(str(exc)) from exc
                self.cache[key] = (identity, info)
            model = dict(
                info,
                id=hashlib.sha256(key.encode()).hexdigest()[:20],
                path=key,
                name=f"{path.parent.name} / {path.name}",
                fingerprint=identity,
                is_default=path == DEFAULT_CHECKPOINT.resolve(),
            )
            with self.lock:
                self.models[model["id"]] = model
                if persist and key not in self.registered:
                    self.registered.append(key)
                    atomic_json(self.path, {"paths": self.registered})
            return dict(model)

    def get(self, model_id) -> dict:
        if not isinstance(model_id, str):
            raise ValueError("model_id 必须是已注册的模型 ID")
        with self.lock:
            model = self.models.get(model_id)
        if model is None:
            raise ValueError("模型不存在或尚未通过校验")
        return self.add(model["path"], persist=False)

    def snapshot(self):
        with self.lock:
            return {
                "models": sorted(
                    self.models.values(),
                    key=lambda m: (not m["is_default"], -(m["global_step"] or 0), m["name"]),
                ),
                "scanning": self.scanning,
                "error": self.error,
            }

    def start(self):
        def scan():
            try:
                candidates = [DEFAULT_CHECKPOINT, *map(Path, self.registered)]
                for root in self.roots:
                    candidates.extend(sorted(root.rglob("*.ckpt")))
                for path in dict.fromkeys(candidates):
                    if self.closed.is_set():
                        break
                    try:
                        self.add(str(path), persist=False)
                    except (ValueError, OSError):
                        pass
            except Exception as exc:
                self.error = str(exc)
            finally:
                self.scanning = False

        self.scanning = True
        self.thread = threading.Thread(target=scan, name="checkpoint-discovery", daemon=True)
        self.thread.start()

    def close(self):
        self.closed.set()
        if self.thread:
            self.thread.join(timeout=5)
