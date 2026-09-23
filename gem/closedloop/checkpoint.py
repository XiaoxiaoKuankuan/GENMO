"""Stage 1 Actor 的可审计权重迁移和版本化 checkpoint。

旧 music-only checkpoint 只通过原 ``BumiMusicGEM`` 的 qpos30/contact2 表示校验后，
按三个明确的参数前缀迁移音乐编码、条件存在标记与原 Transformer/head。新增历史、前缀和
48 维物理尺度模块保留其初始化；缺失、未知键、非张量值和形状冲突均会报告并拒绝加载。
旧权重迁移不调用旧 GEM 的宽松 load_state_dict，也不恢复 optimizer、scheduler 或 global_step。

新 checkpoint 记录 Actor 条件接口、完整配置、qpos30 stats/FK 文件 SHA 和权重。加载新
checkpoint 时同时核验接口及资产内容身份，随后使用 strict=True；跨机器路径可不同，文件
内容与关节顺序必须一致。旧 checkpoint 未内嵌资产身份时明确记录证据缺口，可传入已核验
的 source_assets 加强绑定，不能把表示维度匹配冒充原统计量已经匹配。

第5步另设真正训练 checkpoint：保存 optimizer 参数名布局、scheduler/AMP 状态、更新步数
以及训练器提供的各 rank 随机状态和采样游标。恢复前核验接口、资产、全部模型参数、优化器
类型与参数布局以及调用方指定的数据/采样契约；旧 weights-only 文件始终不能用于 resume。
训练产物先在同目录临时文件写完并 fsync，再以硬链接原子发布，拒绝覆盖任何已有编号文件。
本模块负责状态持久化与优化器等状态恢复；各 rank RNG/采样游标由训练器根据返回值恢复。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from gem.closedloop.contracts import STAGE1_CONTRACT_VERSION
from gem.robots.bumi.feature_codec import BUMI_REPRESENTATION_CONTRACT_VERSION

STAGE1_CHECKPOINT_VERSION = "genmo.bumi_closedloop.checkpoint.v1"
STAGE1_TRAINING_CHECKPOINT_VERSION = "genmo.bumi_closedloop.training_checkpoint.v1"
_LEGACY_PREFIXES = {
    "pipeline.denoiser3d.denoiser.": "denoiser.",
    "music_embedder.": "music_embedder.",
    "cond_exists_embedder.encoded_music.": "cond_exists_embedder.",
}
_NEW_PREFIXES = ("history_encoder.", "prefix_encoder.", "proprio_normalizer.")


def _plain_config(value: Any) -> Any:
    """将 OmegaConf 等映射容器转成可审计、可序列化的普通 Python 配置。"""

    if isinstance(value, Mapping):
        return {str(key): _plain_config(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain_config(item) for item in value]
    return value


class CheckpointCompatibilityError(RuntimeError):
    """携带完整参数审计表的拒绝加载异常，失败前不会修改 Actor。"""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        self.report = report
        super().__init__(f"{message}: {json.dumps(report, ensure_ascii=False)}")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def actor_asset_identity(actor: nn.Module) -> dict[str, Any]:
    """记录当前实际使用的 stats、FK、关节顺序及归一化下限。"""

    endecoder = actor.endecoder
    kinematics = endecoder.kinematics
    return {
        "stats_path": str(Path(endecoder.stats_path).resolve()),
        "stats_sha256": _sha256(endecoder.stats_path),
        "kinematics_path": str(Path(kinematics.kinematics_path).resolve()),
        "kinematics_sha256": _sha256(kinematics.kinematics_path),
        "joint_order": list(kinematics.joint_order),
        "clip_std_min": float(endecoder.clip_std_min),
        "representation_contract_version": BUMI_REPRESENTATION_CONTRACT_VERSION,
    }


def _asset_content(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in identity.items() if not key.endswith("_path")}


def _read_checkpoint(source: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    # 原 Lightning checkpoint 含配置对象；调用方必须使用自己信任的本地训练产物。
    checkpoint = torch.load(Path(source), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must contain a mapping")
    return checkpoint


def warm_start_weights(
    actor: nn.Module,
    source: str | Path | Mapping[str, Any],
    *,
    source_assets: Mapping[str, Any] | None = None,
    source_model_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """只迁移匹配旧 qpos30/contact2 权重，并返回每个参数的明确分类。"""

    from gem.bumi_gem import BumiMusicGEM

    checkpoint = _read_checkpoint(source)
    BumiMusicGEM._validate_representation_checkpoint(checkpoint)
    current = actor.state_dict()
    report: dict[str, Any] = {
        "mode": "weights_only_warm_start",
        "loaded": [],
        "new": sorted(key for key in current if key.startswith(_NEW_PREFIXES)),
        "missing": [],
        "shape_conflicts": [],
        "unexpected": [],
        "optimizer_restored": False,
        "global_step_restored": False,
        "source_global_step_for_provenance_only": checkpoint.get("global_step"),
        "source_asset_identity": "not_embedded_in_legacy_checkpoint",
        "source_model_configuration": "not_embedded_in_legacy_checkpoint",
    }
    mapped: dict[str, torch.Tensor] = {}
    for original, value in checkpoint["state_dict"].items():
        key = str(original)
        destination = next(
            (
                replacement + key[len(prefix) :]
                for prefix, replacement in _LEGACY_PREFIXES.items()
                if key.startswith(prefix)
            ),
            None,
        )
        if destination is None or destination not in current or destination in mapped:
            report["unexpected"].append(key)
            continue
        if not isinstance(value, torch.Tensor) or value.shape != current[destination].shape:
            report["shape_conflicts"].append(
                {
                    "source": key,
                    "destination": destination,
                    "checkpoint": list(value.shape) if isinstance(value, torch.Tensor) else None,
                    "actor": list(current[destination].shape),
                }
            )
            continue
        mapped[destination] = value
        report["loaded"].append({"source": key, "destination": destination})
    report["missing"] = sorted(set(current) - set(mapped) - set(report["new"]))
    if source_assets is None:
        source_assets = checkpoint.get("asset_identity")
    if source_assets is not None:
        if not isinstance(source_assets, Mapping):
            raise CheckpointCompatibilityError("source asset identity must be a mapping", report)
        report["source_asset_identity"] = "verified_exact_content"
        if _asset_content(source_assets) != _asset_content(actor_asset_identity(actor)):
            report["source_asset_identity"] = "mismatch"
            raise CheckpointCompatibilityError("warm start assets differ", report)
    if source_model_config is not None:
        allowed = {"backbone", "noise_schedule", "diffusion_steps", "condition_modules"}
        if (
            not isinstance(source_model_config, Mapping)
            or not source_model_config
            or set(source_model_config) - allowed
        ):
            raise CheckpointCompatibilityError(
                "source_model_config must explicitly describe Actor backbone/diffusion/condition modules",
                report,
            )
        source_model_config = _plain_config(source_model_config)
        for key, value in source_model_config.items():
            if value != actor.interface_config.get(key):
                report["source_model_configuration"] = {
                    "status": "mismatch",
                    "field": key,
                    "source": value,
                    "actor": actor.interface_config.get(key),
                }
                raise CheckpointCompatibilityError("warm start model configuration differs", report)
        report["source_model_configuration"] = {
            "status": "verified_explicit_fields",
            "fields": sorted(source_model_config),
        }
    if report["missing"] or report["shape_conflicts"] or report["unexpected"]:
        raise CheckpointCompatibilityError("strict warm start rejected", report)
    # 完整映射先验收再提交，新增分支原值明确保留；不会静默忽略任何非预期参数。
    actor.load_state_dict({**current, **mapped}, strict=True)
    return report


def save_stage1_checkpoint(
    actor: nn.Module,
    path: str | Path,
    *,
    config: Mapping[str, Any],
    global_step: int,
    warm_start_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """保存独立 Actor 权重及来源；不包含可被误当续训的旧 optimizer 状态。"""

    from gem.closedloop.actor import STAGE1_ACTOR_INTERFACE_VERSION

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    payload = {
        "checkpoint_version": STAGE1_CHECKPOINT_VERSION,
        "condition_interface_version": STAGE1_ACTOR_INTERFACE_VERSION,
        "stage1_contract_version": STAGE1_CONTRACT_VERSION,
        "bumi_representation_contract_version": BUMI_REPRESENTATION_CONTRACT_VERSION,
        "actor_interface_config": dict(actor.interface_config),
        "config": dict(config),
        "asset_identity": actor_asset_identity(actor),
        "global_step": int(global_step),
        "state_dict": {key: value.detach().cpu() for key, value in actor.state_dict().items()},
        "warm_start_report": dict(warm_start_report) if warm_start_report else None,
        "training_state": "actor_weights_only_not_resumable_optimizer_state",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    # 独占创建避免检查后意外覆盖另一进程刚生成的训练产物。
    with destination.open("xb") as handle:
        torch.save(payload, handle)
    return payload


def load_stage1_checkpoint(
    actor: nn.Module, source: str | Path | Mapping[str, Any]
) -> dict[str, Any]:
    """验证 Stage 1 接口、资产、参数全集并只加载 Actor 权重。"""

    checkpoint = _read_checkpoint(source)
    _validate_actor_checkpoint(actor, checkpoint)
    actor.load_state_dict(checkpoint["state_dict"], strict=True)
    return {
        "mode": "stage1_weights_only",
        "loaded": sorted(checkpoint["state_dict"]),
        "global_step_restored": False,
    }


def _validate_actor_checkpoint(actor: nn.Module, checkpoint: Mapping[str, Any]) -> None:
    """只检查、尚不写入模型；让权重加载与完整续训共用同一严格 Actor 契约。"""

    from gem.closedloop.actor import STAGE1_ACTOR_INTERFACE_VERSION

    if checkpoint.get("checkpoint_version") not in (
        STAGE1_CHECKPOINT_VERSION,
        STAGE1_TRAINING_CHECKPOINT_VERSION,
    ):
        raise RuntimeError("Stage 1 checkpoint checkpoint_version mismatch")
    expected = {
        "condition_interface_version": STAGE1_ACTOR_INTERFACE_VERSION,
        "stage1_contract_version": STAGE1_CONTRACT_VERSION,
        "bumi_representation_contract_version": BUMI_REPRESENTATION_CONTRACT_VERSION,
        "actor_interface_config": dict(actor.interface_config),
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"Stage 1 checkpoint {key} mismatch")
    assets = checkpoint.get("asset_identity")
    if not isinstance(assets, Mapping) or _asset_content(assets) != _asset_content(
        actor_asset_identity(actor)
    ):
        raise RuntimeError("Stage 1 checkpoint asset identity mismatch")
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("Stage 1 checkpoint is missing state_dict")
    current = actor.state_dict()
    missing = sorted(set(current) - set(state))
    unexpected = sorted(set(state) - set(current))
    conflicts = sorted(
        key
        for key in set(current) & set(state)
        if not isinstance(state[key], torch.Tensor) or state[key].shape != current[key].shape
    )
    if missing or unexpected or conflicts:
        raise RuntimeError(
            f"Stage 1 state mismatch: missing={missing}, unexpected={unexpected}, shape_conflicts={conflicts}"
        )


def _cpu_state(value: Any) -> Any:
    """递归快照状态，避免保存 payload 与继续训练的 CPU Tensor 共用可变内存。"""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_state(item) for item in value)
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    return value


def _class_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _optimizer_layout(actor: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """用 Actor 参数名绑定 optimizer 顺序，避免形状相同的层在续训时串接动量。"""

    named = {id(parameter): name for name, parameter in actor.named_parameters()}
    seen: set[int] = set()
    groups = []
    for group in optimizer.param_groups:
        entries = []
        for parameter in group["params"]:
            if id(parameter) not in named or id(parameter) in seen:
                raise RuntimeError("optimizer parameters must belong to Actor without duplicates")
            seen.add(id(parameter))
            entries.append(
                {
                    "name": named[id(parameter)],
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                    "requires_grad": parameter.requires_grad,
                }
            )
        groups.append(entries)
    return {"class": _class_name(optimizer), "parameter_groups": groups}


def _component_state(component: Any) -> dict[str, Any] | None:
    if component is None:
        return None
    return {"class": _class_name(component), "state_dict": _cpu_state(component.state_dict())}


def _validate_component(name: str, component: Any, saved: Any) -> None:
    """不用 scheduler/scaler 时必须双向为 None，不把缺失状态当成默认初始化。"""

    if component is None:
        if saved is not None:
            raise RuntimeError(f"resume {name} presence mismatch")
        return
    if not isinstance(saved, Mapping) or saved.get("class") != _class_name(component):
        raise RuntimeError(f"resume {name} class/presence mismatch")
    state = saved.get("state_dict")
    current = component.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(current):
        raise RuntimeError(f"resume {name} state structure mismatch")
    for key, reference in current.items():
        item = state[key]
        if isinstance(reference, torch.Tensor):
            if not isinstance(item, torch.Tensor) or item.shape != reference.shape:
                raise RuntimeError(f"resume {name} tensor mismatch: {key}")
        elif isinstance(reference, (list, tuple)):
            if not isinstance(item, (list, tuple)) or len(item) != len(reference):
                raise RuntimeError(f"resume {name} sequence mismatch: {key}")
        elif type(item) is not type(reference):
            raise RuntimeError(f"resume {name} state type mismatch: {key}")


def _validate_optimizer_state(optimizer: torch.optim.Optimizer, state: Any) -> None:
    if not isinstance(state, Mapping) or set(state) != {"state", "param_groups"}:
        raise RuntimeError("resume optimizer state structure mismatch")
    groups, values = state["param_groups"], state["state"]
    if not isinstance(groups, list) or len(groups) != len(optimizer.param_groups):
        raise RuntimeError("resume optimizer parameter groups mismatch")
    if not isinstance(values, Mapping):
        raise RuntimeError("resume optimizer state must be a mapping")
    indexed = {}
    adam_required = {}
    for saved, current in zip(groups, optimizer.param_groups):
        if not isinstance(saved, Mapping) or set(saved) != set(current):
            raise RuntimeError("resume optimizer group keys mismatch")
        identifiers = saved["params"]
        if not isinstance(identifiers, list) or len(identifiers) != len(current["params"]):
            raise RuntimeError("resume optimizer parameter count mismatch")
        for key, parameter in zip(identifiers, current["params"]):
            if not isinstance(key, int) or key in indexed:
                raise RuntimeError("resume optimizer parameter identifiers invalid")
            indexed[key] = parameter
            if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
                adam_required[key] = {"step", "exp_avg", "exp_avg_sq"}
                if saved.get("amsgrad"):
                    adam_required[key].add("max_exp_avg_sq")
    if set(values) - set(indexed):
        raise RuntimeError("resume optimizer contains unbound parameter state")
    for index, parameter_state in values.items():
        if not isinstance(parameter_state, Mapping):
            raise RuntimeError("resume optimizer parameter state must be a mapping")
        if index in adam_required and set(parameter_state) != adam_required[index]:
            raise RuntimeError(f"resume optimizer Adam state keys mismatch: {index}")
        for name, item in parameter_state.items():
            if isinstance(item, torch.Tensor):
                # Adam 的 step 是标量；一阶/二阶动量必须与绑定参数逐元素对应。
                if name == "step" and item.numel() == 1:
                    continue
                if item.shape != indexed[index].shape:
                    raise RuntimeError(f"resume optimizer tensor shape mismatch: {index}/{name}")


def _validate_runtime_state(runtime: Any, global_step: int) -> None:
    """检查每个 rank 的随机状态和精确采样位置，缺失任一 rank 均拒绝续训。"""

    required = {
        "world_size", "batch_size", "gradient_accumulation", "precision", "data_fingerprint",
        "sampling", "training_contract", "rank_states", "completed_steps",
    }
    if not isinstance(runtime, Mapping) or required - set(runtime):
        raise RuntimeError("resume runtime_state is missing required training/sampling/RNG fields")
    for key in ("world_size", "batch_size", "gradient_accumulation"):
        if type(runtime[key]) is not int or runtime[key] < 1:
            raise RuntimeError(f"resume runtime positive integer required: {key}")
    if type(runtime["completed_steps"]) is not int or runtime["completed_steps"] != global_step:
        raise RuntimeError("resume runtime completed_steps/global_step mismatch")
    if not isinstance(runtime["precision"], str) or not runtime["precision"]:
        raise RuntimeError("resume runtime precision missing")
    if not isinstance(runtime["data_fingerprint"], (str, Mapping)) or not runtime["data_fingerprint"]:
        raise RuntimeError("resume runtime data_fingerprint missing")
    for key in ("sampling", "training_contract"):
        if not isinstance(runtime[key], Mapping) or not runtime[key]:
            raise RuntimeError(f"resume runtime {key} missing")
    ranks = runtime["rank_states"]
    if not isinstance(ranks, list) or len(ranks) != runtime["world_size"]:
        raise RuntimeError("resume rank_states/world_size mismatch")
    for rank, saved in enumerate(ranks):
        if not isinstance(saved, Mapping) or saved.get("rank") != rank:
            raise RuntimeError("resume rank_states must cover ordered ranks 0..world_size-1")
        for key in ("sampler_epoch", "sampler_offset"):
            if type(saved.get(key)) is not int or saved[key] < 0:
                raise RuntimeError(f"resume rank {rank} {key} invalid")
        rng = saved.get("rng")
        if not isinstance(rng, Mapping) or set(rng) != {"python", "numpy", "torch", "cuda"}:
            raise RuntimeError(f"resume rank {rank} required RNG states missing")
        if not isinstance(rng["python"], tuple) or len(rng["python"]) != 3:
            raise RuntimeError(f"resume rank {rank} Python RNG invalid")
        if not isinstance(rng["numpy"], tuple) or len(rng["numpy"]) != 5:
            raise RuntimeError(f"resume rank {rank} NumPy RNG invalid")
        if not isinstance(rng["cuda"], (tuple, list)):
            raise RuntimeError(f"resume rank {rank} CUDA RNG list invalid")
        for value in (rng["torch"], *rng["cuda"]):
            if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
                raise RuntimeError(f"resume rank {rank} torch/CUDA RNG tensor invalid")
        # 在独立生成器上验收，不改变本进程 RNG；避免加载模型后才发现随机状态不可恢复。
        try:
            random.Random().setstate(rng["python"])
            np.random.RandomState().set_state(rng["numpy"])
            torch.Generator(device="cpu").set_state(rng["torch"].cpu())
        except (TypeError, ValueError, RuntimeError) as error:
            raise RuntimeError(f"resume rank {rank} RNG state is not restorable") from error


def _atomic_save_checkpoint(payload: Mapping[str, Any], destination: Path) -> None:
    """同目录写完才发布；硬链接原子创建可同时避免半文件与检查后的覆盖竞争。"""

    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_training_checkpoint(
    actor: nn.Module,
    path: str | Path,
    *,
    config: Mapping[str, Any],
    global_step: int,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    scaler: Any = None,
    runtime_state: Mapping[str, Any],
    warm_start_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """保存真正可续训的完整状态；仅由训练器 rank0 调用，步数表示已完成的更新。"""

    from gem.closedloop.actor import STAGE1_ACTOR_INTERFACE_VERSION

    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
        raise ValueError("global_step must be a nonnegative integer")
    _validate_runtime_state(runtime_state, global_step)
    payload = {
        "checkpoint_version": STAGE1_TRAINING_CHECKPOINT_VERSION,
        "condition_interface_version": STAGE1_ACTOR_INTERFACE_VERSION,
        "stage1_contract_version": STAGE1_CONTRACT_VERSION,
        "bumi_representation_contract_version": BUMI_REPRESENTATION_CONTRACT_VERSION,
        "actor_interface_config": _plain_config(actor.interface_config),
        "config": _plain_config(config),
        "asset_identity": actor_asset_identity(actor),
        "global_step": global_step,
        "state_dict": _cpu_state(actor.state_dict()),
        "optimizer_layout": _optimizer_layout(actor, optimizer),
        "optimizer_state_dict": _cpu_state(optimizer.state_dict()),
        "scheduler": _component_state(scheduler),
        "scaler": _component_state(scaler),
        "runtime_state": _cpu_state(runtime_state),
        "warm_start_report": _plain_config(warm_start_report),
        "training_state": "full_stage1_training_state",
    }
    _atomic_save_checkpoint(payload, Path(path))
    return payload


def load_training_checkpoint(
    actor: nn.Module,
    path: str | Path | Mapping[str, Any],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    scaler: Any = None,
    expected_runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """严格恢复训练状态，返回 RNG/采样游标交训练器按 rank 恢复；不降级为 warm start。"""

    checkpoint = _read_checkpoint(path)
    if (
        checkpoint.get("checkpoint_version") != STAGE1_TRAINING_CHECKPOINT_VERSION
        or checkpoint.get("training_state") != "full_stage1_training_state"
    ):
        raise RuntimeError("resume requires a full Stage 1 training checkpoint; weights-only is not resumable")
    _validate_actor_checkpoint(actor, checkpoint)
    current = actor.state_dict()
    dtype_conflicts = [
        key for key, value in checkpoint["state_dict"].items()
        if value.dtype != current[key].dtype or value.layout != current[key].layout
    ]
    if dtype_conflicts:
        raise RuntimeError(f"resume model tensor dtype/layout mismatch: {dtype_conflicts}")
    step = checkpoint.get("global_step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise RuntimeError("resume checkpoint global_step invalid")
    if checkpoint.get("optimizer_layout") != _optimizer_layout(actor, optimizer):
        raise RuntimeError("resume optimizer parameter layout/class mismatch")
    _validate_optimizer_state(optimizer, checkpoint.get("optimizer_state_dict"))
    for name, component in (("scheduler", scheduler), ("scaler", scaler)):
        if name not in checkpoint:
            raise RuntimeError(f"resume checkpoint missing explicit {name} state")
        _validate_component(name, component, checkpoint[name])
    runtime = checkpoint.get("runtime_state")
    _validate_runtime_state(runtime, step)
    checked = []
    for key, expected in (expected_runtime or {}).items():
        if key not in runtime or _plain_config(runtime[key]) != _plain_config(expected):
            raise RuntimeError(f"resume runtime contract mismatch: {key}")
        checked.append(key)
    # 上述校验全部成功后才修改任何组件，避免常见配置错误导致半恢复状态。
    actor.load_state_dict(checkpoint["state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"]["state_dict"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"]["state_dict"])
    return {
        "mode": "stage1_training_resume",
        "global_step": step,
        "global_step_restored": True,
        "model_restored": True,
        "optimizer_restored": True,
        "scheduler_restored": scheduler is not None,
        "scaler_restored": scaler is not None,
        "scheduler_used": scheduler is not None,
        "scaler_used": scaler is not None,
        "runtime_state": runtime,
        "runtime_contract_checked": sorted(checked),
        "runtime_rng_and_sampler_restored_by": "caller_per_rank",
        "warm_start_report": checkpoint.get("warm_start_report"),
        "loaded": sorted(checkpoint["state_dict"]),
    }
