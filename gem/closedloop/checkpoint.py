"""Stage 1 Actor 的可审计权重迁移和版本化 checkpoint。

旧 music-only checkpoint 只通过原 ``BumiMusicGEM`` 的 qpos30/contact2 表示校验后，
按三个明确的参数前缀迁移音乐编码、条件存在标记与原 Transformer/head。新增历史、前缀和
48 维物理尺度模块保留其初始化；缺失、未知键、非张量值和形状冲突均会报告并拒绝加载。
这里不调用旧 GEM 的宽松 load_state_dict，也不恢复 optimizer、scheduler 或 global_step。

新 checkpoint 记录 Actor 条件接口、完整配置、qpos30 stats/FK 文件 SHA 和权重。加载新
checkpoint 时同时核验接口及资产内容身份，随后使用 strict=True；跨机器路径可不同，文件
内容与关节顺序必须一致。旧 checkpoint 未内嵌资产身份时明确记录证据缺口，可传入已核验
的 source_assets 加强绑定，不能把表示维度匹配冒充原统计量已经匹配。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from gem.closedloop.contracts import STAGE1_CONTRACT_VERSION
from gem.robots.bumi.feature_codec import BUMI_REPRESENTATION_CONTRACT_VERSION

STAGE1_CHECKPOINT_VERSION = "genmo.bumi_closedloop.checkpoint.v1"
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

    from gem.closedloop.actor import STAGE1_ACTOR_INTERFACE_VERSION

    checkpoint = _read_checkpoint(source)
    expected = {
        "checkpoint_version": STAGE1_CHECKPOINT_VERSION,
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
    actor.load_state_dict(state, strict=True)
    return {"mode": "stage1_weights_only", "loaded": sorted(state), "global_step_restored": False}
