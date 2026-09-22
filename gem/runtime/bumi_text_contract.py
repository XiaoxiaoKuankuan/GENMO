"""BUMI 文本模型的后端与资产契约。

将文本、序列策略、qpos30 表示、损失和机器人资产身份写入 checkpoint，推理读取
保存值而非当前 YAML。此模块不导入训练框架；部署、网页发现和训练共享同一校验。
旧音乐/93D/SMPL checkpoint 不会因为输出形状相似而被当作 BUMI 文本模型。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

SCHEMA = "genmo.bumi_text.v1"
REPRESENTATION = "genmo.bumi_motion_features.qpos30.v3"
MJCF_SHA256 = "fe93472dd764704fe8389b0f82052ae84ed8bc90f6d71b1467872f86e08a9ad3"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_contract(contract):
    from gem.utils.sequence_contract import normalize_sequence_contract

    if not isinstance(contract, dict) or contract.get("schema") != SCHEMA:
        raise ValueError("缺少 BUMI 文本模型契约；不能加载 SMPL、音乐或旧93D模型")
    expected = dict(
        motion_backend="bumi",
        representation=REPRESENTATION,
        feature_dim=30,
        contact_dim=2,
        qpos_dim=28,
        quaternion_convention="wxyz",
        fps=30,
        max_text_len=150,
        encoded_text_dim=1024,
        condition="text",
    )
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"BUMI 文本契约 {key} 必须为 {value!r}")
    names = contract.get("joint_names", [])
    if len(names) != 21 or len(set(names)) != 21:
        raise ValueError("需要21个不重复的有序关节名")
    sequence = normalize_sequence_contract(contract.get("sequence"))
    if sequence is None or (
        sequence["sequence_mode"] != "full" and sequence["schema_version"] != 2
    ):
        raise ValueError("BUMI文本接受旧full300或v2 crop120契约")
    if contract.get("loss_reduction") != "valid_per_sample":
        raise ValueError("BUMI 文本必须使用 valid_per_sample")
    for key in ("kinematics", "stats"):
        asset = contract.get("assets", {}).get(key, {})
        digest = asset.get("sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"无效 {key} 指纹")
    if contract.get("source_mjcf_sha256") != MJCF_SHA256:
        raise ValueError("本版 BUMI 文本要求 fe934 MJCF")
    return contract


def inspect_payload(payload):
    contract = validate_contract(payload.get("bumi_text_contract"))
    if payload.get("genmo_sequence_contract") != contract["sequence"]:
        raise ValueError("checkpoint 序列字段与 BUMI 契约矛盾")
    text = payload.get("genmo_text_contract", {})
    if (text.get("max_text_len"), text.get("encoded_text_dim"), text.get("text_only")) != (
        150,
        1024,
        True,
    ):
        raise ValueError("checkpoint 文本字段与 BUMI 契约矛盾")
    state = payload.get("state_dict", {})
    prefix = "pipeline.denoiser3d.denoiser."

    def shape(name):
        return tuple(getattr(state.get(prefix + name), "shape", ()))

    out = shape("final_layer.fc2.weight")
    contact = shape("static_conf_head.fc2.weight")
    if len(out) != 2 or out[0] != 30 or len(contact) != 2 or contact[0] != 2:
        raise ValueError("缺少30D运动和2D接触权重")
    if shape("add_cond_linear.weight") != (out[1], out[1] + 30):
        raise ValueError("扩散条件维度错误")
    if not any(k.startswith(prefix + "text_encoder_layers.") for k in state):
        raise ValueError("缺少文本交叉注意力权重")
    if any("music_embedder" in k or "pred_cam_head" in k for k in state):
        raise ValueError("BUMI 文本权重不能包含音乐或相机预测头")
    return contract


def resolve_assets(contract, *, kinematics=None, stats=None, checkpoint=None):
    validate_contract(contract)
    result = {}
    for name, override in (("kinematics", kinematics), ("stats", stats)):
        path = Path(override or contract["assets"][name]["path"]).expanduser()
        if override is None and not path.is_file() and checkpoint is not None:
            # 迁回本地后可以携带同SHA的assets，不编辑训练checkpoint或信任同名异物。
            directory = Path(checkpoint).expanduser().resolve().parent / "assets"
            candidates = [directory / path.name, directory / (name + ".json")]
            path = next(
                (
                    p
                    for p in candidates
                    if p.is_file() and sha256_file(p) == contract["assets"][name]["sha256"]
                ),
                path,
            )
        path = path.resolve(strict=True)
        if sha256_file(path) != contract["assets"][name]["sha256"]:
            raise ValueError(f"{name} 文件指纹与 checkpoint 不一致")
        result[name] = path
    return result
