#!/usr/bin/env python3
"""Export a GENMO BUMI differentiable-kinematics JSON from the real MJCF.

Sole proxies must be supplied explicitly in a versioned proxy config.  This
tool never guesses BUMI link, foot, or geometry names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

KINEMATICS_CONTRACT_VERSION = "genmo.bumi_kinematics.v1"
PROXY_CONTRACT_VERSION = "genmo.bumi_proxy_config.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _name(model: mujoco.MjModel, object_type, object_id: int) -> str:
    value = mujoco.mj_id2name(model, object_type, int(object_id))
    if not value:
        raise ValueError(f"MuJoCo object {object_type}/{object_id} has no name")
    return str(value)


def _id(model: mujoco.MjModel, object_type, name: str) -> int:
    value = int(mujoco.mj_name2id(model, object_type, str(name)))
    if value < 0:
        raise ValueError(f"MuJoCo object {name!r} of type {object_type} does not exist")
    return value


def _actuated_joints(model: mujoco.MjModel) -> list[int]:
    joint_ids: list[int] = []
    for actuator_id in range(model.nu):
        if int(model.actuator_trntype[actuator_id]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            raise ValueError(f"Actuator {actuator_id} is not a joint transmission")
        joint_ids.append(int(model.actuator_trnid[actuator_id, 0]))
    if len(set(joint_ids)) != len(joint_ids):
        raise ValueError("BUMI actuators must map one-to-one to joints")
    return sorted(joint_ids, key=lambda joint_id: int(model.jnt_qposadr[joint_id]))


def _nearest_feature_ancestor(
    model: mujoco.MjModel,
    body_id: int,
    feature_body_ids: set[int],
    root_body_id: int,
) -> int:
    current = int(body_id)
    if current in feature_body_ids or current == root_body_id:
        return current
    parent = int(model.body_parentid[current])
    while parent not in feature_body_ids and parent != root_body_id:
        if parent == 0:
            body_name = _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            raise ValueError(f"Body {body_name!r} is outside the BUMI root subtree")
        parent = int(model.body_parentid[parent])
    return parent


def _relative_pose(
    data: mujoco.MjData, parent_body_id: int, child_body_id: int
) -> tuple[np.ndarray, np.ndarray]:
    parent_rotation = np.asarray(data.xmat[parent_body_id]).reshape(3, 3)
    child_rotation = np.asarray(data.xmat[child_body_id]).reshape(3, 3)
    position = parent_rotation.T @ (
        np.asarray(data.xpos[child_body_id]) - np.asarray(data.xpos[parent_body_id])
    )
    rotation = parent_rotation.T @ child_rotation
    quaternion = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return position, quaternion


def _read_proxy_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"BUMI proxy config does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("contract_version") != PROXY_CONTRACT_VERSION:
        raise ValueError(
            f"Proxy config {path} must declare contract_version={PROXY_CONTRACT_VERSION!r}"
        )
    proxies = value.get("sole_proxies")
    if not isinstance(proxies, list) or not proxies:
        raise ValueError(f"Proxy config {path} must contain explicit sole_proxies")
    feet = {str(item.get("foot")) for item in proxies if isinstance(item, dict)}
    if feet != {"left", "right"}:
        raise ValueError(f"Proxy config {path} must cover foot='left' and foot='right'")
    return value


def _export_proxy(
    item: dict[str, Any],
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    feature_body_ids: set[int],
    root_body_id: int,
    body_index: dict[int, int],
    is_sole: bool,
) -> dict[str, Any]:
    if not isinstance(item, dict) or not str(item.get("name", "")):
        raise ValueError("Every BUMI proxy must be an object with a non-empty name")
    if "body_name" in item:
        proxy_body_id = _id(model, mujoco.mjtObj.mjOBJ_BODY, str(item["body_name"]))
        feature_body_id = _nearest_feature_ancestor(
            model, proxy_body_id, feature_body_ids, root_body_id
        )
        position, quaternion = _relative_pose(data, feature_body_id, proxy_body_id)
        if "local_position_offset" in item:
            offset = np.asarray(item["local_position_offset"], dtype=np.float64)
            if offset.shape != (3,) or not np.isfinite(offset).all():
                raise ValueError(f"Invalid local_position_offset for proxy {item['name']!r}")
            proxy_rotation = np.zeros((3, 3), dtype=np.float64)
            mujoco.mju_quat2Mat(proxy_rotation.reshape(-1), quaternion)
            position = position + proxy_rotation @ offset
    else:
        required = {"feature_body_name", "local_position"}
        missing = required - set(item)
        if missing:
            raise ValueError(
                f"Proxy {item['name']!r} must define body_name or {sorted(required)}"
            )
        feature_body_id = _id(
            model, mujoco.mjtObj.mjOBJ_BODY, str(item["feature_body_name"])
        )
        if feature_body_id not in body_index:
            raise ValueError(
                f"Proxy {item['name']!r} feature_body_name is not an actuated feature body"
            )
        position = np.asarray(item["local_position"], dtype=np.float64)
        quaternion = np.asarray(item.get("local_quat_wxyz", [1.0, 0.0, 0.0, 0.0]))
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(f"Invalid local_position for proxy {item['name']!r}")
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
            raise ValueError(f"Invalid local_quat_wxyz for proxy {item['name']!r}")
        quaternion = quaternion / np.linalg.norm(quaternion)
    result: dict[str, Any] = {
        "name": str(item["name"]),
        "feature_body_name": _name(model, mujoco.mjtObj.mjOBJ_BODY, feature_body_id),
        "feature_body_index": int(body_index[feature_body_id]),
        "local_position": position.tolist(),
        "local_quat_wxyz": quaternion.tolist(),
    }
    if is_sole:
        if item.get("foot") not in {"left", "right"}:
            raise ValueError(f"Sole proxy {item['name']!r} requires foot='left' or 'right'")
        if "radius" in item:
            radius = float(item["radius"])
        elif "geom_name" in item:
            geom_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, str(item["geom_name"]))
            radius = float(model.geom_size[geom_id, 0])
        else:
            raise ValueError(
                f"Sole proxy {item['name']!r} requires explicit radius or real geom_name"
            )
        if not np.isfinite(radius) or radius < 0.0:
            raise ValueError(f"Invalid radius for sole proxy {item['name']!r}: {radius}")
        result["radius"] = radius
        result["foot_id"] = 0 if item["foot"] == "left" else 1
    return result


def export_bumi_spec(mjcf_path: Path, proxy_config_path: Path) -> dict[str, Any]:
    mjcf_path = mjcf_path.expanduser().resolve()
    proxy_config_path = proxy_config_path.expanduser().resolve()
    proxy_config = _read_proxy_config(proxy_config_path)
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    root_joint_ids = [
        joint_id
        for joint_id in range(model.njnt)
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if len(root_joint_ids) != 1:
        raise ValueError(f"Expected exactly one free root joint, got {root_joint_ids}")
    root_body_id = int(model.jnt_bodyid[root_joint_ids[0]])
    joint_ids = _actuated_joints(model)
    if model.nq != 28 or len(joint_ids) != 21:
        raise ValueError(f"Real BUMI MJCF must have nq=28 and 21 actuated joints, got {model.nq}/{len(joint_ids)}")
    addresses = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    if addresses != list(range(7, 28)):
        raise ValueError(f"BUMI actuated qpos addresses must be 7..27, got {addresses}")
    if any(not bool(model.jnt_limited[joint_id]) for joint_id in joint_ids):
        raise ValueError("Every BUMI actuated joint must declare a finite range in MJCF")
    child_body_ids = [int(model.jnt_bodyid[joint_id]) for joint_id in joint_ids]
    if len(set(child_body_ids)) != 21:
        raise ValueError("Each BUMI actuated joint must have a distinct child feature body")
    feature_body_ids = set(child_body_ids)
    body_ids = [root_body_id, *child_body_ids]
    body_index = {body_id: index for index, body_id in enumerate(body_ids)}

    zero_qpos = np.asarray(model.qpos0, dtype=np.float64).copy()
    zero_qpos[7:] = 0.0
    data = mujoco.MjData(model)
    data.qpos[:] = zero_qpos
    mujoco.mj_forward(model, data)
    parents: list[int] = []
    children: list[int] = []
    origins: list[list[float]] = []
    origin_quaternions: list[list[float]] = []
    for joint_index, child_body_id in enumerate(child_body_ids):
        parent_body_id = _nearest_feature_ancestor(
            model,
            int(model.body_parentid[child_body_id]),
            feature_body_ids,
            root_body_id,
        )
        if parent_body_id not in body_index:
            raise ValueError("Failed to fold fixed-body transform into a feature-body parent")
        parent_index = body_index[parent_body_id]
        if parent_index >= joint_index + 1:
            raise ValueError("MuJoCo qpos order is not topological for BUMI feature bodies")
        position, quaternion = _relative_pose(data, parent_body_id, child_body_id)
        parents.append(parent_index)
        children.append(joint_index + 1)
        origins.append(position.tolist())
        origin_quaternions.append(quaternion.tolist())

    sole_proxies = [
        _export_proxy(
            item,
            model=model,
            data=data,
            feature_body_ids=feature_body_ids,
            root_body_id=root_body_id,
            body_index=body_index,
            is_sole=True,
        )
        for item in proxy_config["sole_proxies"]
    ]
    evaluation_proxies = [
        _export_proxy(
            item,
            model=model,
            data=data,
            feature_body_ids=feature_body_ids,
            root_body_id=root_body_id,
            body_index=body_index,
            is_sole=False,
        )
        for item in proxy_config.get("evaluation_proxies", [])
    ]
    joint_names = [_name(model, mujoco.mjtObj.mjOBJ_JOINT, value) for value in joint_ids]
    feature_names = [_name(model, mujoco.mjtObj.mjOBJ_BODY, value) for value in child_body_ids]
    root_name = _name(model, mujoco.mjtObj.mjOBJ_BODY, root_body_id)
    body_order = [root_name, *feature_names]
    return {
        "contract_version": KINEMATICS_CONTRACT_VERSION,
        "robot_name": "bumi",
        "source_mjcf": mjcf_path.name,
        "source_mjcf_sha256": _sha256(mjcf_path),
        "proxy_config_sha256": _sha256(proxy_config_path),
        "root_body": root_name,
        "quaternion_convention": "wxyz",
        "qpos_order": "mujoco_native",
        "qpos_dim": 28,
        "joint_dim": 21,
        "qpos_layout": "root_xyz_3 + root_quaternion_wxyz_4 + joint_dof_21",
        "body_order": body_order,
        "feature_body_names": feature_names,
        "body_name_to_index": {name: index for index, name in enumerate(body_order)},
        "joint_order": joint_names,
        "joint_name_to_qpos_index": {
            name: index for index, name in enumerate(joint_names)
        },
        "joint_name_to_qpos_address": {
            name: address for name, address in zip(joint_names, addresses, strict=True)
        },
        "joint_qpos_addresses": addresses,
        "parent_body_indices": parents,
        "child_body_indices": children,
        "joint_axes": [np.asarray(model.jnt_axis[value]).tolist() for value in joint_ids],
        "joint_origin_xyz": origins,
        "joint_origin_quat_wxyz": origin_quaternions,
        "joint_anchor_xyz": [np.asarray(model.jnt_pos[value]).tolist() for value in joint_ids],
        "joint_lower_limits": [float(model.jnt_range[value, 0]) for value in joint_ids],
        "joint_upper_limits": [float(model.jnt_range[value, 1]) for value in joint_ids],
        "default_qpos": np.asarray(model.qpos0).tolist(),
        "sole_proxies": sole_proxies,
        "evaluation_proxies": evaluation_proxies,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", required=True, type=Path)
    parser.add_argument("--proxy-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    spec = export_bumi_spec(args.mjcf, args.proxy_config)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "qpos_dim": spec["qpos_dim"],
                "joints": len(spec["joint_order"]),
                "feature_bodies": len(spec["feature_body_names"]),
                "sole_proxies": len(spec["sole_proxies"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
