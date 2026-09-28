#!/usr/bin/env python3
"""依据原 CSV 和已记录偏移恢复 Mine 发布库，保留所有可追溯旧版本。

本工具只处理明确标为 legacy_body_origin_min_zero 的 Mine 数据。逐条核对原 CSV
身份、时间轴和旧偏移的逆变换，直接以相同 30 Hz 时间点的原始 root Z 替换发布值；
其余 qpos 坐标、动作长度、关节顺序、音频、音乐特征和分集保持不变。不会通过足底
最低值猜测平移，不会重定向，不会以新资产改写旧关节。现代资产可重新执行原音乐
质量门禁；旧资产只恢复有证据的坐标变换，并明确保留旧质量判断的来源。

动作与元数据在同盘 staging 中更新，重算固定世界 z=0 的接触标签并通过 reader。
发布时将旧目录移入具名备份，再替换为已验证的新目录；失败时恢复原入口。备份保存
旧质量报告与训练统计的依据，不删除历史 checkpoint，也不宣称旧统计适用于新数据。
对同一已恢复数据重复调用仅验证原 CSV 高度，不再次施加偏移。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from gem.datasets.music_dance.music_dance_bumi import BumiMusicDatasetReader
from gem.robots.bumi.contacts import BUMI_CONTACT_CONTRACT_VERSION, derive_bumi_foot_contact
from gem.robots.bumi.kinematics import BumiKinematics
from gem.robots.bumi.motion_utils import sha256_file
from tools.data.bumi.filter_robot_retargeter_npz_motions import load_config
from tools.data.bumi.npz_quality_utils import evaluate_motion
from tools.data.bumi.umr_qpos_adapter import qpos_arrays

SEMANTICS = "source_csv_root_z_preserved_v1"
VERSION = "genmo.mine_restore_csv_root_z.v1"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    pending.replace(path)  # staging 可以包含硬链接，绝不原位改写。


def raw_index(root):
    result = {}
    for part in ("dance_2", "dance_3"):
        for path in sorted((Path(root) / (part + "_csv")).glob("bumi_*fps.csv")):
            song, fps = path.stem.removeprefix("bumi_").rsplit("_", 1)
            key = part + "__" + song
            if key in result:
                raise ValueError(f"重复原始动作: {key}")
            result[key] = (path, int(fps.removesuffix("fps")))
    if not result:
        raise ValueError("没有找到原始 Mine CSV")
    return result


def original_z(path, fps, frames):
    csv = np.loadtxt(path, delimiter=",", skiprows=1)
    if csv.ndim != 2 or csv.shape[1] != 28 or not np.isfinite(csv).all():
        raise ValueError(f"原 CSV 格式异常: {path}")
    time = np.arange(frames, dtype=np.float64) / 30
    if time[-1] > len(csv) / fps:
        raise ValueError("恢复时间轴超出原 CSV")
    return torch.from_numpy(np.interp(time, np.arange(len(csv)) / fps, csv[:, 2]).astype(np.float32))


def restore_tensor(qpos, z, recorded_offset):
    """只允许有原 CSV 证据的常量偏移逆操作；拒绝用任意新高度掩盖源变化。"""
    error = float((qpos[:, 2] - float(recorded_offset) - z).abs().max())
    if error > 2e-6:
        raise ValueError(f"原 CSV 与记录偏移不匹配: {error} m")
    restored = qpos.clone()
    restored[:, 2] = z
    return restored, error


def restore(root, raw_root, kin_path, offset_reference=None, quality_config=None):
    root = Path(root).resolve(strict=True)
    info = json.loads((root / "meta/dataset_info.json").read_text())
    if info.get("source_dataset") != "mine" or info["ground_semantics"] not in {SEMANTICS, "legacy_body_origin_min_zero"}:
        raise ValueError(f"不是明确支持的 Mine 发布: {root}")
    raw = raw_index(raw_root)
    kin = BumiKinematics(kin_path)
    if kin.source_mjcf_sha256 != info["source_mjcf_sha256"] or list(kin.joint_order) != info["joint_names"]:
        raise ValueError("恢复必须使用该发布自身的资产和关节顺序")
    quality = load_config(quality_config) if quality_config else None
    if quality and quality.kinematics_sha256 != kin.kinematics_sha256:
        raise ValueError("原音乐质量规则与本次资产不匹配")
    manifests = {p.name: [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
                 for p in sorted((root / "manifests").glob("*.jsonl"))}
    rows = [r for values in manifests.values() for r in values]
    if len({r["motion_path"] for r in rows}) != len(rows):
        raise ValueError("分集中存在重复动作")
    backup = root.with_name(root.name + ".legacy_height_backup_20260928")
    if info["ground_semantics"] == SEMANTICS:
        for row in rows:
            p = torch.load(root / row["motion_path"], map_location="cpu", weights_only=False)
            csv, fps = raw[Path(row["motion_path"]).stem]
            if not torch.equal(p["qpos"][:, 2], original_z(csv, fps, len(p["qpos"]))):
                raise ValueError("已恢复发布与原 CSV 高度不一致")
        return {"status": "ALREADY_RESTORED", "root": str(root), "motions": len(rows)}
    if backup.exists():
        raise FileExistsError(f"拒绝覆盖历史备份: {backup}")
    staging = Path(tempfile.mkdtemp(prefix="." + root.name + ".restore-", dir=root.parent))
    old_info = dict(info)
    records = []
    try:
        shutil.copytree(root, staging, dirs_exist_ok=True, copy_function=os.link, symlinks=True)
        for row in rows:
            relative = row["motion_path"]
            old_path = root / relative
            payload = torch.load(old_path, map_location="cpu", weights_only=False)
            key = Path(relative).stem
            csv, fps = raw[key]
            raw_sha = sha256_file(csv)
            reference = payload
            if "root_z_adjustment_m" not in reference:
                if offset_reference is None:
                    raise ValueError(f"{key}: 缺少原始偏移证据")
                reference = torch.load(Path(offset_reference) / relative, map_location="cpu", weights_only=False)
                reference_digests = {sha256_file(Path(offset_reference) / relative),
                                     reference.get("height_restoration", {}).get("legacy_motion_sha256")}
                if payload["source_motion_sha256"] not in reference_digests:
                    raise ValueError(f"{key}: 上游偏移参考 SHA 不匹配")
            if reference["source_motion_sha256"] != raw_sha:
                raise ValueError(f"{key}: 原 CSV SHA 与构建记录不符")
            offset = reference.get("height_restoration", {}).get("legacy_root_z_adjustment_m", reference["root_z_adjustment_m"])
            qpos = torch.as_tensor(payload["qpos"]).detach().cpu().float()
            z = original_z(csv, fps, len(qpos))
            restored, error = restore_tensor(qpos, z, offset)
            decision = evaluate_motion(qpos_arrays(restored, kin, quality), quality) if quality else None
            if decision and decision["status"] != "PASS":
                raise ValueError(f"{key}: 恢复后的原音乐门禁未通过: {decision['reason_codes']}")
            evidence = {"version": VERSION, "raw_csv": str(csv), "raw_csv_sha256": raw_sha,
                        "legacy_motion_sha256": sha256_file(old_path), "legacy_root_z_adjustment_m": float(offset),
                        "inverse_offset_max_error_m": error, "original_source_motion_sha256": payload["source_motion_sha256"],
                        "original_quality_report_sha256": payload.get("quality_report_sha256"),
                        "non_z_coordinates_unchanged": True, "original_root_z_exact": True}
            payload.update(qpos=restored, root_z_adjustment_m=0.0, root_z_adjusted=False,
                           ground_semantics=SEMANTICS, height_restoration=evidence,
                           source_motion_sha256=raw_sha)
            with torch.no_grad():
                ground = float(kin.forward_kinematics(restored)["body_pos_w"][..., 2].min())
            payload.update(body_origin_ground_before_adjustment_m=ground, body_origin_ground_after_adjustment_m=ground)
            if "foot_contact" in payload:
                contact = derive_bumi_foot_contact(restored, kin, fps=30,
                    valid_mask=torch.ones(len(restored), dtype=torch.bool), ground_height=0.0,
                    estimate_ground_mask=torch.tensor(False))
                payload.update(foot_contact=contact.contact.contiguous(),
                    foot_contact_contract_version=BUMI_CONTACT_CONTRACT_VERSION,
                    foot_contact_ground_height_m=0.0, foot_contact_source="derived_from_full_qpos_fk_fixed_world_zero")
            if decision:
                payload["quality_config_sha256"] = sha256_file(quality_config)
                payload["retarget_quality"] = {"status": "PASS", "reason_codes": decision["reason_codes"]}
            row["source_motion_sha256"] = raw_sha
            destination = staging / relative
            pending = destination.with_name(destination.name + ".pending")
            torch.save(payload, pending)
            pending.replace(destination)
            records.append({"sample_id": key, "motion_path": relative, "frames": len(qpos), **evidence,
                            "quality_recheck": decision,
                            "quality_scope": "original_music_gate_recomputed" if decision else "historical_gate_with_verified_inverse_translation"})
        quality_path = staging / "reports/root_z_restoration_quality.json"
        write_json(quality_path, {"version": VERSION, "records": records})
        quality_sha = sha256_file(quality_path)
        for row in rows:
            path = staging / row["motion_path"]
            p = torch.load(path, map_location="cpu", weights_only=False)
            p["quality_report_sha256"] = quality_sha
            pending = path.with_name(path.name + ".pending")
            torch.save(p, pending)
            pending.replace(path)
        for filename, values in manifests.items():
            path = staging / "manifests" / filename
            pending = path.with_name(path.name + ".pending")
            pending.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in values))
            pending.replace(path)
        info.update(ground_semantics=SEMANTICS, root_z_adjusted=False, quality_report_sha256=quality_sha,
                    height_restoration_version=VERSION, legacy_backup_root=str(backup))
        if quality:
            info["quality_config_sha256"] = sha256_file(quality_config)
        write_json(staging / "meta/dataset_info.json", info)
        for filename, values in manifests.items():
            if values:
                BumiMusicDatasetReader(staging, info["dataset_name"], Path(filename).stem, kin,
                    joint_limit_tolerance=info.get("reader_joint_limit_tolerance_rad", 0.25),
                    validate_payloads_on_init=True, validate_source_hashes_on_init=True)
        report = {"status": "PASS", "version": VERSION, "root": str(root), "backup": str(backup),
                  "motions": len(rows), "frames": sum(r["frames"] for r in records),
                  "old_dataset_info": old_info, "new_ground_semantics": SEMANTICS,
                  "all_original_root_z_exact": True, "non_z_qpos_unchanged": True,
                  "stats_policy": "old stats and checkpoints are historical; recompute stats before new training",
                  "motions_sha256": {r["motion_path"]: sha256_file(staging / r["motion_path"]) for r in rows}}
        write_json(staging / "reports/root_z_restoration.json", report)
        root.rename(backup)
        try:
            staging.rename(root)
        except BaseException:
            backup.rename(root)
            raise
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--kinematics", type=Path, required=True)
    parser.add_argument("--offset-reference", type=Path)
    parser.add_argument("--quality-config", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = restore(args.root, args.raw_root, args.kinematics, args.offset_reference, args.quality_config)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
