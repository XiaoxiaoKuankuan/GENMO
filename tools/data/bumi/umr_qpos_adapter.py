"""将 UMR qpos 清单和已验证自建库接入现有 BUMI 筛选、发布入口。

本模块只负责不同生产者的数据契约适配，不另设质量阈值。UMR 输入必须具有显式
关节名称、30 Hz 帧率、连续 frame_ids 和已核验的 MJCF；按名称重排后利用相同的
fe934 运动学计算 body 状态，再调用现有 evaluate_motion 执行三态质量门禁。自建
CSV 先由原有构建器完成原始配对审计、重采样及限位迁移，再接受同一 30 Hz 门禁。

发布仅消费完整报告中的 PASS，绑定源文件、选择清单、配置和资产 SHA256。公开库
沿用人体库的音乐特征及 split，自建库沿用已核验的 WAV/EDGE35 配对。UMR 的世界
地面为零，保留原始 Root Z；自建库保留原先 body-origin 地面语义，分别计算接触。
所有正式文件先写同盘 staging，成功后原子发布，失败时清理精确 staging 路径。
入口继续使用 filter_robot_retargeter_npz_motions.py 和对应 build 脚本的
--input-format umr-qpos 参数，不按日期或 checkpoint 复制运行脚本。
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from gem.datasets.music_dance.music_dance_bumi import BUMI_MUSIC_CONTRACT_VERSION
from gem.robots.bumi.contacts import BUMI_CONTACT_CONTRACT_VERSION, derive_bumi_foot_contact
from gem.robots.bumi.kinematics import BumiKinematics
from gem.robots.bumi.legacy_motion import (
    _NumpyCompatibleUnpickler,
    root_tilt_statistics,
    sha256_file,
)
from tools.data.bumi.build_bumi_music_dataset import (
    DATASET_SPECS,
    _aist_index,
    _human_jsonl_index,
    _materialize,
    _music_tensor,
    _relative_file,
    _write_json,
    _write_jsonl,
    pairing_fields,
)
from tools.data.bumi.filter_sonic_npz_motions import (
    _central_difference,
    build_summary,
    evaluate_motion,
    write_reports,
)

REPORT_VERSION = "genmo.bumi_quality_report.umr_and_mine_qpos30.v1"
SOURCE_VERSION = "umr.bumi3_qpos30.v1"
BUILDER_VERSION = "genmo.bumi_umr_and_mine_pass_builder.v1"
GROUND = "umr_foot_sole_ground_zero_v1"
SPECS = {**DATASET_SPECS, "mine": {"output": "Mine", "contract_name": "mine_bumi"}}
_WORKER_KIN = None


def verify_assets(config, robot_xml: Path, retarget_config: Path, kinematics: Path) -> dict:
    """复核同款机器人；UMR 运行配置使用自身指纹，不冒充其他重定向器。"""
    paths = {"robot_xml": robot_xml, "retarget_config": retarget_config, "kinematics": kinematics}
    actual = {k: sha256_file(v) for k, v in paths.items()}
    if actual["robot_xml"] != config.robot_xml_sha256:
        raise ValueError("UMR MJCF 与质量阈值绑定的资产不同")
    if actual["kinematics"] != config.kinematics_sha256:
        raise ValueError("UMR kinematics SHA 不匹配")
    kin = BumiKinematics(kinematics)
    if kin.source_mjcf_sha256 != actual["robot_xml"]:
        raise ValueError("运动学内部 MJCF 指纹不匹配")
    return {
        **{f"{k}_path": str(v.resolve()) for k, v in paths.items()},
        **{f"{k}_sha256": v for k, v in actual.items()},
    }


def load_umr_qpos(path: Path, kin: BumiKinematics, robot_xml: Path) -> tuple[torch.Tensor, dict]:
    """按显式关节名读取可信 UMR NPZ，并保持每帧原始位姿和时间线。"""
    # UMR 在 NumPy 2 环境中保存 object 名称数组；训练环境仍为 NumPy 1。
    # 仅对此可信元数据使用已有兼容 reader，不修改全局 numpy 模块或环境。
    with zipfile.ZipFile(path) as archive, archive.open("robot_joint_names.npy") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            _, _, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            _, _, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"不支持的关节名称 NPY 版本: {version}")
        names_value = _NumpyCompatibleUnpickler(handle).load() if dtype.hasobject else None
    with np.load(path, allow_pickle=False) as z:
        if names_value is None:
            names_value = z["robot_joint_names"]
        names = tuple(str(x) for x in names_value.tolist())
        qpos = np.asarray(z["qpos"], dtype=np.float32)
        fps = float(np.asarray(z["fps"]).reshape(-1)[0])
        frames = np.asarray(z["frame_ids"])
        source_xml = Path(str(z["robot_xml"].item())).resolve()
        meta = {
            "source_sequence_key": str(z["source_sequence_key"].item()),
            "source_data": str(z["source_data"].item()),
            "source_format": str(z["source_format"].item()),
            "source_robot_xml": str(source_xml),
        }
        if str(z["robot_name"].item()) != "bumi3" or meta["source_format"] != "smplx_npz":
            raise ValueError("UMR robot_name/source_format 不匹配")
    if source_xml != robot_xml.resolve():
        raise ValueError("UMR 逐条 robot_xml 与核验资产路径不一致")
    if len(names) != 21 or len(set(names)) != 21 or set(names) != set(kin.joint_order):
        raise ValueError("UMR 必须显式提供 21 个唯一且完整的关节名称")
    if fps != 30 or qpos.ndim != 2 or qpos.shape[1] != 28 or len(qpos) < 4:
        raise ValueError("UMR 必须为 30 Hz qpos[T,28] 且至少四帧")
    if not np.isfinite(qpos).all() or frames.shape != (len(qpos),):
        raise ValueError("UMR 非有限位姿或 frame_ids 维度错误")
    if not np.array_equal(frames, np.arange(len(qpos))):
        raise ValueError("UMR frame_ids 不是从零开始的完整连续时间线")
    quat_error = np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1)
    if np.max(quat_error) > 1e-3:
        raise ValueError("UMR wxyz 四元数不满足单位范数")
    reorder = [names.index(name) + 7 for name in kin.joint_order]
    value = torch.from_numpy(np.concatenate([qpos[:, :7], qpos[:, reorder]], axis=1))
    return value.contiguous(), meta


def source_path(row: dict, input_root: Path, mine_root: Path | None) -> Path:
    root = mine_root if row["dataset"] == "mine" else input_root
    if root is None:
        raise ValueError("mine 条目缺少 --mine-root")
    return _relative_file(root, row["source_relative_path"], row["sample_id"])


def load_source(
    row: dict, input_root: Path, mine_root: Path | None, kin: BumiKinematics, robot_xml: Path
) -> tuple[torch.Tensor, dict]:
    path = source_path(row, input_root, mine_root)
    if row["dataset"] != "mine":
        qpos, meta = load_umr_qpos(path, kin, robot_xml)
        sequence_key = meta["source_sequence_key"].removeprefix(row["dataset"] + "__")
        if sequence_key != row["sample_id"].split("/", 1)[1]:
            raise ValueError("UMR 清单文件名与 source_sequence_key 不一致")
        source_data = Path(meta["source_data"]).resolve()
        # 以减号开头的样本由上游复制成 dataset__sample 文件名；身份由人体NPZ元数据
        # 复核，不把 motions_keep 目录名误当成数据集，也不放宽来源/坐标检查。
        with np.load(source_data, allow_pickle=False) as human:
            if (str(human["dataset"].item()) != row["dataset"]
                    or str(human["sample_id"].item()) != sequence_key
                    or str(human["coordinate_system"].item()) != "right_handed_z_up_metric"
                    or float(human["fps"].item()) != 30.0
                    or int(human["num_frames"].item()) != len(qpos)):
                raise ValueError("UMR 源人体身份、Z-up坐标或帧数不一致")
        return qpos, meta
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        tuple(payload["joint_names"]) != kin.joint_order
        or payload["fps"] != 30
        or payload["source_mjcf_sha256"] != kin.source_mjcf_sha256
        or payload["quaternion_convention"] != "wxyz"
        or payload["ground_semantics"] != "legacy_body_origin_min_zero"
    ):
        raise ValueError("自建库资产、关节、帧率或地面契约不一致")
    qpos = torch.as_tensor(payload["qpos"]).float().contiguous()
    if qpos.ndim != 2 or qpos.shape[1] != 28 or len(qpos) < 4 or not torch.isfinite(qpos).all():
        raise ValueError("自建库 qpos 格式错误")
    return qpos, {"original_source_motion_sha256": payload["source_motion_sha256"]}


def qpos_arrays(qpos: torch.Tensor, kin: BumiKinematics, config) -> dict:
    """仅在内存中适配通用筛选器需要的具名 body/joint 状态。"""
    with torch.no_grad():
        fk = kin.forward_kinematics(qpos)
    bi = [kin.body_order.index(name) for name in config.body_order]
    ji = [kin.joint_order.index(name) + 7 for name in config.joint_order]
    body = fk["body_pos_w"][:, bi].numpy()
    quat = fk["body_quat_w"][:, bi].numpy()
    quat[:, 0] = qpos[:, 3:7].numpy()
    joints = qpos[:, ji].numpy()
    return {
        "joint_pos": joints,
        "joint_vel": _central_difference(joints, 30),
        "body_pos_w": body,
        "body_quat_w": quat,
        "body_lin_vel_w": _central_difference(body, 30),
    }


def input_rows(selection: Path, input_root: Path, mine_root: Path | None) -> list[dict]:
    selected = json.loads(selection.read_text())
    clips = selected["clips"]
    files = [r["file"] for r in clips]
    if (
        len(clips) != selected["n"]
        or files != selected["files"]
        or len(files) != len(set(files))
        or selected["paths"] != [str(input_root / f) for f in files]
    ):
        raise ValueError("UMR 选择清单条数、顺序、去重或路径不一致")
    rows = []
    for item in clips:
        ds, name = item["dataset"], item["file"]
        if ds not in DATASET_SPECS or Path(name).name != name or not name.endswith("_bumi3.npz"):
            raise ValueError("UMR 选择清单身份非法")
        sample = name.removesuffix("_bumi3.npz").removeprefix(ds + "__")
        rows.append(
            {
                "dataset": ds,
                "sample_id": f"{ds}/{sample}",
                "source_relative_path": name,
                "source_motion_contract_version": SOURCE_VERSION,
                "foot_spread_rank": item["rank"],
                "foot_spread_score_m": item["score_m"],
            }
        )
    if mine_root:
        for line in (mine_root / "manifests/train.jsonl").read_text().splitlines():
            item = json.loads(line)
            rows.append(
                {
                    "dataset": "mine",
                    "sample_id": f"mine/{item['sample_id']}",
                    "source_relative_path": item["motion_path"],
                    "source_motion_contract_version": "genmo.bumi_music.v1",
                }
            )
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("合并输入身份重复")
    return rows


def _init_worker(kinematics: str):
    global _WORKER_KIN
    torch.set_num_threads(1)
    _WORKER_KIN = BumiKinematics(kinematics)


def _evaluate(task):
    row, input_root, mine_root, robot_xml, config, config_sha = task
    base = {**row, "quality_config_sha256": config_sha}
    try:
        path = source_path(row, input_root, mine_root)
        base.update(source_sha256=sha256_file(path), source_bytes=path.stat().st_size)
        qpos, meta = load_source(row, input_root, mine_root, _WORKER_KIN, robot_xml)
        decision = evaluate_motion(qpos_arrays(qpos, _WORKER_KIN, config), config)
        return {**base, **decision, **meta, "report_contract_version": REPORT_VERSION}
    except Exception as exc:
        return {
            **base,
            "report_contract_version": REPORT_VERSION,
            "status": "REJECT",
            "status_without_joint_limit": "REJECT",
            "quality_accepted": False,
            "reason_codes": ["MOTION_CONTRACT_ERROR"],
            "reason_statuses": {"MOTION_CONTRACT_ERROR": "REJECT"},
            "metrics": {},
            "floor_intervals": [],
            "valid_intervals": [],
            "error_message": str(exc),
        }


def filter_main(args, config):
    if args.selection_manifest is None:
        raise ValueError("UMR 输入必须提供 --selection-manifest")
    root = args.input_root.resolve(strict=True)
    mine = args.mine_root.resolve(strict=True) if args.mine_root else None
    output = args.output_dir.resolve()
    if output == root or root in output.parents:
        raise ValueError("报告目录必须位于 UMR 源目录之外")
    assets = verify_assets(config, args.robot_xml, args.retarget_config, args.kinematics)
    rows = input_rows(args.selection_manifest, root, mine)
    if args.limit:
        rows = rows[: args.limit]
    config_sha = sha256_file(args.config)
    tasks = [(r, root, mine, args.robot_xml, config, config_sha) for r in rows]
    if args.workers == 1:
        _init_worker(str(args.kinematics))
        decisions = list(tqdm(map(_evaluate, tasks), total=len(tasks)))
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker, initargs=(str(args.kinematics),)
        ) as pool:
            decisions = list(tqdm(pool.map(_evaluate, tasks, chunksize=8), total=len(tasks)))
    summary = build_summary(
        decisions,
        input_root=root,
        config_path=args.config,
        config_sha256=config_sha,
        config=config,
        assets=assets,
        report_version=REPORT_VERSION,
        decision_scope="UMR 与自建库统一30Hz离线门禁；未进行动力学回放",
        compatibility_notes={
            "thresholds_reused_without_relaxation": True,
            "legacy_release_counts_not_applicable": True,
        },
    )
    summary.update(
        selection_manifest=str(args.selection_manifest.resolve()),
        selection_manifest_sha256=sha256_file(args.selection_manifest),
        mine_root=str(mine) if mine else None,
        partial_scan=args.limit is not None,
        source_bytes=sum(r.get("source_bytes", 0) for r in decisions),
        source_motion_contract_version="umr_qpos30_and_verified_mine_v1",
    )
    write_reports(output, decisions, summary, args.config, overwrite=args.overwrite)
    _write_jsonl(output / "merged_input_manifest.jsonl", rows)
    print(
        json.dumps(
            {
                "report_dir": str(output),
                "sequences": len(rows),
                "status_counts": summary["status_counts"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _references(human_roots: dict, audio_roots: dict, mine_root: Path | None):
    result = {}
    for ds, root in human_roots.items():
        index = _aist_index(root) if ds == "aistpp" else _human_jsonl_index(root)
        for sid, row in index.items():
            row.update(pairing_fields(ds, sid, row))
            row["_feature"] = str(_relative_file(root, row["music_feature_path"], sid))
            row["_audio"] = str(audio_roots[ds] / f"{row['audio_key']}.wav")
        result[ds] = index
    if mine_root:
        index = {}
        for line in (mine_root / "manifests/train.jsonl").read_text().splitlines():
            row = json.loads(line)
            row["_feature"] = str(
                _relative_file(mine_root, row["music_feature_path"], row["sample_id"])
            )
            row["_audio"] = str(_relative_file(mine_root, row["audio_path"], row["sample_id"]))
            index[row["sample_id"]] = row
        result["mine"] = index
    return result


def build_main(args, human_roots: dict, audio_roots: dict) -> dict:
    """以统一三态报告为白名单发布多库；训练 reader 验收前不发布 staging。"""
    from gem.datasets.music_dance.music_dance_bumi import BumiMusicDatasetReader
    from tools.data.bumi.filter_robot_retargeter_npz_motions import load_config

    torch.set_num_threads(1)
    root = args.source_root.resolve(strict=True)
    mine = args.mine_root.resolve(strict=True) if args.mine_root else None
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有正式数据: {output}")
    config = load_config(args.quality_config)
    assets = verify_assets(config, args.robot_xml, args.retarget_config, args.kinematics)
    kin = BumiKinematics(args.kinematics)
    summary = json.loads(args.quality_summary.read_text())
    quality_sha, config_sha = sha256_file(args.quality_report), sha256_file(args.quality_config)
    if (
        summary.get("report_contract_version") != REPORT_VERSION
        or summary.get("partial_scan")
        or summary.get("quality_config_sha256") != config_sha
        or summary.get("mine_root") != (str(mine) if mine else None)
    ):
        raise ValueError("质量报告版本、完整性、配置或自建库来源不一致")
    if sha256_file(Path(summary["selection_manifest"])) != summary["selection_manifest_sha256"]:
        raise ValueError("选择清单在筛选后发生变化")
    rows = [json.loads(x) for x in args.quality_report.read_text().splitlines() if x.strip()]
    expected = input_rows(Path(summary["selection_manifest"]), root, mine)
    if (
        len(rows) != summary["sequences"]
        or len({r["sample_id"] for r in rows}) != len(rows)
        or {r["sample_id"] for r in rows} != {r["sample_id"] for r in expected}
    ):
        raise ValueError("质量报告不完整或包含重复/额外样本")
    for row in rows:
        if (
            row.get("report_contract_version") != REPORT_VERSION
            or row.get("quality_config_sha256") != config_sha
            or (row["status"] == "PASS") != (row["quality_accepted"] is True)
        ):
            raise ValueError("逐条质量报告契约不一致")
    accepted = [r for r in rows if r["status"] == "PASS"]
    if len(accepted) != summary["quality_accepted_sequences"] or not accepted:
        raise ValueError("PASS 数量与汇总不一致或为空")
    if args.expected_pass is not None and len(accepted) != args.expected_pass:
        raise ValueError("PASS 数量不符合预期")
    references = _references(human_roots, audio_roots, mine)
    datasets = sorted({r["dataset"] for r in rows})
    mine_info = json.loads((mine / "meta/dataset_info.json").read_text()) if mine else {}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    manifests = {d: {s: [] for s in ("train", "val", "test")} for d in datasets}
    tilts = {d: [] for d in datasets}
    cache = {}

    def digest(path):
        if path not in cache:
            cache[path] = sha256_file(path)
        return cache[path]

    def semantics(ds):
        return "legacy_body_origin_min_zero" if ds == "mine" else GROUND

    def retarget_sha(ds):
        return (
            mine_info["retarget_config_sha256"]
            if ds == "mine"
            else assets["retarget_config_sha256"]
        )

    try:
        for row in tqdm(accepted, desc="publish UMR/Mine PASS"):
            ds, sid = row["sample_id"].split("/", 1)
            src = source_path(row, root, mine)
            if digest(src) != row["source_sha256"]:
                raise ValueError(f"{sid}: 筛选后源动作 SHA 改变")
            qpos, source_meta = load_source(row, root, mine, kin, args.robot_xml)
            reference = references[ds][sid]
            music, audio = Path(reference["_feature"]), Path(reference["_audio"])
            if not audio.is_file() or len(qpos) != reference["num_frames"]:
                raise ValueError(f"{ds}/{sid}: WAV 缺失或动作长度与配对不一致")
            if len(_music_tensor(music, sid)) != len(qpos):
                raise ValueError(f"{ds}/{sid}: EDGE35 与动作不等长")
            contact = derive_bumi_foot_contact(
                qpos,
                kin,
                fps=30,
                valid_mask=torch.ones(len(qpos), dtype=torch.bool),
                ground_height=None if ds == "mine" else 0.0,
                estimate_ground_mask=torch.tensor(ds == "mine"),
            )
            dst = staging / SPECS[ds]["output"]
            motion_rel = Path("motions") / f"{sid}.pt"
            music_rel, audio_rel = Path("musicfeat_v2") / music.name, Path("audio") / audio.name
            payload = {
                "contract_version": BUMI_MUSIC_CONTRACT_VERSION,
                "source_motion_contract_version": row["source_motion_contract_version"],
                "qpos": qpos,
                "fps": 30,
                "robot_name": "bumi",
                "joint_names": list(kin.joint_order),
                "quaternion_convention": "wxyz",
                "qpos_order": "mujoco_native",
                "source_dataset": ds,
                "source_sample_id": sid,
                "source_motion_sha256": digest(src),
                "source_mjcf_sha256": kin.source_mjcf_sha256,
                "retarget_config_sha256": retarget_sha(ds),
                "quality_config_sha256": config_sha,
                "quality_report_sha256": quality_sha,
                "quality_accepted": True,
                "retarget_quality": {"status": "PASS", "reason_codes": row["reason_codes"]},
                "ground_semantics": semantics(ds),
                "root_z_adjusted": ds == "mine",
                "root_z_second_adjustment_applied": False,
                "foot_contact": contact.contact.contiguous(),
                "foot_contact_contract_version": BUMI_CONTACT_CONTRACT_VERSION,
                "foot_contact_ground_height_m": float(contact.ground_height),
                "source_metadata": source_meta,
            }
            (dst / "motions").mkdir(parents=True, exist_ok=True)
            torch.save(payload, dst / motion_rel)
            _materialize(music, dst / music_rel)
            _materialize(audio, dst / audio_rel)
            public_row = {k: v for k, v in reference.items() if not k.startswith("_")}
            public_row.update(
                dataset=SPECS[ds]["contract_name"],
                sample_id=sid,
                fps=30,
                motion_path=motion_rel.as_posix(),
                music_feature_path=music_rel.as_posix(),
                audio_path=audio_rel.as_posix(),
                quality_accepted=True,
                source_motion_sha256=digest(src),
                source_music_feature_sha256=digest(music),
                source_audio_sha256=digest(audio),
            )
            manifests[ds][reference["split"]].append(public_row)
            quat = qpos[:, 3:7].numpy()
            tilts[ds].append(
                np.degrees(np.arccos(np.clip(1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), -1, 1)))
            )
        counts = {}
        for ds, splits in manifests.items():
            dst = staging / SPECS[ds]["output"]
            n = sum(len(v) for v in splits.values())
            if not n:
                raise ValueError(f"{ds}: 严格 PASS 后无数据")
            for split, values in splits.items():
                _write_jsonl(
                    dst / "manifests" / f"{split}.jsonl",
                    sorted(values, key=lambda r: r["sample_id"]),
                )
            info = {
                "contract_version": BUMI_MUSIC_CONTRACT_VERSION,
                "builder_contract_version": BUILDER_VERSION,
                "robot_name": "bumi",
                "dataset_name": SPECS[ds]["contract_name"],
                "source_dataset": ds,
                "fps": 30,
                "qpos_dim": 28,
                "joint_dim": 21,
                "joint_names": list(kin.joint_order),
                "quaternion_convention": "wxyz",
                "qpos_order": "mujoco_native",
                "quality_filter_applied": True,
                "quality_acceptance_policy": "PASS_ONLY",
                "reader_joint_limit_tolerance_rad": config.joint_limit_violation_max,
                "mjcf_sha256": kin.source_mjcf_sha256,
                "source_mjcf_sha256": kin.source_mjcf_sha256,
                "kinematics_sha256": kin.kinematics_sha256,
                "retarget_config_sha256": retarget_sha(ds),
                "quality_config_sha256": config_sha,
                "quality_report_sha256": quality_sha,
                "ground_semantics": semantics(ds),
                "root_z_adjusted": ds == "mine",
                "root_orientation_gate": {
                    "scope": "per_dataset_all_frames",
                    "all_sequences_recomputed_and_dataset_passed": True,
                    "statistics": root_tilt_statistics(np.concatenate(tilts[ds])),
                },
                "split_counts": {s: len(v) for s, v in splits.items()},
            }
            _write_json(dst / "meta/dataset_info.json", info)
            for split, values in splits.items():
                if values:
                    BumiMusicDatasetReader(
                        root=dst,
                        dataset_name=SPECS[ds]["contract_name"],
                        split=split,
                        kinematics=kin,
                        joint_limit_tolerance=config.joint_limit_violation_max,
                        validate_source_hashes_on_init=True,
                    )
            counts[ds] = {
                "n": n,
                "frames": sum(r["num_frames"] for v in splits.values() for r in v),
                "splits": {s: len(v) for s, v in splits.items()},
            }
        report = {
            "status": "passed",
            "contract_version": BUILDER_VERSION,
            "quality_input_sequences": len(rows),
            "total_pass_sequences": len(accepted),
            "quality_status_counts": summary["status_counts"],
            "datasets": counts,
            "quality_report_sha256": quality_sha,
            "quality_config_sha256": config_sha,
            "verified_assets": assets,
            "source_files_modified": False,
            "root_z_second_adjustment_applied": False,
        }
        _write_json(staging / "conversion_report.json", report)
        os.replace(staging, output)
        return report
    except Exception:
        shutil.rmtree(staging)
        raise
